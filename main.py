# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate_hoi
from models import build_model
from args import build_args


def main(args):
    utils.init_distributed_mode(args)       # 分布式初始化，判断是否能进行分布式训练
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:     # 分割时，限制部分权重的训练,同时给定masks
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)      # 获取GPU，构造device对象

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()     # 设置一个数字用于当作固定的随机种子数
    # 下面三个设置随机数种子，固定每一次训练的初始化参数结果，让每次的模型相同且结果可复现
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)    # 模型初始化，得到模型、损失、后处理函数(转换最后的输出结果为符合数据集的格式，用于计算map)
    model.to(device)    # 把模型转移到GPU上，在GPU上训练

    # ddp(DistributedDataParallel的缩写)
    model_without_ddp = model
    if args.distributed:    # 当采用分布式时，将模型采用分布式的手段
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # 统计模型中所有可训练的参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # 下面这部分包括设置优化器，学习率策略
    # 将backbone部分的参数与其他参数分开，以使用不同的学习率
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 构建训练、验证数据集
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # 根据是否有分布式，构建数据采样器
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)   # DistributedSampler()每一个子进程划分出一部分数据集，以避免不同进程之间数据重复。而不是如dataparallel直接把每个batch切分到每个子进程
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 构造每个batch的数据采样   BatchSampler()将得到的数据打包起来，每次迭代返回batch size大小的index列表
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # 将上面的数据采样器装在DataLoader上，以进行迭代batch训练和验证
    # 使用collate_fn方法来重新组装一个batch的数据
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        # 类似迁移学习的微调，固定住权重，仅训练分割头(只在分割任务中使用)
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
        del checkpoint

    output_dir = Path(args.output_dir)
    # 下面这部分表示当从历史的某个训练阶段恢复过来(包括检查点或预训练文件)，加载当时的模型权重、优化器、学习率等参数
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    elif args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        del checkpoint
    # 当设置了评估时，仅进行测试不进行训练
    if args.eval:
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device)
        return

    # 开始正式训练，每个周期后根据学习策略调整学习率
    print("Start training")
    best_map = 0.0
    start_time = time.time()    # 记录开始训练时间
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:    # 当有分布式时，调整每个epoch的分布式采样
            sampler_train.set_epoch(epoch)
        # 获取一个周期的训练结果
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()     # 根据学习策略调整学习率
        # 当设置了output_dir后，将训练结果和相关参数记录到指定文件中
        if args.output_dir:
            # 每个epoch训练完后创建当前检查点文件保存路径
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            # 在第100轮和lr衰退之前添加额外的检查点，以记录相关参数
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            # 生成检查点文件保存模型等各种参数
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # 每个epoch训练后完，进行评估
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device)

        # 对训练和验证结果加上字符串前缀，注明epoch次数和参数数量
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        map_all = float(log_stats['test_mAP_all'])
        if map_all > best_map:
            best_map = map_all
            checkpoint_path = output_dir / 'best.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        if args.output_dir and utils.is_main_process():
            # 将训练和验证结果写入到(分布式)主节点指定的文件中
            # 创建log.txt文件并将结果写入
            if args.dataset_file == 'vcoco':
                with (output_dir / "log_vcoco.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                with (output_dir / "log_hico.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))    # 输出整个训练的总时长


if __name__ == '__main__':
    args = build_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

