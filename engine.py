# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # 模型设置为训练模式
    model.train()
    criterion.train()
    # 使用了MetricLogger类，主要用于log输出，在一定次数的batch和每次epoch结束后，在终端打印log，包括训练结果
    # MetricLogger类中使用了defaultdict来记录训练过程中各种数据(这些数据为SmoothedValue类型，通过指定窗口大小，来存储数据的历史步长，如1表示每次新的覆盖旧的)的历史值
    # 另外SmoothedValue还实现了统计中位数、均值等方法，并且能够在各进程间通信
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))    # 通过key 添加SmoothedValue类型的数据
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # MetricLogger类最主要的就是实现了一个log_every()方法
    # log_every方法是一个生成器，每一次for循环会yield一个batch的数据，并且方法暂停在那里
    # 待这个for循环完成一次迭代(即模型训练完一个迭代)，到下一次循环时
    # log_every方法内部才会继续执行其内容，待其中内容执行完毕后，继续yield下一个batch的数据
    # 同时暂定在那里，待模型训练下一个迭代.....依次重复这个过程，直至所有batch均处理完毕
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # 将一个batch的数据转移到gpu上
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 得到qpic模型的输出(中间层和最后一层)
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)     # 匹配和计算各部分损失
        weight_dict = criterion.weight_dict
        # 计算各部分损失的权重之和
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # reduce_dict()将loss在各个进程间同步，并返回同样类型是dict的已经同步过的loss数据，默认是总和/进程总数
        # 避免使用分布式时，loss不同步
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}      # 对同步后的各部分损失乘以对应权重
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())      # 计算同步过后各部分损失的权重之和

        loss_value = losses_reduced_scaled.item()   # 同步后的最终损失值

        # 若loss溢出(梯度爆炸时)，退出训练
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 避免梯度爆炸，进行梯度裁剪
        optimizer.zero_grad()
        losses.backward()   # 对未同步的总损失进行反向传播
        # 在backward得到梯度后，如果有设置梯度裁剪的参数，即先进行梯度裁剪，再网络更新
        if max_norm > 0:
            # 默认采用第二范式进行裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # 往metric_logger添加保存的dict数据，用于打印log
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # print_dict = {k: v for k, v in loss_dict_reduced_scaled.items() if 'interactiveness' in k}
        # metric_logger.update(loss=loss_value, **print_dict)
        # hasattr()查看该类中，是否存在该属性或方法，是则返回True
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            # 更新dict中obj_class_error的值
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])    # 更新dict中lr值

        del samples
        del targets
        del outputs
        del loss_dict
        del loss_dict_reduced
        del loss_dict_reduced_unscaled
        del weight_dict
        del losses
        del losses_reduced_scaled

    # 当所有batch处理完后，收集所有进程上同步的各种数据
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # global_avg是SmoothedValue类里的属性方法(用@property修饰)
    # 返回的值是各个进程间同步后的历史均值
    # 比方说对于loss这项数据，在训练中被计算了n次
    # 那么历史均值就是，这n次的总和在进程间同步后除以n
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device):
    # 大体与上面的训练过程类似，但没有匹配与损失
    # 进行eval模式
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)     # 对预测输出进行后处理，恢复原图片中的形式

        # all_gather()对最终输出返回为各个进程聚合的形式，返回值为list[data]形式
        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        # 对每个batch的GT表示为各个进程的形式，返回值为List[data]，并使用deepcopy避免runtime error
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

    # gather the stats from all processes
    # 当所有batch处理完后，收集所有进程上同步的各种数据
    metric_logger.synchronize_between_processes()

    # 遍历所有预测和GT，确保所有输出都在图片中
    img_ids = [img_gts['id'] for img_gts in gts]    # 所有图片的索引
    _, indices = np.unique(img_ids, return_index=True)  # numpy.unique()去除重复的元素，并进行排序
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    # 根据不同数据集采用不同评估策略
    if dataset_file == 'hico':
        # 初始化评估类，并对预测输出和GT进行处理
        evaluator = HICOEvaluator(preds, gts, subject_category_id, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat)

    # 调用evaluate方法,返回map, map rare, map no-rare, mean max recall四个值
    # 评估中使用阈值排除置信度不够的预测，并根据score(物体类最大 x 动作值)，选择每一hoi实例得分最大的预测作为最终预测，再跟GT计算map等值
    stats = evaluator.evaluate()

    del samples
    del targets
    del outputs

    torch.cuda.empty_cache()

    return stats
