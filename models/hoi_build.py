# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .hoi import (DETRHOI, SetCriterionHOI, PostProcessHOI)
from .transformer import build_transformer


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)     # 初始化主干

    transformer = build_transformer(args)   # 初始化transformer

    model = DETRHOI(
        backbone,
        transformer,
        num_obj_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        num_decoder_layers=args.dec_layers,
        num_feature_levels=4,
        aux_loss=args.aux_loss,
    )

    matcher = build_matcher(args)   # 初始化匹配算法
    weight_dict = {}
    # 获取计算hoi损失时的权重超参数
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    weight_dict['loss_heatmap'] = 1.0
    # TODO this is a hack
    if args.aux_loss:   # 当要计算辅助损失时，也设置用于辅助损失计算的各层的对应超参数
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # HOI损失，包含四部分（物体类损失obj_labels、动作类损失verb_labels、框回归L1损失和GIOU损失sub_obj_boxes、以及obj_cardinality损失）
    # obj_cardinality损失：计算物体类预测的数量与GT数量的L1误差，仅当作衡量性能的指标，不参与反向梯度计算
    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
    # 初始化损失函数       verb_loss_type表示计算动作类损失时的损失算法    eos_coef表示对象类中无对象类的相对分类权重
    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                verb_loss_type=args.verb_loss_type)
    criterion.to(device)    # 损失函数保存至GPU上
    # 初始化HOI后处理函数(并不是进行NMS等操作，而是将输出转化为 数据集 对应的格式)
    postprocessors = {'hoi': PostProcessHOI(args.subject_category_id)}

    return model, criterion, postprocessors
