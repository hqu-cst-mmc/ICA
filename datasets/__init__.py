# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch.utils.data
import torchvision

from .hico import build as build_hico
from .vcoco import build as build_vcoco


def build_dataset(image_set, args):
    # 根据不同的数据集名，构造不同的数据集
    # image_set表示是构造train和val数据集
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
