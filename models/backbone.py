# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):       # 使用FrozenBN2D 代替 BN，是因为每个batch太小，使得批量统计很差并降低了性能
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()      # rsqrt()取平方根的倒数
        bias = b - rm * scale
        return x * scale + bias     # 最终仿射的计算公式


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():     # 遍历backbone的层和参数
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)     # 当不对Backbone进行训练或不存在层时，不记录参数梯度即不训练
        if return_interm_layers:    # 返回中间层
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # 用于对backbone的输出特征图转换为输入进encoder的维度
            self.strides = [4, 8, 16, 32]
            self.num_channels = [256, 512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [num_channels]
        # IntermediateLayerGetter从模型中设置输出的层，根据前面得到的中间层(return_layers)dict，对backbone中被选择的层重命名，并最终从return_layers含有的中间层取出输出
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # F.interpolate() 实现插值和上采样，参数有:
            #           input:输入张量、size:输出大小、scale_factor:指定输出倍数、
            #           mode:上采样算法(最近邻、线性(3D-only)、双线性, 双三次(bicubic,4D-only)和三线性(trilinear,5D-only)插值算法和area算法),默认为nearest、
            #           align_corners:布尔值，当为True,输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值；当为False时，输入和输出张量由它们的角像素的角点对齐，插值使用边界外值的边值填充
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out      # 最终返回backbone输出


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # getattr() 根据属性名或方法名从目标类中获取属性或方法； 这里使用的话是根据backbone的名字从torchvision.models中获取对应models的实例化方法，后面括号中是传递需要的参数
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)     # pretrained参数根据是否为主进程判断是否初始化预训练好的backbone参数
                                                                        # FrozenBatchNorm2d 这边为自定义的BN类，它的仿射(affine)中的参数weight和bias是固定的，且BN计算中的批量统计均值和方差也是固定的。
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)      # 对父类BackboneBase进行init
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        # 用于对backbone的输出特征图转换为输入进encoder的维度
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)      # 初始化位置编码
    train_backbone = args.lr_backbone > 0       # 是否固定训练好的主干参数，如果有lr则主干参与训练
    return_interm_layers = True       # 返回中间层,为True时返回1, 2，3，4层(resnet中第0层为提取特征层，后4层为4组残差块输出)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)     # args.dilation值为bool，判断是否将最后一卷积块内的步幅换成空洞(设置卷积层时，存在一个参数空洞率，默认空洞率=1级普通卷积，当空洞率不为1时可看成空洞卷积，对卷积核填充空洞，作用时：增大感受野，对大物体分割有用。可用来代替步幅取得同样的输出大小)
    model = Joiner(backbone, position_embedding)    # 返回每个中间层的输出，和对应的pos编码
    model.num_channels = backbone.num_channels      # 得到主干的通道数
    return model
