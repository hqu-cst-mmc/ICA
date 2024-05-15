# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcherHOI(nn.Module):

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1):
        super().__init__()
        self.cost_obj_class = cost_obj_class    # 代价权重init
        self.cost_verb_class = cost_verb_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # self.cost_verb_score = 1.0
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    # 计算匹配代价时，不计算梯度
    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2]      # 取出bs和num_queries

        # 将一个batch的预测结果前两位展平(bs,num_q,dim)->(bs x num_q,dim)
        out_interactiveness = outputs['pred_interactiveness'].flatten(0, 1)
        out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)     # 对最后一维特征维作softmax计算
        out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()     # 动作类为多类分类问题 作sigmoid计算
        out_verb_prob = out_verb_prob * out_interactiveness.sigmoid()
        out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
        out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

        # 通过concat获得一个batch中所有真值的物体类、动作类、人框、物体框
        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
        tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
        tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])

        # 计算一个batch下物体类损失，得到形状为(bs x num_q, tgt_obj_labels_num)
        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]

        # 计算一个batch下动作类损失，形状跟上面一样
        tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)     # 将第一维动作类总数转置到前面
        cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                            (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                            ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2

        # 计算框间的L1损失
        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        if cost_sub_bbox.shape[1] == 0:
            cost_bbox = cost_sub_bbox
        else:
            cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

        # 计算框间的GIOU损失
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        if cost_sub_giou.shape[1] == 0:
            cost_giou = cost_sub_giou
        else:
            cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

        # 得到最终的匹配代价矩阵
        C = self.cost_obj_class * cost_obj_class + self.cost_verb_class * cost_verb_class + \
            self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v['obj_labels']) for v in targets]     # batch中每个sample中物体类的个数
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]   # 相当于选择每个样本的sample与target的相似度矩阵进行二分匹配，获得索引集合
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]     # 长度为batchsize的元组list，list中每个元组包括一个样本中(所有匹配上的预测索引，与跟其匹配的真值索引)


def build_matcher(args):
    # hoi匹配代价，传入四部分匹配代价的权重超参数，物体类代价、动作类代价、框回归代价、框GIOU代价
    return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, cost_verb_class=args.set_cost_verb_class,
                               cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
