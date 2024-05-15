# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)
import copy
import math


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DETRHOI(nn.Module):  # hoi模型

    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, num_decoder_layers=6, num_feature_levels=4,
                 aux_loss=False,
                 ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1)
        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes+1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # init bbox_embed
        nn.init.constant_(self.sub_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.sub_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.obj_bbox_embed.layers[-1].bias.data, 0)

        self.backbone = backbone
        self.aux_loss = aux_loss

        self.obj_class_embed = _get_clones(self.obj_class_embed, num_decoder_layers)
        self.verb_class_embed = _get_clones(self.verb_class_embed, num_decoder_layers)
        self.sub_bbox_embed = _get_clones(self.sub_bbox_embed, num_decoder_layers)
        self.obj_bbox_embed = _get_clones(self.obj_bbox_embed, num_decoder_layers)

    def forward(self, samples: NestedTensor):
        # 判断输入进来的batch图像是否为NestedTensor这一类型的
        # 如果为False,就对samples进行统一尺寸大小操作
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        # transformer最终输出五个：detector and classifier各层输出[0], 每层的先验人的的框和物体框 和 与输入特征图shape一样的memory[1]
        hs, interactiveness, ht, _ = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[:]
        outputs_interactiveness = []
        outputs_sub_coords = []
        outputs_obj_coords = []
        outputs_obj_class = []
        outputs_verb_class = []
        # 每个中间层用不同的预测头部
        # 多个FFN，分别得到物体类、动词类、人框、物体框
        # 前向计算时(中间层数，bs,num_queries,hidden_dim)x(hidden_dim,num_classes)->(中间层数，bs,num_queries,num_classes)
        for lvl in range(hs.shape[0]):
            outputs_sub_coords.append(self.sub_bbox_embed[lvl](hs[lvl]).sigmoid())
            outputs_obj_coords.append(self.obj_bbox_embed[lvl](hs[lvl]).sigmoid())
            outputs_obj_class.append(self.obj_class_embed[lvl](hs[lvl]))
            outputs_verb_class.append(self.verb_class_embed[lvl](hs[lvl]))
            outputs_interactiveness.append(interactiveness[lvl])

        outputs_interactiveness = torch.stack(outputs_interactiveness)
        outputs_sub_coords = torch.stack(outputs_sub_coords)
        outputs_obj_coords = torch.stack(outputs_obj_coords)
        outputs_obj_class = torch.stack(outputs_obj_class)
        outputs_verb_class = torch.stack(outputs_verb_class)

        # 取decoder最后一层输出
        out = {'ratio': mask, 'src_heatmap': ht,
               'pred_interactiveness': outputs_interactiveness[-1],
               'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coords[-1], 'pred_obj_boxes': outputs_obj_coords[-1]}
        # 当辅助损失为真时，存储decoder除最后一层外中间层输出
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_interactiveness,
                                                    outputs_obj_class, outputs_verb_class,
                                                    outputs_sub_coords, outputs_obj_coords)
        return out

    # 保存中间各层的输出，用于辅助解码损失计算
    @torch.jit.unused
    def _set_aux_loss(self, outputs_interactiveness, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_interactiveness': a, 'pred_obj_logits': b, 'pred_verb_logits': c, 'pred_sub_boxes': d, 'pred_obj_boxes': e}
                for a, b, c, d, e in zip(outputs_interactiveness[:-1],
                                            outputs_obj_class[:-1], outputs_verb_class[:-1],
                                            outputs_sub_coord[:-1], outputs_obj_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    # 简单的多层感知机，用于返回框坐标
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 最后一层不用relu激活
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses,
                 verb_loss_type):
        super().__init__()

        assert verb_loss_type == 'bce' or verb_loss_type == 'focal'

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        # 设置在物体类loss中，各物体类的权重为1，非物体类的权重由传进来的参数指定
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef

        # 将这部分注册到buffer，能够被static_dict记录且不会有梯度传播到此处，可看作在内存中定一个常量，模型保存和加载时可以写入和读出
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type

    # 计算物体类损失
    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        # 最终层输出的物体类
        src_logits = outputs['pred_obj_logits']

        # 得到一个tuple,第一个元素是各个匹配object的batch index，第二个元素是各个匹配object的query index
        idx = self._get_src_permutation_idx(indices)
        # 获得batch所有匹配的真值的物体类索引
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        # 生成一个大小为(bs,num_queries)，值全为num_obj_classes的二维    表示初始化一个batch中所有query的物体类匹配结果，都为背景类
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 匹配的预测索引对应的值置为GT的索引
        target_classes[idx] = target_classes_o

        # 下面为计算celoss
        # 因为celoss需要第一维对应类别数， 对物体类进行转置 (bs,num_query,num_object) -> (bs,num_object,num_query)
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_obj_ce': loss_obj_ce}

        # 当log为true时，计算top-1精度(即预测概率最大的类别与匹配的GT的类别是否一致)，结果是百分数，这部分仅用于log，不参与模型训练
        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # 计算obj_cardinality损失：计算物体类预测的数量与GT数量的L1误差，仅当作衡量性能的指标，不参与反向梯度计算
    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    # 计算动作类损失，与计算物体类损失类似
    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']
        interactiveness = outputs['pred_interactiveness']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        if self.verb_loss_type == 'bce':
            # F.binary_cross_entropy_with_logits = sigmoid + F.binary_cross_entropy
            # loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
            src_logits = src_logits.sigmoid()
            loss_verb_ce = F.binary_cross_entropy(src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            src_logits = src_logits * interactiveness.sigmoid()
            loss_verb_ce = self._neg_loss(src_logits, target_classes)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses

    # 计算框的损失，包括框的回归L1损失和GIOU损失
    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        # 得到(batch indices, query indices)
        idx = self._get_src_permutation_idx(indices)
        # 获取匹配的预测的人框和物体框坐标   src_sub_boxes的shape为(num_query_index,4)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        # 获取匹配的GT的人框和物体框坐标
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # any(dim=1)表示对每行查看是否满足条件(target_obj_boxes != 0),如果是则返回True
        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)  # exist_obj_boxes的shape为(匹配GT物体框的数量)

        losses = {}
        if src_sub_boxes.shape[0] == 0:  # 如果不存在匹配的预测query，则L1损失和GIOU损失为匹配的预测框之和
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            # L1损失的reduction参数不设置时，默认为mean,即返回所有涉及误差计算的元素的均值
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions  # num_interactions 为一个batch中交互的数量，损失为其均值
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                        exist_obj_boxes.sum() + 1e-4)  # exist_obj_boxes.sum()为一个batch中GT有效的物体框的数量
            # 使用torch.diag()获取对角线元素
            # 由于generalized_box_iou得到的是所有匹配的预测跟所有GT的GIOU值 shape为(num_query_index,num_gtbox)
            # 而因为前面matcher中最后得到的索引集合已经是一一对应的匹配索引，所以这边拿到的框也都是一一对应匹配的
            # 因此所要计算的GIOU loss 为对角线上的元素
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    # 计算匹配上的交互性损失
    def loss_heatmap(self, outputs, targets):
        src_heatmaps = outputs['src_heatmap'].sigmoid()
        targets_heatmaps = self.heatmap_generate(outputs['ratio'], targets)
        # focal loss
        loss_heatmap = self._neg_loss(src_heatmaps, targets_heatmaps)
        losses = {'loss_heatmap': loss_heatmap}
        return losses

    def _neg_loss(self, pred, gt):
        # focal loss 计算
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def _get_src_permutation_idx(self, indices):
        # 一个batch所有的匹配索引集合 -> 一个元组包括所有匹配的预测值的batch index(batch中第几个图像)和query index(预测值属于第几个query对象)
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        # 根据指定的loss类型，调用对应的计算损失方法
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        # HOI模型最终输出为一个dict，包含经过FFN后的中间层输出与最后一层输出，这边只取得最后一层的FFN输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 将最后一层输出与target经过matcher的前向计算后，得到长度为bs的匹配索引集合List
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # 计算一个batch中的真值交互instance数量，当使用了分布式时，在所有分布式节点之间同步
        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        # 计算所有需要的损失
        losses = {}
        losses.update(self.loss_heatmap(outputs, targets))
        for loss in self.losses:
            # 根据前面要求的四部分损失，分别计算并保存至dict losses中
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # 若要计算辅助损失，就重复跟上面一样的步骤
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # 循环中间层输出，并进行匹配得到索引集合
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        # 对于辅助损失计算时，不去记录top-1精度，只保存最后一层输出的top-1精度
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    # 分别保存每层的四部分损失，如：loss_obj_giou_1:...
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses  # 最终返回各层损失的dict

    def heatmap_generate(self, ratios, targets):
        targets_heatmaps = []
        for ratio, target in zip(ratios, targets):
            ratio_img_h = (ratio[:, 0] == False).sum(dim=0)
            ratio_img_w = (ratio[0] == False).sum(dim=0)
            ratio_img_size = torch.tensor([ratio_img_w, ratio_img_h], device=ratio.device)
            sub_center = target['sub_boxes'][..., :2] * ratio_img_size
            obj_center = target['obj_boxes'][..., :2] * ratio_img_size
            ho_center = (sub_center+obj_center)/2
            s_r = torch.min(target['sub_boxes'][..., 2:]*ratio_img_size, dim=-1)[0]
            s_r = s_r/2
            o_r = torch.min(target['obj_boxes'][..., 2:]*ratio_img_size, dim=-1)[0]
            o_r = o_r/2
            ho_r = torch.sum((sub_center-obj_center)**2, dim=-1)
            ho_r = torch.sqrt(ho_r)/2
            center = torch.cat((sub_center, obj_center, ho_center), dim=0)
            r = torch.cat((s_r, o_r, ho_r), dim=0)

            x = torch.arange(ratio_img_w, device=ratio_img_w.device)
            y = torch.arange(ratio_img_h, device=ratio_img_h.device)
            gaussian_x = (x-center[..., 0, None])**2
            gaussian_y = (y-center[..., 1, None])**2
            gaussian_2d = torch.exp(-(gaussian_x.unsqueeze(1)+gaussian_y.unsqueeze(2)) /
                                    2*(r.view(-1, 1, 1).repeat(1, ratio_img_h, ratio_img_w)/3)**2)
            heatmap = torch.zeros((ratio_img_h, ratio_img_w), device=ratio.device)
            for g in gaussian_2d:
                heatmap = torch.where(g > heatmap, g, heatmap)
            pad_heatmap = torch.zeros_like(ratio).float()
            pad_heatmap[:ratio_img_h, :ratio_img_w] = heatmap
            targets_heatmaps.append(pad_heatmap)

        return torch.stack(targets_heatmaps)


# 对qpic模型的输出outputs转化成数据集对应的格式，方便进行数据集的test
class PostProcessHOI(nn.Module):

    def __init__(self, subject_category_id):
        super().__init__()
        self.subject_category_id = subject_category_id

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_interactiveness, out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_interactiveness'], \
                                                                                             outputs['pred_obj_logits'], \
                                                                                             outputs['pred_verb_logits'], \
                                                                                             outputs['pred_sub_boxes'], \
                                                                                             outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        # 数据集中的物体类评估不包含背景类，因此在生成预测结果时去掉最后一类
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()
        verb_scores = verb_scores * out_interactiveness.sigmoid()

        img_h, img_w = target_sizes.unbind(1)  # 拿到图像的高宽
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]  # 因为最终模型输出的框的坐标已经经过归一化在[0,1]，需要在乘以原来的高宽，拿到正常的坐标
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:]})

        return results

