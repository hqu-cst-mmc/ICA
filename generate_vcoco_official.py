# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import argparse
from pathlib import Path
import numpy as np
import copy
import pickle
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.vcoco import build as build_dataset
from models.backbone import build_backbone
from models.transformer import build_transformer
import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from args import build_args


def _get_clones(module, num_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


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
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes+1)
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
        self.transformer.decoder.interactiveness_pre = self.verb_class_embed

    def forward(self, samples: NestedTensor):
        # 判断输入进来的batch图像是否为NestedTensor这一类型的
        # 如果为False,就对samples进行统一尺寸大小操作
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        # transformer最终输出五个：detector and classifier各层输出[0], 每层的先验人的的框和物体框 和 与输入特征图shape一样的memory[1]
        hs, _ = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[:]
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
            verb_class = self.verb_class_embed[lvl](hs[lvl])
            outputs_verb_class.append(verb_class[..., :-1])
            outputs_interactiveness.append(verb_class[..., -1, None])

        outputs_interactiveness = torch.stack(outputs_interactiveness)
        outputs_sub_coords = torch.stack(outputs_sub_coords)
        outputs_obj_coords = torch.stack(outputs_obj_coords)
        outputs_obj_class = torch.stack(outputs_obj_class)
        outputs_verb_class = torch.stack(outputs_verb_class)

        # 取decoder最后一层输出
        out = {'pred_interactiveness': outputs_interactiveness[-1],
               'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coords[-1], 'pred_obj_boxes': outputs_obj_coords[-1]}
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PostProcessHOI(nn.Module):

    def __init__(self, num_queries, subject_category_id, correct_mat):
        super().__init__()
        self.max_hois = 100

        self.num_queries = num_queries
        self.subject_category_id = subject_category_id

        correct_mat = np.concatenate((correct_mat, np.ones((correct_mat.shape[0], 1))), axis=1)
        self.register_buffer('correct_mat', torch.from_numpy(correct_mat))

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
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()
        verb_scores = verb_scores * out_interactiveness.sigmoid()

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(b.to('cpu').numpy(), l.to('cpu').numpy())]

            hoi_scores = vs * os.unsqueeze(1)

            verb_labels = torch.arange(hoi_scores.shape[1], device=self.correct_mat.device).view(1, -1).expand(
                hoi_scores.shape[0], -1)
            object_labels = ol.view(-1, 1).expand(-1, hoi_scores.shape[1])
            masks = self.correct_mat[verb_labels.reshape(-1), object_labels.reshape(-1)].view(hoi_scores.shape)
            hoi_scores *= masks

            ids = torch.arange(b.shape[0])

            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                    subject_id, object_id, category_id, score in zip(ids[:ids.shape[0] // 2].to('cpu').numpy(),
                                                                     ids[ids.shape[0] // 2:].to('cpu').numpy(),
                                                                     verb_labels.to('cpu').numpy(), hoi_scores.to('cpu').numpy())]

            results.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })
            results[-1] = self.triplet_nms_filter(results[-1])

        return results

    def triplet_nms_filter(self, preds):
        pred_bboxes = preds['predictions']
        pred_hois = preds['hoi_prediction']
        all_triplets = {}
        for index, pred_hoi in enumerate(pred_hois):
            triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                      str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

            if triplet not in all_triplets:
                all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
            all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
            all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
            all_triplets[triplet]['scores'].append(pred_hoi['score'])
            all_triplets[triplet]['indexes'].append(index)

        all_keep_inds = []
        for triplet, values in all_triplets.items():
            subs, objs, scores = values['subs'], values['objs'], values['scores']
            keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

            keep_inds = list(np.array(values['indexes'])[keep_inds])
            all_keep_inds.extend(keep_inds)

        preds_filtered = {
            'predictions': pred_bboxes,
            'hoi_prediction': list(np.array(preds['hoi_prediction'])[all_keep_inds])
            }

        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        max_scores = np.max(scores, axis=1)
        order = max_scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter/sub_union, 1.0) * np.power(obj_inter / obj_union, 0.5)
            inds = np.where(ovr <= 0.7)[0]

            order = order[inds + 1]
        return keep_inds


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                     24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                     37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                     48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                     58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                     72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                     82, 84, 85, 86, 87, 88, 89, 90)

    verb_classes = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj',
                    'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                    'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
                    'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr', 'kick_obj',
                    'point_instr', 'read_obj', 'snowboard_instr']

    device = torch.device(args.device)

    dataset_val = build_dataset(image_set='val', args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    args.lr_backbone = 0
    args.masks = False
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DETRHOI(backbone, transformer, len(valid_obj_ids) + 1, len(verb_classes), args.num_queries, args.dec_layers, 4, args.aux_loss)
    # model = DETRHOI(backbone, transformer, len(valid_obj_ids) + 1, len(verb_classes),
    #                 args.num_queries)
    post_processor = PostProcessHOI(args.num_queries, args.subject_category_id, dataset_val.correct_mat)
    model.to(device)
    post_processor.to(device)

    checkpoint = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    missing_category_id = 80
    detections = generate(model, post_processor, data_loader_val, device, verb_classes, missing_category_id)

    save_path = 'logs/log-vcoco.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump(detections, f, protocol=2)


@torch.no_grad()
def generate(model, post_processor, data_loader, device, verb_classes, missing_category_id):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate:'

    detections = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = post_processor(outputs, orig_target_sizes)

        for img_results, img_targets in zip(results, targets):
            for hoi in img_results['hoi_prediction']:
                detection = {
                    'image_id': img_targets['img_id'],
                    'person_box': img_results['predictions'][hoi['subject_id']]['bbox'].tolist()
                }
                if img_results['predictions'][hoi['object_id']]['category_id'] == missing_category_id:
                    object_box = [np.nan, np.nan, np.nan, np.nan]
                else:
                    object_box = img_results['predictions'][hoi['object_id']]['bbox'].tolist()
                cut_agent = 0
                hit_agent = 0
                eat_agent = 0
                for idx, score in zip(hoi['category_id'], hoi['score']):
                    verb_class = verb_classes[idx]
                    score = score.item()
                    if len(verb_class.split('_')) == 1:
                        detection['{}_agent'.format(verb_class)] = score
                    elif 'cut_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        cut_agent = score if score > cut_agent else cut_agent
                    elif 'hit_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        hit_agent = score if score > hit_agent else hit_agent
                    elif 'eat_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        eat_agent = score if score > eat_agent else eat_agent
                    else:
                        detection[verb_class] = object_box + [score]
                        detection['{}_agent'.format(
                            verb_class.replace('_obj', '').replace('_instr', ''))] = score
                detection['cut_agent'] = cut_agent
                detection['hit_agent'] = hit_agent
                detection['eat_agent'] = eat_agent
                detections.append(detection)

    return detections


if __name__ == '__main__':
    args = build_args()
    main(args)
