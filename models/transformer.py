# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from models.gate_att import GateAttention


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 ):
        super().__init__()

        # 初始化一层encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.conv_pre = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.mlp_pre = MLP(d_model, d_model, 1, 3)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers

    def _reset_parameters(self):  # self.parameters()获取网络参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 参数Xavier服从正态分布初始化

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)  # 生成decoder的初始query(全为0,shape与位置编码相同),而其位置编码由nn.embedding生成
        interactiveness = torch.zeros((query_embed.shape[0], query_embed.shape[1], 1), device=query_embed.device)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        memory_2d = memory.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        heatmap = self.conv_pre(memory_2d)
        heatmap = heatmap.flatten(2).permute(2, 0, 1)
        heatmap = self.mlp_pre(heatmap)

        hs, interactiveness = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed,
                                           query_pos=query_embed,
                                           interactiveness=interactiveness, img_interactiveness=heatmap)[:]

        hs = hs.transpose(1, 2)
        interactiveness = interactiveness.transpose(1, 2)
        heatmap = heatmap.permute(1, 2, 0).contiguous().view(bs, h, w)
        # 最终输出hs.transpose()交换第1维和第2维，原decoder输出hs为(中间层数,num_queries,bs,hidden_dim) -> (中间层数,bs,num_queries,hidden_dim)
        # 另一个输出memory.permute(1, 2, 0).view(bs, c, h, w),是把encoder的输出转化成原始特征图的shape  permute()调换维度，view()展开成目标维度
        return hs, interactiveness, heatmap, memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)  # _get_clones()函数是结构的深度复制，即参数是不同的。
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ):
        output = src

        for layer in self.layers:  # 遍历所有encoder层得到最终输出
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos,
                           )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                interactiveness: Optional[Tensor] = None,
                img_interactiveness: Optional[Tensor] = None,
                ):
        output = tgt
        intermediate = []  # 存储decoder中间层输出
        intermediate_interactiveness = []

        for i, layer in enumerate(self.layers):
            output, interactiveness = layer(output, memory, tgt_mask=tgt_mask,
                                            memory_mask=memory_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                            memory_key_padding_mask=memory_key_padding_mask,
                                            pos=pos, query_pos=query_pos,
                                            interactiveness=interactiveness,
                                            img_interactiveness=img_interactiveness,
                                            )
            # topk_mask[torch.arange(atte_matrix.shape[0]).reshape(-1, 1, 1), torch.arange(atte_matrix.shape[1]).reshape(1, -1, 1), topk_indices] = 0

            if self.norm is not None:
                intermediate.append(self.norm(output))
            else:
                intermediate.append(output)
            intermediate_interactiveness.append(interactiveness)

        return torch.stack(intermediate), torch.stack(
            intermediate_interactiveness)  # 以栈的方式，返回一个沿新维度把输入张量序列连接起来(可以保留 序列 和 张量矩阵 信息)，最后一个变成第一个
        # return output   # 不要求返回中间层时，就输出最终结果


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, ):
        super().__init__()
        # 一层encoder结构，基本是固定的(多头自注意力,add&norm,FFN,add&norm)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # update interactiveness
        self.update1 = MLP(d_model, d_model, 1, 2)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 给输入的embedding加上位置编码，对q,k作用
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     ):
        # 自注意
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                # transformer的forward计算中只传入该参数src_key_padding_mask:考虑所有level，每个位置是否mask的标志。
                pos: Optional[Tensor] = None,
                ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # 一层decoder结构，基本是固定的(包括多头自注意力、多头注意力、Add&norm、FFN)
        self.self_attn = GateAttention(d_model, nhead, dropout=dropout)
        self.multi_qconf_proj1 = MLP(1, d_model, nhead, 2)
        self.multi_qpconf_proj1 = MLP(d_model, d_model, 1, 2)
        self.multi_kconf_proj1 = MLP(1, d_model, nhead, 2)
        self.multi_kpconf_proj1 = MLP(d_model, d_model, 1, 2)
        self.multi_vconf_proj1 = MLP(1, d_model, nhead, 2)
        self.multihead_attn = GateAttention(d_model, nhead, dropout=dropout)
        self.multi_qconf_proj2 = MLP(1, d_model, nhead, 2)
        self.multi_qpconf_proj2 = MLP(d_model, d_model, 1, 2)
        self.multi_kconf_proj2 = MLP(1, d_model, nhead, 2)
        self.multi_kpconf_proj2 = MLP(d_model, d_model, 1, 2)
        self.multi_vconf_proj2 = MLP(1, d_model, nhead, 2)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # update interactiveness
        self.update0 = MLP(nhead, d_model, 1, 2)
        self.update1 = MLP(nhead, d_model, 1, 2)
        self.update2 = MLP(d_model, d_model, 1, 2)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     interactiveness: Optional[Tensor] = None,
                     img_interactiveness: Optional[Tensor] = None,
                     ):
        # 自注意力
        q = k = self.with_pos_embed(tgt, query_pos)  # 自注意力得到的初始q,k，可看成由nn.embedding生成(因为初始query全为0)
        tgt2, new_interactiveness, _ = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                                      key_padding_mask=tgt_key_padding_mask,
                                                      query_conf=self.multi_qconf_proj1(
                                                          self.with_pos_embed(interactiveness,
                                                                              self.multi_qpconf_proj1(query_pos))),
                                                      key_conf=self.multi_kconf_proj1(
                                                          self.with_pos_embed(interactiveness,
                                                                              self.multi_kpconf_proj1(query_pos))),
                                                      value_conf=self.multi_vconf_proj1(interactiveness),
                                                      )[:]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # update interactiveness
        new_interactiveness = self.update0(new_interactiveness)
        interactiveness = new_interactiveness + interactiveness

        # 交叉注意力
        tgt2, new_interactiveness, _ = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                                           key=self.with_pos_embed(memory, pos),
                                                           value=memory, attn_mask=memory_mask,
                                                           key_padding_mask=memory_key_padding_mask,
                                                           query_conf=self.multi_qconf_proj2(
                                                               self.with_pos_embed(interactiveness,
                                                                                   self.multi_qpconf_proj2(query_pos))),
                                                           key_conf=self.multi_kconf_proj2(
                                                               self.with_pos_embed(img_interactiveness,
                                                                                   self.multi_kpconf_proj2(pos))),
                                                           value_conf=self.multi_vconf_proj2(img_interactiveness),
                                                           )[:]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # update interactiveness
        new_interactiveness = self.update1(new_interactiveness)
        interactiveness = new_interactiveness + interactiveness

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # update interactiveness
        new_interactiveness = self.update2(tgt)
        interactiveness = new_interactiveness + interactiveness

        return tgt, interactiveness

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    interactiveness: Optional[Tensor] = None,
                    img_interactiveness: Optional[Tensor] = None,
                    ):
        # 自注意力
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # 交叉注意力
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   )[0]
        tgt = tgt + self.dropout2(tgt2)

        # FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, interactiveness

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                # transformer的forward计算中只传入该参数memory_key_padding_mask:考虑encoder输出的所有level，每个位置是否mask的标志。
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                interactiveness: Optional[Tensor] = None,
                img_interactiveness: Optional[Tensor] = None,
                ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                    interactiveness, img_interactiveness)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos,
                                 interactiveness, img_interactiveness)


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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,  # 布尔值，为true时在encoder的forward前进行norm操作;为False时，则放到最后进行add&norm

    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

