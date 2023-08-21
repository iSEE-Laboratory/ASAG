import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision
import numpy as np
from .adamixer_decoder_utils import decode_box, bbox_overlaps, position_embedding
from typing import Dict, List
import util.box_ops as box_ops
from .adamixer_decoder_utils import build_activation, MLP


class RelationBlock(nn.Module):

    def __init__(self, num_classes: int, featmap_strides: List, d_model: int=256, activation='relu', decoder_type='AdaMixer', num_query_pattern=1):
        super().__init__()
        self.decoder_type = decoder_type
        self.d_model = d_model
        self.featmap_strides = featmap_strides
        self.num_query_pattern = num_query_pattern
        if self.num_query_pattern > 1:
            self.pattern = nn.Embedding(self.num_query_pattern, d_model)
        self.roi = nn.Linear(self.d_model * 7 * 7, self.d_model)

        self.featmap_name = ['C2', 'C3', 'C4', 'C5']
        self.roialign = torchvision.ops.MultiScaleRoIAlign(self.featmap_name, 7, 2)

        # Self-Attention
        self.tokens_norm1 = nn.LayerNorm(d_model)
        self.tokens_proj = MLP(self.d_model, 2048, self.d_model, 2, activation)
        self.tokens_norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(self.d_model, 8)
        self.iof_tau = nn.Parameter(torch.ones(8,))
        self.tokens_norm3 = nn.LayerNorm(d_model)
        # cls.
        num_cls = 1
        cls_fcs = list()
        for _ in range(num_cls):
            cls_fcs.append(nn.Linear(d_model, d_model, False))
            cls_fcs.append(nn.LayerNorm(d_model))
            cls_fcs.append(build_activation(activation))
        self.cls_fcs = nn.ModuleList(cls_fcs)
        self.fc_cls = nn.Linear(d_model, num_classes)

        # reg.
        num_reg = 3
        reg_fcs = list()
        for _ in range(num_reg):
            reg_fcs.append(nn.Linear(d_model, d_model, False))
            reg_fcs.append(nn.LayerNorm(d_model))
            reg_fcs.append(build_activation(activation))
        self.reg_fcs = nn.ModuleList(reg_fcs)
        self.fc_reg = nn.Linear(d_model, 4)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass

        bias_init = float(-np.log((1 - 0.01) / 0.01))
        nn.init.constant_(self.fc_cls.bias, bias_init)

        nn.init.zeros_(self.fc_reg.weight)
        nn.init.zeros_(self.fc_reg.bias)
        nn.init.uniform_(self.iof_tau, 0.0, 4.0)


    def forward(self, x: List[Tensor], anchors_detach, imgs_whwh: Tensor, hw, dn_masks: Tensor=None, valid_mask: Tensor=None):
        '''
        x: (B, d_model, H, W)
        anchors_detach: (B, num_queries, 4) image_size cxcywh
        '''
        bs = x[0].shape[0]
        num_queries = anchors_detach.shape[1]

        # sample tokens adaptively
        proposals_detach = box_ops.box_cxcywh_to_xyxy(anchors_detach)
        # proposals_detach = proposals_detach * imgs_whwh
        # image_size = list(torch.stack([imgs_whwh[:, 0, 1], imgs_whwh[:, 0, 0]], dim=-1).cpu().numpy())
        # image_size = [tuple(size) for size in image_size]
        x = {k:v for k,v in zip(self.featmap_name, x)}
        proposals_detach_list = [*torch.unbind(proposals_detach)]

        roi_content = self.roialign(
            x, proposals_detach_list, hw
        )  # (bs * num_queries, c, 7, 7)

        tokens = roi_content.reshape(bs, num_queries, -1)
        tokens = self.roi(tokens) # (bs, num_queries, d_model)
        query_content = self.tokens_norm1(tokens)

        if self.num_query_pattern > 1:
            patterns = self.pattern.weight.reshape(1, self.num_query_pattern, 1, self.d_model).repeat(bs, 1, num_queries, 1).reshape(
                        bs, self.num_query_pattern * num_queries, self.d_model)
            query_content = query_content.unsqueeze(1).repeat(1, self.num_query_pattern, 1, 1).reshape(bs, self.num_query_pattern * num_queries, self.d_model)
            query_content += patterns
            proposals_detach = proposals_detach.unsqueeze(1).repeat(1, self.num_query_pattern, 1, 1).reshape(bs, self.num_query_pattern * num_queries, 4)

            valid_mask = valid_mask.unsqueeze(1).repeat(1, self.num_query_pattern, 1).reshape(bs, self.num_query_pattern * num_queries)
            num_dn = dn_masks.shape[1] - num_queries
            mask_size = num_dn + num_queries * self.num_query_pattern
            new_dn_masks = torch.ones((dn_masks.shape[0], mask_size, mask_size)).to(dn_masks.device) < 0 # False
            new_dn_masks[:, :num_dn, :num_dn] = dn_masks[:, :num_dn, :num_dn]
            new_dn_masks[:, :num_dn, num_dn:] = dn_masks[:, :num_dn, num_dn:].repeat(1, 1, self.num_query_pattern)
            new_dn_masks[:, num_dn:, :] = True
            new_dn_masks[:, num_dn:, num_dn:] = dn_masks[:, num_dn:, num_dn:].repeat(1, self.num_query_pattern, self.num_query_pattern)
            dn_masks = new_dn_masks

        
        # translate to xyzr
        proposals_detach = proposals_detach / self.featmap_strides[0]
        xy = 0.5 * (proposals_detach[..., 0:2] + proposals_detach[..., 2:4])
        wh = proposals_detach[..., 2:4] - proposals_detach[..., 0:2]
        z = ((wh).prod(-1, keepdim=True) + 1e-7).log2() / 2
        r = ((wh[..., 1:2]/(wh[..., 0:1] + 1e-7)) + 1e-7).log2() / 2
        xyzr = torch.cat([xy, z, r], dim=-1)

        # self-attention
        with torch.no_grad():
            rois = decode_box(xyzr)
            roi_box_batched = rois
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[:, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(xyzr, query_content.size(-1) // 4)

        '''IoF'''
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)).flatten(0, 1)
        if dn_masks is not None:
            attn_bias.masked_fill_(dn_masks, float('-inf'))

        query_content = query_content.permute(1, 0, 2)
        pe = pe.permute(1, 0, 2)
        '''sinusoidal positional embedding'''
        query_content_attn = query_content + pe
        query_content = self.attn(
            query_content_attn, query_content_attn, query_content_attn,
            attn_mask=attn_bias,
        )[0]
        query_content += query_content_attn
        query_content = self.tokens_norm3(query_content)
        query_content = query_content.permute(1, 0, 2) # (B, num_proposal, d_model)

        res = self.tokens_proj(query_content)
        query_content = query_content + res
        query_content = self.tokens_norm2(query_content)

        # predict class and bbox for surpervision
        cls_feat = query_content.clone()
        reg_feat = query_content.clone()

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.fc_cls(cls_feat) # (B, num_proposal, num_classes)
        xyzr_delta = self.fc_reg(reg_feat) # (B, num_proposal, 4)

        # refine xyzr
        z = xyzr[..., 2:3]
        new_xy = xyzr[..., 0:2] + xyzr_delta[..., 0:2] * (2 ** z)
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_xy, new_zr], dim=-1)
        proposal = box_ops.box_xyxy_to_cxcywh(decode_box(xyzr) * self.featmap_strides[0] / imgs_whwh)
        if self.decoder_type == 'AdaMixer':
            return proposal, cls_score, query_content, xyzr.clone().detach(), dn_masks, valid_mask
        else:
            return proposal, cls_score, query_content, proposal.clone().detach(), dn_masks, valid_mask



def build_rpn(args, num_classes, featmap_strides):
    return RelationBlock(
        num_classes=num_classes,
        featmap_strides=featmap_strides,
        d_model=args.hidden_dim,
        activation=args.activation,
        decoder_type=args.decoder_type,
        num_query_pattern=args.num_query_pattern,
    )


