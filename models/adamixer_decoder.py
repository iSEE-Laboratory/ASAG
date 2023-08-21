
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import box_ops
from .adamixer_decoder_utils import (sampling_3d, AdaptiveMixing, bbox_overlaps, position_embedding, decode_box, make_sample_points, build_activation, MLP)

class AdaptiveSamplingMixing(nn.Module):

    def __init__(self,
                 in_points=32,
                 out_points=128,
                 n_groups=4,
                 content_dim=256,
                 feat_channels=None,
                 activation='relu',
                 featmap_strides=[4,8,16,32]
                 ):
        super().__init__()
        self.in_points = in_points
        self.out_points = out_points
        self.n_groups = n_groups
        self.content_dim = content_dim
        self.feat_channels = feat_channels if feat_channels is not None else self.content_dim
        self.featmap_strides = featmap_strides

        self.sampling_offset_generator = nn.Sequential(
            nn.Linear(content_dim, in_points * n_groups * 3)
        )

        self.norm = nn.LayerNorm(content_dim)

        self.adaptive_mixing = AdaptiveMixing(
            self.feat_channels,
            query_dim=self.content_dim,
            in_points=self.in_points,
            out_points=self.out_points,
            n_groups=self.n_groups,
            activation=activation,
        )

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.sampling_offset_generator[-1].weight)
        nn.init.zeros_(self.sampling_offset_generator[-1].bias)

        bias = self.sampling_offset_generator[-1].bias.data.view(
            self.n_groups, self.in_points, 3)
        # initialize sampling delta x, delta y
        bandwidth = 0.5 * 1.0
        nn.init.uniform_(bias, -bandwidth, bandwidth)

        # initialize sampling delta z
        nn.init.constant_(bias[:, :, 2:3], -1.0)

        self.adaptive_mixing.init_weights()

    def forward(self, x, query_feat, query_roi):
        offset = self.sampling_offset_generator(query_feat)

        sample_points_xyz = make_sample_points(
            offset, self.n_groups * self.in_points, query_roi,
        )

        sampled_feature, _ = sampling_3d(sample_points_xyz, x,
                                         featmap_strides=self.featmap_strides,
                                         n_points=self.in_points,
                                         )


        query_feat = self.adaptive_mixing(sampled_feature, query_feat)
        query_feat = self.norm(query_feat)
        return query_feat



class AdaMixerDecoderStage(nn.Module):

    def __init__(self,
                 num_classes,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 content_dim=256,
                 feat_channels=256,
                 dropout=0.0,
                 in_points=32,
                 out_points=128,
                 n_groups=4,
                 activation='relu',
                 featmap_strides=[4,8,16,32]):

        super().__init__()
        self.num_classes = num_classes
        self.content_dim = content_dim
        self.featmap_strides = featmap_strides
        self.attention = nn.MultiheadAttention(content_dim, num_heads, dropout=dropout)
        self.attention_norm = nn.LayerNorm(content_dim)

        self.ffn = MLP(content_dim, feedforward_channels, content_dim, 2, activation)
        self.ffn_norm = nn.LayerNorm(content_dim)

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(nn.Linear(content_dim, content_dim, False))
            self.cls_fcs.append(nn.LayerNorm(content_dim))
            self.cls_fcs.append(build_activation(activation))
        self.fc_cls = nn.Linear(content_dim, self.num_classes)

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(nn.Linear(content_dim, content_dim, False))
            self.reg_fcs.append(nn.LayerNorm(content_dim))
            self.reg_fcs.append(build_activation(activation))
        # over load the self.fc_cls in BBoxHead
        self.fc_reg = nn.Linear(content_dim, 4)

        self.in_points = in_points
        self.n_heads = n_groups
        self.out_points = out_points

        self.sampling_n_mixing = AdaptiveSamplingMixing(
            content_dim=content_dim,  # query dim
            feat_channels=feat_channels,
            in_points=self.in_points,
            out_points=self.out_points,
            n_groups=self.n_heads,
            activation=activation,
            featmap_strides=self.featmap_strides,
        )

        self.iof_tau = nn.Parameter(torch.ones(num_heads,))
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
        self.sampling_n_mixing.init_weights()

    def forward(self,
                x,
                query_xyzr,
                query_content, dn_masks=None):

        with torch.no_grad():
            rois = decode_box(query_xyzr)
            roi_box_batched = rois
            iof = bbox_overlaps(roi_box_batched, roi_box_batched, mode='iof')[:, None, :, :]
            iof = (iof + 1e-7).log()
            pe = position_embedding(query_xyzr, query_content.size(-1) // 4)

        '''IoF'''
        attn_bias = (iof * self.iof_tau.view(1, -1, 1, 1)).flatten(0, 1)
        if dn_masks is not None:
            attn_bias.masked_fill_(dn_masks, float('-inf'))

        query_content = query_content.permute(1, 0, 2)
        pe = pe.permute(1, 0, 2)
        '''sinusoidal positional embedding'''
        query_content_attn = query_content + pe
        query_content = self.attention(
            query_content_attn, query_content_attn, query_content_attn,
            attn_mask=attn_bias,
        )[0]
        query_content += query_content_attn
        query_content = self.attention_norm(query_content)
        query_content = query_content.permute(1, 0, 2)

        ''' adaptive 3D sampling and mixing '''
        query_content = self.sampling_n_mixing(
            x, query_content, query_xyzr)

        # FFN
        query_content = self.ffn_norm(self.ffn(query_content) + query_content)

        cls_feat = query_content
        reg_feat = query_content

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat.clone())
        for reg_layer in self.reg_fcs:
            reg_feat = reg_layer(reg_feat.clone())

        cls_score = self.fc_cls(cls_feat)
        xyzr_delta = self.fc_reg(reg_feat)

        return cls_score, xyzr_delta, query_content

    def refine_xyzr(self, xyzr, xyzr_delta, return_bbox=True):
        z = xyzr[..., 2:3]
        new_xy = xyzr[..., 0:2] + xyzr_delta[..., 0:2] * (2 ** z)
        new_zr = xyzr[..., 2:4] + xyzr_delta[..., 2:4]
        xyzr = torch.cat([new_xy, new_zr], dim=-1)
        if return_bbox:
            return xyzr, decode_box(xyzr) * self.featmap_strides[0]
        else:
            return xyzr



class AdaMixerDecoder(nn.Module):
    def __init__(self, num_classes,
                 num_layers=6,
                 content_dim=256,
                 featmap_strides=[4, 8, 16, 32],
                 num_heads=8,
                 num_cls_fcs=1,
                 num_reg_fcs=3,
                 feedforward_channels=2048,
                 dropout=0.0,
                 in_points=32,
                 out_points=128,
                 n_groups=4,
                 activation='relu'):
        super().__init__()
        self.featmap_strides = featmap_strides
        self.num_layers = num_layers
        self.content_dim = content_dim
        self.decoder_embedding = nn.Embedding(num_layers, content_dim)
        self.bbox_head = nn.ModuleList()
        for i in range(num_layers):
            self.bbox_head.append(AdaMixerDecoderStage(num_classes, num_heads, num_cls_fcs, num_reg_fcs,
                 feedforward_channels, content_dim, content_dim, dropout, in_points, out_points, n_groups, activation, featmap_strides=self.featmap_strides))

    def _bbox_forward(self, stage, img_feat, query_xyzr, query_content, dn_masks=None):
        bbox_head = self.bbox_head[stage]
        cls_score, delta_xyzr, query_content = bbox_head(img_feat, query_xyzr, query_content, dn_masks=dn_masks)

        query_xyzr, decoded_bboxes = self.bbox_head[stage].refine_xyzr(query_xyzr, delta_xyzr)

        bbox_results = dict(
            cls_score=cls_score,
            query_xyzr=query_xyzr,
            decode_bbox_pred=decoded_bboxes,
            query_content=query_content,
        )
        return bbox_results

    def forward(self,
                x,
                query_xyzr,
                query_content,
                imgs_whwh, dn_masks=None):

        all_stage_bbox_results = []
        for stage in range(self.num_layers):
            bbox_results = self._bbox_forward(stage, x, query_xyzr, query_content + self.decoder_embedding.weight[stage].unsqueeze(0).unsqueeze(0), dn_masks)
            all_stage_bbox_results.append(bbox_results)

            # query_xyzr = bbox_results['query_xyzr'].detach()
            # query_content = bbox_results['query_content']
            bbox_results['decode_bbox_pred'] = box_ops.box_xyxy_to_cxcywh(bbox_results['decode_bbox_pred'] / imgs_whwh)

        return all_stage_bbox_results


def build_adamixer_decoder(args, num_classes, featmap_strides):
    return AdaMixerDecoder(
        num_classes=num_classes,
        num_layers=args.num_decoder_layers,
        content_dim=args.hidden_dim,
        featmap_strides=featmap_strides,
        num_heads=args.nheads,
        num_cls_fcs=1,
        num_reg_fcs=3,
        feedforward_channels=2048,
        dropout=args.dropout,
        in_points=args.in_points,
        out_points=args.out_points,
        n_groups=args.n_groups,
        activation=args.activation,
    )

