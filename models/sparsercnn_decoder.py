
"""
SparseRCNN Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import util.box_ops as box_ops
from .adamixer_decoder_utils import build_activation

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHead(nn.Module):

    def __init__(self, num_classes, num_layers=6, content_dim=256, feedforward_channels=2048, num_heads=8, dropout=0.0, activation='relu'):
        super().__init__()
        
        # Build heads.
        self.d_model = content_dim
        self.head_series = nn.ModuleList()
        for i in range(num_layers):
            self.head_series.append(RCNNHead(self.d_model, num_classes, feedforward_channels, num_heads, dropout, activation))
        self.return_intermediate = True
        self.decoder_embedding = nn.Embedding(num_layers, content_dim)
        self.num_layers = num_layers
        
        # Init parameters.
        self.use_focal = True
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = 0.01
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)


    def forward(self, features, init_bboxes, init_features, imgs_whwh, dn_masks):

        all_stage_results = []

        bs = len(features[0])
        bboxes = init_bboxes
        
        # init_features = init_features[None].repeat(1, bs, 1)
        proposal_features = init_features.clone()
        
        for i, rcnn_head in enumerate(self.head_series):
            class_logits, pred_bboxes, _ = rcnn_head(features, bboxes, proposal_features + self.decoder_embedding.weight[i].unsqueeze(0).unsqueeze(0), imgs_whwh, dn_masks)
            all_stage_results.append({'cls_score': class_logits, 'decode_bbox_pred': pred_bboxes})
            # bboxes = pred_bboxes.detach()

        return all_stage_results


class RCNNHead(nn.Module):

    def __init__(self, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # Build RoI.
        self.featmap_name = ['C2', 'C3', 'C4', 'C5']
        self.box_pooler = torchvision.ops.MultiScaleRoIAlign(self.featmap_name, 7, 2)

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(self.d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = build_activation(activation)

        # cls.
        num_cls = 1
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(build_activation(activation))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = 3
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(build_activation(activation))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = True
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights
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
        nn.init.constant_(self.class_logits.bias, bias_init)

        nn.init.zeros_(self.bboxes_delta.weight)
        nn.init.zeros_(self.bboxes_delta.bias)


    def forward(self, features, bboxes, pro_features, imgs_whwh, dn_masks=None):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]
        
        # roi_feature.
        # sample tokens adaptively
        proposals_detach = box_ops.box_cxcywh_to_xyxy(bboxes)
        proposals_detach = proposals_detach * imgs_whwh
        image_size = list(torch.stack([imgs_whwh[:, 0, 1], imgs_whwh[:, 0, 0]], dim=-1).cpu().numpy())
        image_size = [tuple(size) for size in image_size]
        x = {k:v for k,v in zip(self.featmap_name, features)}
        proposals_detach_list = [*torch.unbind(proposals_detach)]

        roi_features = self.box_pooler(
            x, proposals_detach_list, image_size
        )  # (bs * num_queries, c, 7, 7)         
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)        

        # self_att.
        attn_bias = None
        if dn_masks is not None:
            attn_bias = torch.zeros_like(dn_masks, dtype=torch.float)
            attn_bias.masked_fill_(dn_masks, float('-inf'))
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features, attn_mask=attn_bias,)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features
    

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2]
        heights = boxes[:, 3]
        ctr_x = boxes[:, 0]
        ctr_y = boxes[:, 1]

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0] / wx
        dy = deltas[:, 1] / wy
        dw = deltas[:, 2] / ww
        dh = deltas[:, 3] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h], dim=-1)
        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, hidden_dim=256, dim_dynamic=64, num_dynamic=2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dim_dynamic = dim_dynamic
        self.num_dynamic = num_dynamic
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = 7
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic) # (N * nr_boxes, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_sparsercnn_decoder(args, num_classes):
    return DynamicHead(
        num_classes=num_classes,
        num_layers=args.num_decoder_layers,
        content_dim=args.hidden_dim,
        num_heads=args.nheads,
        feedforward_channels=2048,
        dropout=args.dropout,
        activation=args.activation,
    )


