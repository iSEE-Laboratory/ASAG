
import torch
import torch.nn as nn
import torchvision
from torch import Tensor
import torch.nn.functional as F
from typing import List
from .adamixer_decoder_utils import MLP, build_activation
import util.box_ops as box_ops



class AnchorGenerator(nn.Module):

    def __init__(self, d_model: int=256, num_pred_anchor1: int=50, num_pred_anchor2: int=20, patch_size_interpolate: int=15, patch_size: int=15, activation='relu', compress_dim: int=8, 
                stride: List[int]=[4, 8, 16, 32], threshold1: float=0.7, threshold2: float=0.1, nms_threshold: float=0.25, use_target_epoch: int=1, use_anchor_threshold: float=0.1, used_inference_level: float='P3P6', dataset='coco'):
        super().__init__()
        self.interpolate_size = 30
        self.hidden_dim = 125 * compress_dim
        self.compress_dim = compress_dim
        self.patch_size_interpolate = patch_size_interpolate
        self.patch_size = patch_size
        self.num_pred_anchor1 = num_pred_anchor1
        self.num_pred_anchor2 = num_pred_anchor2
        self.d_model = d_model
        self.stride = stride
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.use_anchor_threshold = use_anchor_threshold
        self.base_crop_pos = [[0, 0], [0, self.interpolate_size-self.patch_size_interpolate], [self.interpolate_size-self.patch_size_interpolate, 0], [self.interpolate_size-self.patch_size_interpolate, self.interpolate_size-self.patch_size_interpolate]]
        self.base_crop_pos = torch.tensor(self.base_crop_pos, dtype=torch.int64)
        self.targets_offset = 100
        self.minial_training_patch = 4
        self.use_target_epoch = use_target_epoch
        self.nms_threshold = nms_threshold
        self.max_anchor = 200 if num_pred_anchor1 == 50 else 500
        self.min_anchor = 5 if num_pred_anchor1 == 50 else 50
        self.dataset = dataset
        if self.dataset == 'crowdhuman':
            self.max_anchor = 600
            self.min_anchor = 50
        # 'P3P6', 'P4P6', 'P5P6'
        if used_inference_level == 'P3P6':
            self.used_inference_level = 3
        elif used_inference_level == 'P4P6':
            self.used_inference_level = 2
        else:
            self.used_inference_level = 1
        
        self.compress = nn.ModuleList()
        for i in range(len(stride)):
            self.compress.append(
                nn.Sequential(
                    nn.Conv2d(d_model, d_model // 2, 1, 1, 0),
                    build_activation(activation),
                    nn.Conv2d(d_model // 2, self.compress_dim, 3, 1, 1),
                )
            )
        self.P6 = nn.Conv2d(self.compress_dim, self.compress_dim, 3, 2, 2, 2)

        self.layernorm = nn.ModuleList()
        self.pos_embedding = nn.ModuleList() 
        self.layernorm.append(nn.LayerNorm((self.interpolate_size*self.interpolate_size//4, self.compress_dim)))
        self.pos_embedding.append(nn.Embedding(self.interpolate_size*self.interpolate_size//4, self.compress_dim))
        for i in range(len(stride)-1):
            self.pos_embedding.append(nn.Embedding(self.patch_size*self.patch_size, self.compress_dim))
            self.layernorm.append(nn.LayerNorm((self.patch_size*self.patch_size, self.compress_dim)))

        # pred dynamic anchor
        self.proj_all = MLP(self.patch_size_interpolate*self.patch_size_interpolate*self.compress_dim, self.hidden_dim, 5*self.num_pred_anchor1, 3, activation)
        self.proj_patch = MLP(self.patch_size*self.patch_size*self.compress_dim, self.hidden_dim, 5*self.num_pred_anchor2, 3, activation)
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
        for m in self.pos_embedding:
            nn.init.zeros_(m.weight)


    @torch.no_grad()
    def generate_crop_pos(self, anchors, imgs_whwh, stride): # generate_crop_pos from box center
        if len(anchors) == 0:
            return anchors.new_zeros((0, 2), dtype=torch.int64)
        center = (anchors[:, :2] + anchors[:, 2:]) / 2 / stride
        crop_pos = torch.floor(center - self.patch_size / 2) # top_left, 'xy'
        offset = torch.ceil(imgs_whwh[:, 0, :2] / stride) - (crop_pos + self.patch_size) # move crop_pos if patch is out of boundaries
        offset[offset>0] = 0
        crop_pos += offset
        crop_pos[crop_pos < 0] = 0
        return crop_pos.long()

    
    @torch.no_grad()
    def generate_targets_crop_pos(self, original_crop_pos, original_batch_idx, targets, imgs_whwh, stride, epoch): # generate_crop_pos from box center with a little random shift
        crop_pos = []
        batch_idx = []
        for i, t in enumerate(targets):
            # generate targets patch
            boxes = t['boxes'] * imgs_whwh[i]
            mask = (boxes[:, 2] < stride * self.patch_size / 2) & (boxes[:, 3] < stride * self.patch_size / 2)
            boxes = boxes[mask]
            scores = - torch.sum(torch.sum(torch.square(boxes[:, :2].unsqueeze(1) - boxes[:, :2].unsqueeze(0)), dim=-1), dim=-1)
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            idx = torchvision.ops.nms(boxes, scores, self.nms_threshold)
            boxes = boxes[idx]
            boxes /= stride
            crop_pos_i = torch.floor(boxes[:, :2]) # top_left, 'xy'
            offset = torch.ceil(imgs_whwh[i:i+1, 0, :2] / stride) - (crop_pos_i + self.patch_size) # move crop_pos if patch is out of boundaries
            offset[offset>0] = 0
            crop_pos_i += offset
            offset = crop_pos_i + self.patch_size - boxes[:, 2:] # random move crop_pos for data augmentation
            offset[offset<0] = 0
            offset = torch.rand_like(offset) * offset
            crop_pos_i = torch.floor(crop_pos_i - offset).long()
            crop_pos_i[crop_pos_i < 0] = 0

            # cat with original patch
            if epoch < self.use_target_epoch:
                batch_idx_i = torch.ones(len(crop_pos_i), device=original_batch_idx.device) * i
            else:
                batch_idx_i = torch.ones(torch.sum(original_batch_idx==i), device=original_batch_idx.device) * i
                if len(crop_pos_i) > 0 and torch.sum(original_batch_idx==i) < self.minial_training_patch:
                    random_idx = torch.randperm(len(crop_pos_i))[:self.minial_training_patch-torch.sum(original_batch_idx==i)]
                    crop_pos_i = torch.cat([original_crop_pos[original_batch_idx==i], crop_pos_i[random_idx]])
                    batch_idx_i = torch.cat([batch_idx_i, torch.ones(len(random_idx), device=original_batch_idx.device) * i + self.targets_offset])
                else:
                    crop_pos_i = original_crop_pos[original_batch_idx==i]

            if len(crop_pos_i) < self.minial_training_patch:
                W_max = max(int(torch.ceil(imgs_whwh[i, 0, 0] / stride)) - self.patch_size, 1)
                H_max = max(int(torch.ceil(imgs_whwh[i, 0, 1] / stride)) - self.patch_size, 1)
                add_crop_pos = torch.stack([torch.randint(0, W_max, (self.minial_training_patch-len(crop_pos_i),)), torch.randint(0, H_max, (self.minial_training_patch-len(crop_pos_i),))], dim=1).to(boxes.device).long()
                crop_pos_i = torch.cat([crop_pos_i, add_crop_pos], dim=0)
                batch_idx_i = torch.cat([batch_idx_i, torch.ones(len(add_crop_pos), device=original_batch_idx.device) * i + self.targets_offset])

            crop_pos.append(crop_pos_i)
            batch_idx.append(batch_idx_i)

        crop_pos = torch.cat(crop_pos)
        batch_idx = torch.cat(batch_idx).long()
        return crop_pos, batch_idx

                
    def forward(self, x: List[Tensor], imgs_whwh: Tensor, targets=None, epoch=-1):
        '''
        feature: (B, d_model, H, W)
        mask: (B, H, W)
        '''
        if self.training:
            assert targets is not None
        B = x[0].shape[0]
        anchors_for_loss = []
        output_anchors = [[] for _ in range(B)]
        compressed_feature = [self.compress[i](x_i) for i, x_i in enumerate(x)]# (B, compress_dim, H, W)

        # interpolate
        interpolated_feature = []
        img_H = imgs_whwh[:, 0, 1] / self.stride[-1]
        img_W = imgs_whwh[:, 0, 0] / self.stride[-1]
        for i, (h, w) in enumerate(zip(img_H, img_W)):
            interpolated_feature.append(F.interpolate(compressed_feature[-1][i:i+1, :, :int(torch.round(h)), :int(torch.round(w))], size=(self.interpolate_size, self.interpolate_size), mode='bilinear')) # (1, compress_dim, interpolate_size, interpolate_size)
        interpolated_feature = torch.cat(interpolated_feature, dim=0) # (B, compress_dim, interpolate_size, interpolate_size)

        # get P6 patch
        P6_feature = self.P6(F.relu(interpolated_feature)).unsqueeze(1) # (B, 1, compress_dim, patch_size, patch_size)
        if self.training:
            P6_feature = torch.cat((P6_feature, torch.flip(P6_feature.clone(), dims=[4])), dim=1) # (B, 2, compress_dim, patch_size, patch_size)
            P6_feature = torch.cat((P6_feature, torch.flip(P6_feature.clone(), dims=[3])), dim=1) # (B, 4, compress_dim, patch_size, patch_size)

        # get P5 patch
        crop_pos = self.base_crop_pos.clone().to(P6_feature.device)
        image_token = torch.stack([interpolated_feature[:, :, pos[0]:pos[0]+self.patch_size_interpolate, pos[1]:pos[1]+self.patch_size_interpolate] for pos in crop_pos], dim=1) # (B, num_crops, compress_dim, patch_size, patch_size)
        image_token = torch.cat([image_token, P6_feature], dim=1) # (B, num_crops, compress_dim, patch_size, patch_size)
        image_token = image_token.flatten(3).permute(0, 1, 3, 2) # (B, num_crops, patch_size*patch_size, compress_dim)
        image_token = self.layernorm[0](image_token)
        image_token += self.pos_embedding[0].weight.clone().unsqueeze(0).unsqueeze(0)
        image_token = image_token.flatten(2) # (B, num_crops, patch_size*patch_size*compress_dim)
        # pred dynamic anchor
        total_anchors = self.proj_all(image_token)
        total_anchors = total_anchors.reshape(B, -1, self.num_pred_anchor1, 5) # 'cxcywh' conf
        total_anchors[..., :4] = torch.sigmoid(total_anchors[..., :4])
        batch_idx = torch.arange(0, B).to(P6_feature.device)
        anchors_for_loss.append((total_anchors[:, len(crop_pos):], None, batch_idx))
        anchors_for_loss.append((total_anchors[:, :len(crop_pos)], crop_pos, batch_idx))

        # change the coordinate to full image
        detached_anchors = total_anchors.clone().detach()[:, :len(crop_pos)+1] # (B, num_crops, self.num_proposals1, 5)
        detached_anchors[:, :4, :, :4] = detached_anchors[:, :4, :, :4] * (self.patch_size_interpolate / self.interpolate_size)
        detached_anchors[:, 1, :, 0] = (self.interpolate_size - self.patch_size_interpolate) / self.interpolate_size + detached_anchors[:, 1, :, 0] # top-right cx
        detached_anchors[:, 2, :, 1] = (self.interpolate_size - self.patch_size_interpolate) / self.interpolate_size + detached_anchors[:, 2, :, 1] # bottom-left cy
        detached_anchors[:, 3, :, :2] = (self.interpolate_size - self.patch_size_interpolate) / self.interpolate_size + detached_anchors[:, 3, :, :2] # bottom-right cxcy

        anchors = detached_anchors.flatten(1, 2)
        anchors[..., 4] = torch.sigmoid(anchors[..., 4]) # '0-1'
        anchors[..., :4] = anchors[..., :4] * imgs_whwh # change to image size
        too_small_mask = (anchors[..., 2] < 3) | (anchors[..., 3] < 3)
        too_small_anchors = anchors[too_small_mask]
        too_small_anchors[:, 4] = 0
        anchors[too_small_mask] = too_small_anchors
        
        # pred on other feature pyramid level
        for i in range(1, len(self.stride)):
            if self.used_inference_level == i:
                break # manually stop

            mask = (anchors[:, :, 4] < self.threshold1) & (anchors[:, :, 4] > self.threshold2) & \
                    (anchors[:, :, 3] < self.stride[len(self.stride)-i-1] * self.patch_size / 2) & (anchors[:, :, 2] < self.stride[len(self.stride)-i-1] * self.patch_size / 2)
            if not torch.any(mask) and not self.training:
                break # early-stop
            
            # compress feature
            compressed_feature_i = compressed_feature[len(self.stride)-i-1]

            # generate patches
            with torch.no_grad():
                new_batch_idx = torch.zeros((0), device=batch_idx.device, dtype=batch_idx.dtype)
                total_clustered_anchors = torch.zeros((0, 4), device=anchors.device, dtype=anchors.dtype)
                if epoch >= self.use_target_epoch or not self.training:
                    seletected_anchors = anchors[mask][:, :4]
                    seletected_anchors[:, 2:] = self.stride[len(self.stride)-i-1] * self.patch_size
                    total_clustered_anchors = box_ops.box_cxcywh_to_xyxy(seletected_anchors)
                    new_batch_idx = torch.repeat_interleave(batch_idx, torch.sum(mask, dim=1))
                    scores = anchors[mask][:, 4]
                    seletected_idx = torchvision.ops.batched_nms(total_clustered_anchors, scores, new_batch_idx, self.nms_threshold)
                    new_batch_idx = new_batch_idx[seletected_idx]
                    if self.dataset == 'crowdhuman':
                        seletected_idx = torch.cat([seletected_idx[new_batch_idx == j][:60] for j in range(B)])
                        new_batch_idx = torch.cat([new_batch_idx[new_batch_idx == j][:60] for j in range(B)])
                    total_clustered_anchors = total_clustered_anchors[seletected_idx]
                crop_pos = self.generate_crop_pos(total_clustered_anchors, imgs_whwh[new_batch_idx], self.stride[len(self.stride)-i-1]) # patch_size, xy [m, 2]

            # get output anchors
            if epoch < self.use_target_epoch and self.training:
                output_mask = (anchors[:, :, 4] > self.use_anchor_threshold)
            else:
                output_mask = (anchors[:, :, 4] > self.use_anchor_threshold) & (~mask) # (num_patch, anchors_per_patch)
            for j in range(B):
                if i == 1:
                    if torch.sum(output_mask[j]) < self.min_anchor:
                        scores, idxs = torch.sort(anchors[j, :, 4], descending=True)
                        idxs = idxs[:self.min_anchor]
                        output_anchors[j].append(anchors[j, idxs, :])
                    else:
                        output_anchors[j].append(anchors[j, output_mask[j], :])
                elif torch.any(batch_idx==j):
                    anchors_for_each_image = anchors[batch_idx==j].flatten(0, 1) # (n*anchors_per_patch, 5)
                    mask_for_each_image = output_mask[batch_idx==j].flatten(0, 1) # (n*anchors_per_patch)
                    output_anchors[j].append(anchors_for_each_image[mask_for_each_image])
            
            batch_idx = new_batch_idx
            # padding with gt patch
            if self.training:
                crop_pos, batch_idx = self.generate_targets_crop_pos(crop_pos, batch_idx, targets, imgs_whwh, self.stride[len(self.stride)-i-1], epoch)

            batch_idx_for_loss = batch_idx.clone()
            batch_idx_for_loss[batch_idx_for_loss >= self.targets_offset] -= self.targets_offset
            patches = torch.stack([compressed_feature_i[k, :, h:h+self.patch_size, w:w+self.patch_size] \
                for (w, h), k in zip(crop_pos, batch_idx_for_loss)], dim=0) # [num_patch, compress_dim, patch_size, patch_size]
            # predict anchors
            patches = patches.flatten(2).permute(0, 2, 1) # [num_patch, patch_size*patch_size, compress_dim]
            patches = self.layernorm[i](patches)
            patches += self.pos_embedding[i].weight.clone().unsqueeze(0)
            patches = patches.flatten(1) # [num_patch, patch_size*patch_size, compress_dim]
            anchors = self.proj_patch(patches)
            anchors = anchors.reshape(-1, self.num_pred_anchor2, 5) # 'cxcywh' conf
            anchors[..., :4] = torch.sigmoid(anchors[..., :4]) # '0-1'
            anchors_for_loss.append((anchors, crop_pos, batch_idx_for_loss))

            # change anchors to image size
            anchors = anchors.clone().detach()
            anchors = anchors[batch_idx < self.targets_offset]
            crop_pos = crop_pos[batch_idx < self.targets_offset]
            batch_idx = batch_idx[batch_idx < self.targets_offset]
            anchors[..., 4] = torch.sigmoid(anchors[..., 4]) # '0-1'
            anchors[..., :4] = anchors[..., :4] * (self.patch_size / (imgs_whwh[batch_idx] / self.stride[len(self.stride)-i-1]))
            anchors[..., :2] = anchors[..., :2] + crop_pos.unsqueeze(1) / (imgs_whwh[batch_idx][..., :2] / self.stride[len(self.stride)-i-1])
            out_of_boundary_mask = (anchors[..., 0] >= 1) | (anchors[..., 1] >= 1)
            out_of_boundary_anchors = anchors[out_of_boundary_mask]
            out_of_boundary_anchors[:, 4] = 0
            anchors[out_of_boundary_mask] = out_of_boundary_anchors
            anchors[..., :4] = anchors[..., :4] * imgs_whwh[batch_idx] # change to image size
            too_small_mask = (anchors[..., 2] < 3) | (anchors[..., 3] < 3)
            too_small_anchors = anchors[too_small_mask]
            too_small_anchors[:, 4] = 0
            anchors[too_small_mask] = too_small_anchors

        # get output anchors
        output_mask = (anchors[:, :, 4] > self.use_anchor_threshold) # (num_patch, anchors_per_patch)
        for j in range(B):
            if i == 1:
                if torch.sum(output_mask[j]) < self.min_anchor:
                    scores, idxs = torch.sort(anchors[j, :, 4], descending=True)
                    idxs = idxs[:self.min_anchor]
                    output_anchors[j].append(anchors[j, idxs, :])
                else:
                    output_anchors[j].append(anchors[j, output_mask[j], :])

            elif torch.any(batch_idx==j):
                anchors_for_each_image = anchors[batch_idx==j].flatten(0, 1) # (n*anchors_per_patch, 5)
                mask_for_each_image = output_mask[batch_idx==j].flatten(0, 1) # (n*anchors_per_patch)
                output_anchors[j].append(anchors_for_each_image[mask_for_each_image])
        
        output_anchors_without_cat = output_anchors
        # padding output anchors
        output_anchors = [torch.cat(output) for output in output_anchors]
        for i in range(B):
            if len(output_anchors[i]) > self.max_anchor:
                scores, idxs = torch.sort(output_anchors[i][ :, 4], descending=True)
                output_anchors[i] = output_anchors[i][idxs[:self.max_anchor]]
        output_len = [len(output) for output in output_anchors]
        max_len = max(output_len)
        padded_output_anchors = []
        valid_mask = []
        for i in range(B):
            padded_anchors = torch.tensor([[0.5, 0.5, 1, 1]], dtype=anchors.dtype, device=anchors.device)
            padded_anchors = padded_anchors.repeat(max_len-output_len[i], 1) * imgs_whwh[i]
            padded_output_anchors.append(torch.cat([output_anchors[i][:, :4], padded_anchors], dim=0))
            mask = torch.zeros((max_len), dtype=torch.bool, device=anchors.device)
            mask[:output_len[i]] = True
            valid_mask.append(mask)
        padded_output_anchors = torch.stack(padded_output_anchors, dim=0) # [B, max_len, 4] 'cxcywh' image size
        valid_mask = torch.stack(valid_mask, dim=0) # [B, max_len]

        return anchors_for_loss, padded_output_anchors, valid_mask, max_len, output_anchors_without_cat
    

    def fast_inference(self, x: List[Tensor], imgs_whwh: Tensor, targets=None, epoch=-1):
        '''
        feature: (B, d_model, H, W)
        mask: (B, H, W)
        '''

        B = x[0].shape[0]
        output_anchors = [[] for _ in range(B)]
        compressed_feature = [self.compress[-1](x[-1])]# (B, compress_dim, H, W)

        # interpolate
        interpolated_feature = []
        img_H = imgs_whwh[:, 0, 1] / self.stride[-1]
        img_W = imgs_whwh[:, 0, 0] / self.stride[-1]
        for i, (h, w) in enumerate(zip(img_H, img_W)):
            interpolated_feature.append(F.interpolate(compressed_feature[-1][i:i+1, :, :int(torch.round(h)), :int(torch.round(w))], size=(self.interpolate_size, self.interpolate_size), mode='bilinear')) # (1, compress_dim, interpolate_size, interpolate_size)
        interpolated_feature = torch.cat(interpolated_feature, dim=0) # (B, compress_dim, interpolate_size, interpolate_size)

        # get P6 patch
        P6_feature = self.P6(F.relu(interpolated_feature)).unsqueeze(1) # (B, 1, compress_dim, patch_size, patch_size)
        
        # get P5 patch
        image_token = torch.stack([interpolated_feature[:, :, pos[0]:pos[0]+self.patch_size_interpolate, pos[1]:pos[1]+self.patch_size_interpolate] for pos in self.base_crop_pos], dim=1) # (B, num_crops, compress_dim, patch_size, patch_size)
        image_token = torch.cat([image_token, P6_feature], dim=1) # (B, num_crops, compress_dim, patch_size, patch_size)
        image_token = image_token.flatten(3).permute(0, 1, 3, 2) # (B, num_crops, patch_size*patch_size, compress_dim)
        image_token = self.layernorm[0](image_token)
        image_token += self.pos_embedding[0].weight.clone().unsqueeze(0).unsqueeze(0)
        image_token = image_token.flatten(2) # (B, num_crops, patch_size*patch_size*compress_dim)
        # pred dynamic anchor
        total_anchors = self.proj_all(image_token)
        total_anchors = total_anchors.reshape(B, -1, self.num_pred_anchor1, 5) # 'cxcywh' conf
        detached_anchors = torch.sigmoid(total_anchors)
        
        # change the coordinate to full image
        detached_anchors[:, :4, :, :4] = detached_anchors[:, :4, :, :4] * (self.patch_size_interpolate / self.interpolate_size)
        detached_anchors[:, 1, :, 0] = (self.interpolate_size - self.patch_size_interpolate) / self.interpolate_size + detached_anchors[:, 1, :, 0] # top-right cx
        detached_anchors[:, 2, :, 1] = (self.interpolate_size - self.patch_size_interpolate) / self.interpolate_size + detached_anchors[:, 2, :, 1] # bottom-left cy
        detached_anchors[:, 3, :, :2] = (self.interpolate_size - self.patch_size_interpolate) / self.interpolate_size + detached_anchors[:, 3, :, :2] # bottom-right cxcy

        anchors = detached_anchors.flatten(1, 2)
        anchors[..., :4] = anchors[..., :4] * imgs_whwh # change to image size
        too_small_mask = (anchors[..., 2] < 3) | (anchors[..., 3] < 3)
        too_small_anchors = anchors[too_small_mask]
        too_small_anchors[:, 4] = 0
        anchors[too_small_mask] = too_small_anchors
        
        # pred on other feature pyramid level
        for i in range(1, len(self.stride)):
            if self.used_inference_level == i:
                break # manually stop
            
            mask = (anchors[:, :, 4] < self.threshold1) & (anchors[:, :, 4] > self.threshold2) & \
                    (anchors[:, :, 3] < self.stride[len(self.stride)-i-1] * self.patch_size / 2) & (anchors[:, :, 2] < self.stride[len(self.stride)-i-1] * self.patch_size / 2)
            if not torch.sum(mask) < 4:
                break
            
            # compress feature
            compressed_feature_i = self.compress[len(self.stride)-i-1](x[len(self.stride)-i-1])

            # generate patches
            seletected_anchors = anchors[mask][:, :4]
            seletected_anchors[:, 2:] = self.stride[len(self.stride)-i-1] * self.patch_size
            total_clustered_anchors = box_ops.box_cxcywh_to_xyxy(seletected_anchors)
            scores = anchors[mask][:, 4]
            seletected_idx = torchvision.ops.nms(total_clustered_anchors, scores, self.nms_threshold)[:15]
            total_clustered_anchors = total_clustered_anchors[seletected_idx]
            crop_pos = self.generate_crop_pos(total_clustered_anchors, imgs_whwh, self.stride[len(self.stride)-i-1]) # patch_size, xy [m, 2]

            output_mask = (anchors[:, :, 4] > self.use_anchor_threshold) & (~mask) # (num_patch, anchors_per_patch)
            for j in range(B):
                if i == 1:
                    if torch.sum(output_mask[j]) < self.min_anchor:
                        scores, idxs = torch.sort(anchors[j, :, 4], descending=True)
                        idxs = idxs[:self.min_anchor]
                        output_anchors[j].append(anchors[j, idxs, :])
                    else:
                        output_anchors[j].append(anchors[j, output_mask[j], :])
                else:
                    anchors_for_each_image = anchors.flatten(0, 1) # (n*anchors_per_patch, 5)
                    mask_for_each_image = output_mask.flatten(0, 1) # (n*anchors_per_patch)
                    output_anchors[j].append(anchors_for_each_image[mask_for_each_image])
            
            patches = torch.stack([compressed_feature_i[0, :, h:h+self.patch_size, w:w+self.patch_size] \
                for (w, h) in crop_pos], dim=0) # [num_patch, compress_dim, patch_size, patch_size]
            # predict anchors
            patches = patches.flatten(2).permute(0, 2, 1) # [num_patch, patch_size*patch_size, compress_dim]
            patches = self.layernorm[i](patches)
            patches += self.pos_embedding[i].weight.clone().unsqueeze(0)
            patches = patches.flatten(1) # [num_patch, patch_size*patch_size, compress_dim]
            anchors = self.proj_patch(patches)
            anchors = anchors.reshape(-1, self.num_pred_anchor2, 5) # 'cxcywh' conf
            anchors = torch.sigmoid(anchors) # '0-1'

            # change anchors to image size
            anchors[..., :4] = anchors[..., :4] * (self.patch_size / (imgs_whwh / self.stride[len(self.stride)-i-1]))
            anchors[..., :2] = anchors[..., :2] + crop_pos.unsqueeze(1) / (imgs_whwh[..., :2] / self.stride[len(self.stride)-i-1])
            out_of_boundary_mask = (anchors[..., 0] >= 1) | (anchors[..., 1] >= 1)
            out_of_boundary_anchors = anchors[out_of_boundary_mask]
            out_of_boundary_anchors[:, 4] = 0
            anchors[out_of_boundary_mask] = out_of_boundary_anchors
            anchors[..., :4] = anchors[..., :4] * imgs_whwh # change to image size
            too_small_mask = (anchors[..., 2] < 3) | (anchors[..., 3] < 3)
            too_small_anchors = anchors[too_small_mask]
            too_small_anchors[:, 4] = 0
            anchors[too_small_mask] = too_small_anchors

        # get output anchors
        output_mask = (anchors[:, :, 4] > self.use_anchor_threshold) # (num_patch, anchors_per_patch)
        for j in range(B):
            if i == 1:
                if torch.sum(output_mask[j]) < self.min_anchor:
                    scores, idxs = torch.sort(anchors[j, :, 4], descending=True)
                    idxs = idxs[:self.min_anchor]
                    output_anchors[j].append(anchors[j, idxs, :])
                else:
                    output_anchors[j].append(anchors[j, output_mask[j], :])

            else:
                anchors_for_each_image = anchors.flatten(0, 1) # (n*anchors_per_patch, 5)
                mask_for_each_image = output_mask.flatten(0, 1) # (n*anchors_per_patch)
                output_anchors[j].append(anchors_for_each_image[mask_for_each_image])
        
        # padding output anchots
        output_anchors = [torch.cat(output) for output in output_anchors]
        for i in range(B):
            if len(output_anchors[i]) > self.max_anchor:
                scores, idxs = torch.sort(output_anchors[i][ :, 4], descending=True)
                output_anchors[i] = output_anchors[i][idxs[:self.max_anchor]]

        output_anchors = output_anchors[0].unsqueeze(0) # [B, max_len, 4] 'cxcywh' image size
        max_len = output_anchors.shape[1]
        valid_mask = torch.ones((1, max_len), dtype=torch.bool, device=anchors.device) # [B, max_len]

        return None, output_anchors[..., :4], valid_mask, max_len, None



def build_anchorgenerator(args, stride):
    return AnchorGenerator(
        d_model=args.hidden_dim,
        num_pred_anchor1=args.num_pred_anchor1,
        num_pred_anchor2=args.num_pred_anchor2,
        patch_size_interpolate=args.patch_size_interpolate,
        patch_size=args.patch_size,
        activation=args.activation,
        compress_dim=args.compress_dim,
        stride=stride,
        threshold1=args.threshold1,
        threshold2=args.threshold2,
        nms_threshold=args.patch_nms_threshold,
        use_target_epoch=args.use_target_epoch,
        use_anchor_threshold=args.use_anchor_threshold,
        used_inference_level=args.used_inference_level,
        dataset=args.dataset_file
    )


