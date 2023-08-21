
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from copy import deepcopy
from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized



def quality_focal_loss_with_ignore(
          pred,          # (B, num_query, 91)
          label,         # (B, num_query) 0-91: 91 is neg, 0-90 is positive
          score,         # (B, num_query) reg target 0-1, only positive is non-zero
          a_weight,
          ignore,        # (B, num_query) reg target 0-1, only positive is non-zero
          background_id, # i.e. 91
          num_boxes, 
          beta=2.0,):

    B, num_query = pred.shape[:2]
    pred = pred.flatten(0, 1) # (n, 91)
    label = label.flatten(0, 1) # (n) 0-91: 91 is neg, 0-90 is positive
    score = score.flatten(0, 1) # (n) reg target 0-1, only positive is good
    a_weight = a_weight.flatten(0, 1)
    # all goes to 0
    pred_sigmoid = pred.sigmoid()
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
           pred, zerolabel, reduction='none') * pt.pow(beta)

    pos = torch.nonzero(label != background_id).squeeze(1)
    a = pos
    b = label[pos].long()
    
    # positive goes to bbox quality
    pt = score[a] - pred_sigmoid[a, b]
    loss[a,b] = F.binary_cross_entropy_with_logits(
           pred[a,b], score[a], reduction='none') * pt.pow(beta) * a_weight[a]

    loss = loss.reshape(B, num_query, -1)
    loss[ignore > 0] *= 0 # ignore high iou anchor
    loss = loss.mean(1).sum() / num_boxes
    return loss



def sigmoid_focal_loss(inputs, targets, a_weight, num_boxes, idx, target_classes_o, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none",)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    loss[idx[0], idx[1], target_classes_o] = loss[idx[0], idx[1], target_classes_o] * a_weight[idx]

    return loss

def DW_focal_loss(
          pred,          # (B, num_query, 91)
          label,         # (B, num_query) 0-91: 91 is neg, 0-90 is positive
          pos_weight,    # (B, num_query) weight 0-1, only positive is non -1
          neg_weight,    # (B, num_query) weight 0-1, only positive is non -1
          a_weight, 
          background_id, # i.e. 91
          beta=2.0,):

    B, num_query = pred.shape[:2]
    pred = pred.flatten(0, 1) # (n, 91)
    label = label.flatten(0, 1) # (n) 0-91: 91 is neg, 0-90 is positive
    pos_weight = pos_weight.flatten(0, 1) # (n)
    neg_weight = neg_weight.flatten(0, 1) # (n)
    a_weight = a_weight.flatten(0, 1)
    # all goes to 0
    pred_sigmoid = pred.sigmoid()
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
           pred, zerolabel, reduction='none') * pt.pow(beta)

    pos = torch.nonzero(label != background_id).squeeze(1)
    a = pos
    b = label[pos].long()
    
    # positive goes to bbox quality
    w_pos = pos_weight[a]
    w_neg = neg_weight[a]
    w_a = a_weight[a]
    loss1 = F.binary_cross_entropy_with_logits(
           pred[a,b], torch.ones_like(pred[a,b]), reduction='none') * w_pos
    loss2 = F.binary_cross_entropy_with_logits(
           pred[a,b], torch.zeros_like(pred[a,b]), reduction='none') * w_neg
 
    loss[a,b] = (loss1 + loss2) * w_a
    loss = loss.reshape(B, num_query, -1)
    # loss = loss.mean(1).sum() / num_boxes
    return loss


@torch.no_grad()
def split_targets(targets, patch_size, crop_pos, image_size=30):
    targets_split = []
    for t in targets:
        temp1 = []
        if len(t['boxes']) == 0:
            targets_split += [deepcopy(t) for _ in range(crop_pos.shape[0])]
            continue
        for pos in crop_pos:
            mask = (t['boxes'][..., 0] >= (pos[1] / image_size)) & (t['boxes'][..., 0] <= ((pos[1] + patch_size) / image_size)) \
                 & (t['boxes'][..., 1] >= (pos[0] / image_size)) & (t['boxes'][..., 1] <= ((pos[0] + patch_size) / image_size))
            boxes = t['boxes'][mask, :]
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
            boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], min=(pos[1] / image_size), max=((pos[1]+patch_size) / image_size))
            boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], min=(pos[0] / image_size), max=((pos[0]+patch_size) / image_size))
            boxes[:, 0::2] = (boxes[:, 0::2] - (pos[1] / image_size)) * (image_size / patch_size) # image coordinate to patch coordinate
            boxes[:, 1::2] = (boxes[:, 1::2] - (pos[0] / image_size)) * (image_size / patch_size) # image coordinate to patch coordinate
            boxes = box_ops.box_xyxy_to_cxcywh(boxes)
            labels = t['labels'][mask]
            temp1.append({'boxes': boxes, 'labels': labels})
        targets_split += temp1
    return targets_split


@torch.no_grad()
def get_P6_targets(targets, training):
    targets_split = []
    for t in targets:
        targets_hflip = []
        targets_vflip = []
        targets_vhflip = []
        if len(t['boxes']) == 0:
            if training:
                targets_split += [deepcopy(t), deepcopy(t), deepcopy(t), deepcopy(t)]
            else:
                targets_split += [deepcopy(t)]
            continue
        boxes = t['boxes']
        hflip_boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1], dtype=boxes.dtype, device=boxes.device) + torch.as_tensor([1, 0, 1, 0], dtype=boxes.dtype, device=boxes.device)
        vflip_boxes = boxes[:, [0, 3, 2, 1]] * torch.as_tensor([1, -1, 1, -1], dtype=boxes.dtype, device=boxes.device) + torch.as_tensor([0, 1, 0, 1], dtype=boxes.dtype, device=boxes.device)
        vhflip_boxes = hflip_boxes[:, [0, 3, 2, 1]] * torch.as_tensor([1, -1, 1, -1], dtype=boxes.dtype, device=boxes.device) + torch.as_tensor([0, 1, 0, 1], dtype=boxes.dtype, device=boxes.device)
        labels = t['labels']
        targets_split.append({'boxes': boxes, 'labels': labels})
        targets_hflip.append({'boxes': hflip_boxes, 'labels': labels})
        targets_vflip.append({'boxes': vflip_boxes, 'labels': labels})
        targets_vhflip.append({'boxes': vhflip_boxes, 'labels': labels})
        if training:
            targets_split += targets_hflip
            targets_split += targets_vflip
            targets_split += targets_vhflip
    return targets_split



@torch.no_grad()
def get_anchor_targets(targets, outputs, stride, patch_size, imgs_whwh):
    anchor_targets = []
    stride = deepcopy(stride)
    stride.reverse()
    stride = stride[:len(outputs)]
    for i, s in enumerate(stride):
        if i == 0: # first feature pyramid level
            continue
        else: # other feature pyramid levels
            _, crop_pos, batch_idx = outputs[i]
            normalized_top_left_crop_pos = crop_pos * s / imgs_whwh[batch_idx][:, 0, :2]
            normalized_bottom_right_crop_pos = (crop_pos + patch_size) * s / imgs_whwh[batch_idx][:, 0, :2]
            max_size = patch_size * s / imgs_whwh[batch_idx][:, 0, :2]
            img_H = imgs_whwh[:, 0, 1] / s
            img_W = imgs_whwh[:, 0, 0] / s
            for j, bid in enumerate(batch_idx): # for each patch
                boxes = targets[bid]['boxes']
                if len(boxes) == 0:
                    anchor_targets.append({'boxes': targets[bid]['boxes'], 'labels': targets[bid]['labels']})
                else:
                    mask = (boxes[:, 0] >= normalized_top_left_crop_pos[j, 0]) & (boxes[:, 1] >= normalized_top_left_crop_pos[j, 1]) \
                        & (boxes[:, 0] <= normalized_bottom_right_crop_pos[j, 0]) & (boxes[:, 1] <= normalized_bottom_right_crop_pos[j, 1]) \
                        & (boxes[:, 2] <= max_size[j, 0]) & (boxes[:, 3] <= max_size[j, 1])
                    boxes = boxes[mask, :]
                    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
                    boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], min=(crop_pos[j, 0] / img_W[bid]), max=((crop_pos[j, 0]+patch_size) / img_W[bid]))
                    boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], min=(crop_pos[j, 1] / img_H[bid]), max=((crop_pos[j, 1]+patch_size) / img_H[bid]))
                    boxes[:, 0::2] = (boxes[:, 0::2] - (crop_pos[j, 0] / img_W[bid])) * (img_W[bid] / patch_size) # image coordinate to patch coordinate
                    boxes[:, 1::2] = (boxes[:, 1::2] - (crop_pos[j, 1] / img_H[bid])) * (img_H[bid] / patch_size) # image coordinate to patch coordinate
                    boxes = box_ops.box_xyxy_to_cxcywh(boxes)
                    labels = targets[bid]['labels'][mask]
                    anchor_targets.append({'boxes': boxes, 'labels': labels})
    return anchor_targets


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, anchor_matcher, rpn_matcher, decoder_matcher, weight_dict, eos_coef, losses, patch_size_interpolate, patch_size, stride, gamma1=1.0, gamma2=1.0, area_weight=4.0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.anchor_matcher = anchor_matcher
        self.rpn_matcher = rpn_matcher
        self.decoder_matcher = decoder_matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.patch_size_interpolate = patch_size_interpolate
        self.patch_size = patch_size
        self.stride = stride
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.area_weight = area_weight

    def loss_labels(self, outputs, targets, indices, pos_weight, neg_weight, a_weight, ignore, num_boxes, log=True, valid_mask=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [bs, num_query, num_class]

        with torch.no_grad():
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o # [bs, num_query]

        loss_ce = DW_focal_loss(src_logits, target_classes, pos_weight, neg_weight, a_weight, self.num_classes)
        if valid_mask is not None:
            loss_ce[~valid_mask] *= 0
        loss_ce = loss_ce.sum() / num_boxes
        losses = {'loss_ce': loss_ce}

        with torch.no_grad():
            if log:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
                total_gt = 0.
                for t, (_, i) in zip(targets, indices):
                    total_gt += len(t['boxes'])
                losses['avg_pos_weight'] = torch.mean(pos_weight[idx]) if total_gt != 0. and len(idx[0]) != 0 else torch.tensor(0., device=loss_ce.device)
                losses['avg_neg_weight'] = torch.mean(neg_weight[idx]) if total_gt != 0. and len(idx[0]) != 0 else torch.tensor(0., device=loss_ce.device)
        return losses

    def loss_boxes(self, outputs, targets, indices, pos_weight, neg_weight, a_weight, ignore, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        weight = pos_weight[idx]
        area_weight = a_weight[idx]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = (loss_bbox * weight.unsqueeze(-1) * area_weight.unsqueeze(-1)).sum() / num_boxes

        tgt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            tgt_boxes_xyxy)[1])
        losses['loss_giou'] = (loss_giou * weight * area_weight).sum() / num_boxes

        return losses

    def loss_anchor_labels(self, outputs, targets, indices, pos_weight, neg_weight, a_weight, ignore, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "anchors" in outputs
        src_logits = outputs["anchors"][..., 4:] # [bs, num_query, 1]

        with torch.no_grad():
            idx = self._get_src_permutation_idx(indices)
            target_classes = torch.full(src_logits.shape[:2], 1, # class agnostic
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = 0 # [bs, num_query]

        loss_ce = quality_focal_loss_with_ignore(src_logits, target_classes, pos_weight, a_weight, ignore, 1, num_boxes) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_anchor_boxes(self, outputs, targets, indices, pos_weight, neg_weight, a_weight, ignore, num_boxes):
        """Compute the losses related to the anchors, the centerness (iou) loss, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "anchors" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["anchors"][idx][..., :4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        weight = neg_weight[idx]
        area_weight = a_weight[idx]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = (loss_bbox * weight.unsqueeze(-1) * area_weight.unsqueeze(-1)).sum() / num_boxes

        tgt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
        iou, giou = box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), tgt_boxes_xyxy)
        giou = torch.diag(giou)
        loss_giou = 1 - giou
        losses['loss_giou'] = (loss_giou * weight * area_weight).sum() / num_boxes

        with torch.no_grad():
            iou = torch.diag(iou)
            if len(iou) > 0:
                losses['iou'] = torch.mean(iou)
            else:
                losses['iou'] = torch.tensor(1., device=loss_giou.device)
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @torch.no_grad()
    def get_weight(self, type, outputs, targets, indices, inverse=False):
        alpha = 2
        t = lambda x: 1/(0.5**alpha-1)*x**alpha - 1/(0.5**alpha-1)

        def normalize(x): 
            y = t(x)
            y[x<0.5] = 1
            return y

        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if inverse:
            area_weight = (self.area_weight - 1) * torch.sqrt(target_boxes[:, 2] * target_boxes[:, 3]) + 1
        else:
            area_weight = self.area_weight - torch.sqrt(target_boxes[:, 2] * target_boxes[:, 3]) * (self.area_weight - 1)

        if type == 'anchor': # normalize weigth to prevent it to be too small
            normalizer = 1./ torch.sigmoid(torch.tensor(3., device=outputs['anchors'].device))
            base_value = torch.sigmoid((torch.tensor(0., device=outputs['anchors'].device) - (1./ 3)) * 4.5) * normalizer
            pos_weight = torch.full(outputs['anchors'].shape[:2], -1, dtype=outputs['anchors'].dtype, device=outputs['anchors'].device) # [bs, num_query]
            neg_weight = torch.full(outputs['anchors'].shape[:2], -1, dtype=outputs['anchors'].dtype, device=outputs['anchors'].device)
            a_weight = torch.full(outputs['anchors'].shape[:2], -1, dtype=outputs['anchors'].dtype, device=outputs['anchors'].device) # [bs, num_query]
            ignore = torch.full(outputs['anchors'].shape[:2], 0, dtype=outputs['anchors'].dtype, device=outputs['anchors'].device)
            if len(target_boxes) == 0:
                return pos_weight, neg_weight, a_weight, ignore

            src_boxes = outputs['anchors'][idx][..., :4]
            src_logits = outputs['anchors'][..., 4] # [bs, num_query]
            box_logits = torch.sigmoid(src_logits[idx])
            ious = torch.diag(box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            pos_weight[idx] = ious
            neg_weight[idx] = torch.sigmoid(((box_logits ** self.gamma1) * (ious ** self.gamma2) - (1./ 3)) * 4.5) * normalizer
            a_weight[idx] = area_weight

            for i, tgt in enumerate(targets):
                if len(tgt['boxes']) == 0:
                    continue
                all_iou = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(outputs['anchors'][i, :, :4]), box_ops.box_cxcywh_to_xyxy(tgt['boxes']))[0] # (num_query, num_gt)
                max_iou_per_anchor = torch.max(all_iou, dim=1)[0]
                max_iou_per_anchor[max_iou_per_anchor < 0.5] = 0
                ignore[i] = max_iou_per_anchor
            ignore[idx] = 0
            return pos_weight, neg_weight, a_weight, ignore
        else:
            normalizer = 1./ torch.sigmoid(torch.tensor(3., device=outputs['pred_boxes'].device))
            base_value = torch.sigmoid((torch.tensor(0., device=outputs['pred_boxes'].device) - (1./ 3)) * 4.5) * normalizer
            pos_weight = torch.full(outputs['pred_logits'].shape[:2], -1, dtype=outputs['pred_logits'].dtype, device=outputs['pred_logits'].device) # [bs, num_query]
            neg_weight = torch.full(outputs['pred_logits'].shape[:2], -1, dtype=outputs['pred_logits'].dtype, device=outputs['pred_logits'].device)
            a_weight = torch.full(outputs['pred_logits'].shape[:2], -1, dtype=outputs['pred_logits'].dtype, device=outputs['pred_logits'].device) # [bs, num_query]
            if len(target_boxes) == 0:
                return pos_weight, neg_weight, a_weight, None

            src_boxes = outputs['pred_boxes'][idx]
            src_logits = outputs['pred_logits'] # [bs, num_query, num_class]
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            box_logits = torch.sigmoid(src_logits[idx[0], idx[1], target_classes_o])
            ious = torch.diag(box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))[0])
            # [0, 1] -> [-1/3, 2/3] -> [-3/2, 3] -> [0.1824, 0.9526] -> [0.1915, 1]
            pos_weight[idx] = torch.sigmoid(((box_logits ** self.gamma1) * (ious ** self.gamma2) - (1./ 3)) * 4.5) * normalizer # there are hyperparameters to tune
            neg_weight[idx] = torch.sigmoid((normalize(ious ** self.gamma2) * (box_logits ** self.gamma1) - (1./ 3)) * 4.5) * normalizer - base_value # there are hyperparameters to tune [0, 0.8085]
            a_weight[idx] = area_weight
            return pos_weight, neg_weight, a_weight, None


    def get_loss(self, loss, outputs, targets, indices, pos_weight, neg_weight, a_weight, ignore, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'anchor_labels': self.loss_anchor_labels,
            'anchor_boxes': self.loss_anchor_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, pos_weight, neg_weight, a_weight, ignore, num_boxes, **kwargs)

    def forward(self, outputs, targets, imgs_whwh, training):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}

        if outputs['dn']['dn_targets_boxes'] is not None:
            dn_outputs = []
            num_dn, num_proposals = outputs['dn']['dn_param']
            
            # extract dn components from output
            # rpn_dn
            dn_outputs.append({'pred_logits': outputs['rpn']['pred_logits'][:, :num_dn, :], 'pred_boxes': outputs['rpn']['pred_boxes'][:, :num_dn, :]})
            # aux_dn
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    dn_outputs.append({'pred_logits': aux_outputs['pred_logits'][:, :num_dn, :], 'pred_boxes': aux_outputs['pred_boxes'][:, :num_dn, :]})
            # decoder_dn
            dn_outputs.append({'pred_logits': outputs['pred_logits'][:, :num_dn, :], 'pred_boxes': outputs['pred_boxes'][:, :num_dn, :]})
            
            # delete dn components from matching part
            # rpn
            outputs['rpn']['pred_logits'] = outputs['rpn']['pred_logits'][:, num_dn:, :]
            outputs['rpn']['pred_boxes'] = outputs['rpn']['pred_boxes'][:, num_dn:, :]
            # aux
            if 'aux_outputs' in outputs:
                outputs['aux_outputs'] = [{'pred_logits': a['pred_logits'][:, num_dn:, :], 'pred_boxes': a['pred_boxes'][:, num_dn:, :]} for a in outputs['aux_outputs']]
            # decoder
            outputs['pred_logits'] = outputs['pred_logits'][:, num_dn:, :]
            outputs['pred_boxes'] = outputs['pred_boxes'][:, num_dn:, :]

            # compute dn loss
            bs = outputs['pred_boxes'].shape[0]
            dn_targets = [{'boxes': boxes, 'labels': labels} for boxes, labels in zip(outputs['dn']['dn_targets_boxes'], outputs['dn']['dn_targets_labels'])]
            dn_valid_mask = []
            for b in range(bs):
                mask = torch.zeros(num_dn, dtype=torch.bool, device=dn_outputs[0]['pred_logits'].device)
                mask[:len(outputs['dn']['dn_targets_boxes'][b])] = True
                dn_valid_mask.append(mask)
            dn_valid_mask = torch.stack(dn_valid_mask, dim=0)
            indices = [(torch.arange(0, len(boxes), dtype=torch.int64), torch.arange(0, len(boxes), dtype=torch.int64)) for boxes in outputs['dn']['dn_targets_boxes']]
            for i, dn_out in enumerate(dn_outputs):
                pos_weight, neg_weight, a_weight, ignore = self.get_weight('decoder', dn_out, dn_targets, indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False, "valid_mask": dn_valid_mask}
                    l_dict = self.get_loss(loss, dn_out, dn_targets, indices, pos_weight, neg_weight, a_weight, ignore, num_dn*bs, **kwargs)
                    l_dict = {'dn_' + k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Retrieve the matching between the outputs of the last layer and the targets
        decoder_targets = targets
        indices = self.decoder_matcher(outputs, decoder_targets, outputs['anchors']['valid_mask'])
        pos_weight, neg_weight, a_weight, ignore = self.get_weight('decoder', outputs, decoder_targets, indices)
        decoder_num_boxes = num_boxes
        # Compute all the requested losses
        for loss in self.losses:
            kwargs = {}
            if loss == 'labels':
                kwargs = {"valid_mask": outputs['anchors']['valid_mask']}
            losses.update(self.get_loss(loss, outputs, decoder_targets, indices, pos_weight, neg_weight, a_weight, ignore, decoder_num_boxes, **kwargs))
        
        # anchor
        # P6
        anchor_num_boxes = num_boxes
        anchor_image_targets = get_P6_targets(targets, training=training)
        anchor_image_outputs = {'anchors': outputs['anchors']["anchors_for_loss"][0][0].flatten(0, 1)}

        indices = self.anchor_matcher(anchor_image_outputs, anchor_image_targets, inverse=True)
        pos_weight, neg_weight, a_weight, ignore = self.get_weight('anchor', anchor_image_outputs, anchor_image_targets, indices, inverse=True)
        
        for loss in self.losses:
            anchor_losses = self.get_loss('anchor_' + loss, anchor_image_outputs, anchor_image_targets, indices, pos_weight, neg_weight, a_weight, ignore, anchor_num_boxes)
            anchor_losses = {k + '_anchor': v for k, v in anchor_losses.items()}
            losses.update(anchor_losses)
        losses['iou_anchor'] = [losses['iou_anchor']]

        # P5
        anchor_num_boxes = num_boxes
        anchor_image_targets = split_targets(targets, patch_size=self.patch_size_interpolate, crop_pos=outputs['anchors']["anchors_for_loss"][1][1])
        anchor_image_outputs = {'anchors': outputs['anchors']["anchors_for_loss"][1][0].flatten(0, 1)}

        indices = self.anchor_matcher(anchor_image_outputs, anchor_image_targets)
        pos_weight, neg_weight, a_weight, ignore = self.get_weight('anchor', anchor_image_outputs, anchor_image_targets, indices)
        
        for loss in self.losses:
            anchor_losses = self.get_loss('anchor_' + loss, anchor_image_outputs, anchor_image_targets, indices, pos_weight, neg_weight, a_weight, ignore, anchor_num_boxes)
            anchor_losses = {k + '_anchor': v for k, v in anchor_losses.items()}
            for k, v in anchor_losses.items():
                if k == 'iou_anchor':
                    losses[k].append(v)
                else:
                    losses[k] += v
        # p4-p3
        anchor_patch_targets = get_anchor_targets(targets, outputs['anchors']["anchors_for_loss"][1:], self.stride, self.patch_size, imgs_whwh)
        if len(anchor_patch_targets) != 0:
            anchor_patch_outputs = [o[0] for o in outputs['anchors']["anchors_for_loss"]][2:]
            anchor_patch_outputs = {'anchors': torch.cat(anchor_patch_outputs, dim=0)}

            indices = self.anchor_matcher(anchor_patch_outputs, anchor_patch_targets)
            pos_weight, neg_weight, a_weight, ignore = self.get_weight('anchor', anchor_patch_outputs, anchor_patch_targets, indices)
            
            for loss in self.losses:
                anchor_losses = self.get_loss('anchor_' + loss, anchor_patch_outputs, anchor_patch_targets, indices, pos_weight, neg_weight, a_weight, ignore, anchor_num_boxes)
                anchor_losses = {k + '_anchor': v * 4 / len(anchor_patch_outputs['anchors']) for k, v in anchor_losses.items()}
                for k, v in anchor_losses.items():
                    if k == 'iou_anchor':
                        losses[k].append(v / 4 * len(anchor_patch_outputs['anchors']))
                    else:
                        losses[k] += v
        losses['iou_anchor'] = torch.mean(torch.stack(losses['iou_anchor']))
        
        # log anchor stat
        with torch.no_grad():
            losses['avg_anchor'] = torch.sum(outputs['anchors']['valid_mask']) / len(targets)
            losses['avg_patch'] = torch.tensor((len(anchor_image_targets) + len(anchor_patch_targets)) / len(targets), device=losses['loss_ce_anchor'].device)
            for i in range(len(outputs['anchors']["anchors_for_loss"])):
                losses['avg_patch_level_%d' % (i)] = torch.tensor(len(outputs['anchors']["anchors_for_loss"][i][2]) / len(targets), device=losses['loss_ce_anchor'].device)
            for j in range(i+1, len(self.stride)):
                losses['avg_patch_level_%d' % (j)] = torch.tensor(0., device=losses['loss_ce_anchor'].device)

        # rpn
        rpn_num_boxes = num_boxes
        rpn_targets = targets
        indices = self.rpn_matcher(outputs['rpn'], rpn_targets, outputs['anchors']['valid_mask'])
        pos_weight, neg_weight, a_weight, ignore = self.get_weight('rpn', outputs['rpn'], rpn_targets, indices)
        
        for loss in self.losses:
            kwargs = {}
            if loss == 'labels':
                kwargs = {"valid_mask": outputs['anchors']['valid_mask']}
            rpn_losses = self.get_loss(loss, outputs['rpn'], rpn_targets, indices, pos_weight, neg_weight, a_weight, ignore, rpn_num_boxes, **kwargs)
            rpn_losses = {k + '_rpn': v for k, v in rpn_losses.items()}
            losses.update(rpn_losses)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.decoder_matcher(aux_outputs, decoder_targets, outputs['anchors']['valid_mask'])
                pos_weight, neg_weight, a_weight, ignore = self.get_weight('decoder', aux_outputs, decoder_targets, indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False, "valid_mask": outputs['anchors']['valid_mask']}
                    l_dict = self.get_loss(loss, aux_outputs, decoder_targets, indices, pos_weight, neg_weight, a_weight, ignore, decoder_num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_criterion(args, num_classes, anchor_matcher, rpn_matcher, decoder_matcher, weight_dict, eos_coef, losses, stride):
    return SetCriterion(num_classes, anchor_matcher=anchor_matcher, rpn_matcher=rpn_matcher, decoder_matcher=decoder_matcher, patch_size_interpolate=args.patch_size_interpolate,
                         weight_dict=weight_dict, eos_coef=eos_coef, losses=losses, patch_size=args.patch_size, stride=stride, gamma1=args.gamma1, gamma2=args.gamma2, area_weight=args.area_weight)
