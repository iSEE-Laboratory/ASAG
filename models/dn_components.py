
import torch

def prepare_for_dn(use_dn, training, dn_args, anchors, valid_anchor_mask, imgs_whwh):
    """
    prepare for dn components in forward function
    Args:
        dn_args: (targets, num_dn, box_noise_scale) from engine input
        use_dn: whether using dn
        training: whether it is training or inference
        anchors: original matching anchors from anchor generators (B, num_anchors, 4), 'cxcywh' image_size
        valid_anchor_mask: (B, num_anchors)
        imgs_whwh: (B, 1, 4)

    Returns: dn_targets_boxes, dn_targets_labels, dn_masks, total_anchors
    """
    num_self_attention_heads = 8
    device = anchors.device
    batch_size, num_anchors = anchors.shape[:2]
    num_valid_matching_anchors = torch.sum(valid_anchor_mask, dim=1)
    if training and use_dn:
        targets, num_dn, box_noise_scale = dn_args
        assert targets is not None
        mask_size = num_dn + num_anchors
        dn_targets_boxes = []
        dn_targets_labels = []
        dn_masks = []
        total_anchors = []
    
        for j, t in enumerate(targets):
            if len(t['boxes']) == 0:
                # make targets
                dn_targets_boxes.append(t['boxes'].clone())
                dn_targets_labels.append(t["labels"].clone())
                # make mask
                mask = torch.ones(mask_size, mask_size).to(device) > 0
                mask[num_dn:, num_dn:] = False
                mask[:num_dn, :] = False
                if num_anchors-num_valid_matching_anchors[j] > 0:
                    mask[num_dn:, -(num_anchors-num_valid_matching_anchors[j]):] = True
                dn_masks.append(mask[None].repeat(num_self_attention_heads, 1, 1))
                # concat anchors
                padding_anchor = torch.tensor([[0.5, 0.5, 1, 1]], device=device).repeat(num_dn, 1) * imgs_whwh[j]
                total_anchor = torch.cat((padding_anchor, anchors[j]), dim=0)
                total_anchors.append(total_anchor)
            else:
                repeat_times = num_dn // len(t['boxes'])
                num_padding = num_dn % len(t['boxes'])
                # make targets
                boxes = t['boxes'][None].repeat(repeat_times, 1, 1).flatten(0, 1)
                labels = t["labels"][None].repeat(repeat_times, 1).flatten(0, 1)
                dn_targets_boxes.append(boxes.clone())
                dn_targets_labels.append(labels.clone())
                # noise on the box
                if box_noise_scale > 0:
                    diff = torch.zeros_like(boxes)
                    diff[:, :2] = boxes[:, 2:] / 2
                    diff[:, 2:] = boxes[:, 2:]
                    boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff).to(device) * box_noise_scale
                    boxes = boxes.clamp(min=0.0, max=1.0)
                boxes = boxes * imgs_whwh[j]
                # concat anchors
                padding_anchor = torch.tensor([[0.5, 0.5, 1, 1]], device=device).repeat(num_padding, 1) * imgs_whwh[j]
                total_anchor = torch.cat((boxes, padding_anchor, anchors[j]), dim=0)
                total_anchors.append(total_anchor)

                # make mask
                single_pad = len(t['boxes'])
                mask = torch.ones(mask_size, mask_size).to(device) < 0
                # match query cannot see the reconstruct
                mask[num_dn:, :num_dn] = True
                # reconstruct cannot see each other
                for i in range(repeat_times):
                    mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):num_dn] = True
                    mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                if num_padding > 0:
                    mask[num_dn-num_padding:num_dn, :num_dn-num_padding] = True
                if num_anchors-num_valid_matching_anchors[j] > 0:
                    mask[:-(num_anchors-num_valid_matching_anchors[j]), -(num_anchors-num_valid_matching_anchors[j]):] = True
                dn_masks.append(mask[None].repeat(num_self_attention_heads, 1, 1))
        dn_masks = torch.stack(dn_masks, dim=0).flatten(0, 1) # (N*num_heads, L, S)
        total_anchors = torch.stack(total_anchors, dim=0) # (N, num_dn+num_anchors, 4)
        return dn_targets_boxes, dn_targets_labels, dn_masks, total_anchors
    else:  # no dn for inference
        dn_masks = []
        for j in range(batch_size):
            mask = torch.ones(num_anchors, num_anchors).to(device) < 0
            if num_anchors-num_valid_matching_anchors[j] > 0:
                mask[:-(num_anchors-num_valid_matching_anchors[j]), -(num_anchors-num_valid_matching_anchors[j]):] = True
            dn_masks.append(mask[None].repeat(num_self_attention_heads, 1, 1))
        dn_masks = torch.stack(dn_masks, dim=0).flatten(0, 1) # (N*num_heads, L, S)
        return None, None, dn_masks, anchors


