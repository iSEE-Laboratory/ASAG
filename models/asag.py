
import torch
from torch import nn
import torchvision
from util import box_ops

from .dn_components import prepare_for_dn
from .backbone import build_backbone
from .anchor_generator import build_anchorgenerator
from .rpn import build_rpn
from .adamixer_decoder import build_adamixer_decoder
from .sparsercnn_decoder import build_sparsercnn_decoder
from .criterion import build_criterion
from .matcher import build_matcher
from .criterion import build_criterion



class ASAG(nn.Module):
    """ This is the roq module that performs object detection """
    def __init__(self, args, backbone, neck, anchor_generator, rpn, decoder):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.anchor_generator = anchor_generator
        self.rpn = rpn
        self.decoder = decoder
        self.num_layers = decoder.num_layers
        self.use_dn = args.use_dn
        self.num_dn = args.num_dn

        
    def forward(self, samples, imgs_whwh, hw, targets=None, box_noise_scale=0., epoch=-1):
        """ The forward expects a Tensor:
               - samples: batched images, of shape [batch_size x 3 x H x W]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        with torch.cuda.amp.autocast():
            # backbone
            features = self.backbone(samples)
            
            # neck
            srcs = {}
            for l, feat in enumerate(features):
                srcs['C%d' % (l+2)] = feat
            srcs = self.neck(srcs)
            srcs = list(srcs.values())
        
        srcs = [x.float() for x in srcs]
        # anchor generator
        anchors_for_loss, padded_output_anchors, valid_mask, max_len, output_anchors = self.anchor_generator(srcs[1:], imgs_whwh, targets, epoch)

        # dn
        dn_args = (targets, self.num_dn, box_noise_scale)
        dn_targets_boxes, dn_targets_labels, dn_masks, dn_anchors = prepare_for_dn(self.use_dn, self.training, dn_args, padded_output_anchors, valid_mask, imgs_whwh)
        
        # rpn
        # (B, num_proposals, 4), (B, num_proposals, num_class), (B, num_proposal, d_model), (B, num_proposals, 4)
        proposal, cls_score, query_content, xyzr, dn_masks, valid_mask = self.rpn(srcs, dn_anchors, imgs_whwh, hw, dn_masks, valid_mask)
        
        # decoder
        all_stage_results = self.decoder(srcs, xyzr, query_content, imgs_whwh, dn_masks) # List 'cxcywh' 0-1

        # output
        out = {'pred_logits': all_stage_results[-1]['cls_score'], 'pred_boxes': all_stage_results[-1]['decode_bbox_pred']}  # -1 means last decoder layer
        out['anchors'] = {"anchors_for_loss": anchors_for_loss, "padded_output_anchors": padded_output_anchors, "valid_mask": valid_mask, "output_anchors": output_anchors}
        out['rpn'] = {'pred_logits': cls_score, 'pred_boxes': proposal}
        out['dn'] = {'dn_targets_boxes': dn_targets_boxes, 'dn_targets_labels': dn_targets_labels, 'dn_param': (self.num_dn, max_len)}
        if self.num_layers > 1:
            out['aux_outputs'] = self._set_aux_loss(all_stage_results)
            
        return out

    @torch.jit.unused
    def _set_aux_loss(self, all_stage_results):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a['cls_score'], 'pred_boxes': a['decode_bbox_pred']}
                for a in all_stage_results[:-1]]
    


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, nms=False):
        super().__init__()
        self.nms = nms
        self.transform = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                                        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                                        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90], dtype=torch.int64)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        num_proposal = out_logits.shape[1]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = torch.sigmoid(out_logits)

        topk_values, topk_indexes = torch.topk(prob.reshape(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode='trunc')
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for i in range(len(out_logits)):
            box = boxes[i]
            score = scores[i]
            label = labels[i]
            if self.nms:
                idx = torchvision.ops.batched_nms(box, score, label, 0.7)
                results.append({'scores': score[idx], 'labels': self.transform[label[idx]].to(label.device), 'boxes': box[idx, :]})
            else:
                results.append({'scores': score, 'labels': self.transform[label].to(label.device), 'boxes': box})

        return results



class CrowdHumanPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, nms=False):
        super().__init__()
        self.nms = nms
        self.transform = torch.tensor([1, ], dtype=torch.int64)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        num_proposal = out_logits.shape[1]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = torch.sigmoid(out_logits)

        topk_values, topk_indexes = torch.topk(prob.reshape(out_logits.shape[0], -1), num_proposal, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode='trunc')
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for i in range(len(out_logits)):
            box = boxes[i]
            score = scores[i]
            label = labels[i]
            if self.nms:
                idx = torchvision.ops.batched_nms(box, score, label, 0.8)
                results.append({'scores': score[idx], 'labels': self.transform[label[idx]].to(label.device), 'boxes': box[idx, :]})
            else:
                results.append({'scores': score, 'labels': self.transform[label].to(label.device), 'boxes': box})

        return results




def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset.
    num_classes = 1 if args.dataset_file == 'crowdhuman' else 80
    featmap_strides = [4, 8, 16, 32]
    featmap_channels = [256, 512, 1024, 2048]
    device = torch.device(args.device)
    backbone = build_backbone(args)
    neck = torchvision.ops.FeaturePyramidNetwork(featmap_channels, args.hidden_dim)
    anchor_generator = build_anchorgenerator(args, featmap_strides[1:])
    rpn = build_rpn(args, num_classes, featmap_strides)
    if args.decoder_type == 'AdaMixer':
        decoder = build_adamixer_decoder(args, num_classes, featmap_strides)
    else:
        decoder = build_sparsercnn_decoder(args, num_classes)

    model = ASAG(
        args,
        backbone,
        neck,
        anchor_generator,
        rpn,
        decoder
    )
    anchor_matcher = build_matcher(args, args.anchor_matcher)
    rpn_matcher = build_matcher(args, args.rpn_matcher)
    decoder_matcher = build_matcher(args, args.decoder_matcher)
    weight_dict = {'loss_ce': args.ce_loss_coef * args.other_loss_coef, 'loss_bbox': args.bbox_loss_coef * args.other_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef * args.other_loss_coef

    # TODO this is a hack
    if args.num_decoder_layers > 1:
        aux_weight_dict = {}
        for i in range(args.num_decoder_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    weight_dict['loss_giou_rpn'] = args.giou_loss_coef * args.other_loss_coef
    weight_dict['loss_ce_rpn'] = args.ce_loss_coef * args.other_loss_coef
    weight_dict['loss_bbox_rpn'] = args.bbox_loss_coef * args.other_loss_coef
    weight_dict['loss_giou_anchor'] = 0.5 * args.anchor_loss_coef
    weight_dict['loss_bbox_anchor'] = 1 * args.anchor_loss_coef
    weight_dict['loss_ce_anchor'] = 2 * args.anchor_loss_coef
    if args.use_dn:
        for i in range(args.num_decoder_layers + 1):
            weight_dict.update({'dn_loss_ce' + f'_{i}': args.ce_loss_coef, 'dn_loss_bbox' + f'_{i}': args.bbox_loss_coef, 'dn_loss_giou' + f'_{i}': args.giou_loss_coef, })

    losses = ['labels', 'boxes']
    criterion = build_criterion(args, num_classes, anchor_matcher=anchor_matcher, rpn_matcher=rpn_matcher, 
                                decoder_matcher=decoder_matcher, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses, stride=featmap_strides)
    criterion.to(device)
    postprocessors = {'bbox': CrowdHumanPostProcess(args.nms)} if args.dataset_file == 'crowdhuman' else {'bbox': PostProcess(args.nms)}

    return model, criterion, postprocessors
