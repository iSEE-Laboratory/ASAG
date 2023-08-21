
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import copy
import json
import numpy as np
import torch
import torchvision
from PIL import Image
import util.misc as utils
import util.box_ops as box_ops
from util.recall import fast_eval_recall
from datasets.coco_eval import CocoEvaluator
from util.misc import all_gather
from crowdhumantools.crowdhuman_eval import _evaluate_predictions_on_crowdhuman

# each class has its own unique color
class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler,
                    device: torch.device, epoch: int, warmup_iter: int, box_noise_scale: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()
    if utils.is_main_process():
        print("box_noise_scale is %f" % (box_noise_scale))

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        hw = [t["size"] for t in targets]
        hw = torch.stack(hw, dim=0)
        h, w = hw.unbind(-1)
        whwh = torch.stack([w, h, w, h], dim=-1)
        whwh = whwh[:, None, :]

        hw = list(hw.cpu().numpy())
        hw = [tuple(size) for size in hw]

        outputs = model(samples, whwh, hw, targets, box_noise_scale, epoch)
        loss_dict = criterion(outputs, targets, whwh, True)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Backward
        scaler.scale(losses).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # losses.backward()

        # # accumulate loss
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # optimizer.step()
        # optimizer.zero_grad()

        if epoch == 0 and i < warmup_iter:
            lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, show=False, epoch=0, eval_recall=False, used_head='main', coco_path=''):
    color = Colors()
    model.eval()
    criterion.eval()
    gt_boxes = []
    recall_result = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    i = -1
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        i += 1
        samples = samples.to(device)
        if show:
            ori_img = [t['ori_img'] for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if k != 'ori_img'} for t in targets]
        hw = [t["size"] for t in targets]
        hw = torch.stack(hw, dim=0)
        h, w = hw.unbind(-1)
        whwh = torch.stack([w, h, w, h], dim=-1)
        whwh = whwh[:, None, :]

        hw = list(hw.cpu().numpy())
        hw = [tuple(size) for size in hw]

        outputs = model(samples, whwh, hw)
        loss_dict = criterion(outputs, targets, whwh, False)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # since batch size is 1 during inference, valid mask can be discarded
        if used_head == "main":
            results = postprocessors['bbox'](outputs, orig_target_sizes)
        else:
            results = postprocessors['bbox'](outputs['aux_outputs'][int(used_head[-1])], orig_target_sizes)
        img_h, img_w = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        # show result
        if utils.is_main_process() and show and i < 8:
            if 'aux_outputs' in outputs.keys():
                inner_result = [postprocessors['bbox'](x, orig_target_sizes) for x in outputs['aux_outputs']]
            else:
                inner_result = None
            if 'rpn' in outputs.keys():
                rpn_result = postprocessors['bbox'](outputs['rpn'], orig_target_sizes)
            else:
                rpn_result = None
            if 'anchors' in outputs.keys():
                output_anchors = outputs['anchors']['output_anchors']
                bs = len(output_anchors)
                for b in range(bs):
                    num_level = len(output_anchors[b])
                    for l in range(num_level):
                        output_anchors[b][l] = box_ops.box_cxcywh_to_xyxy(output_anchors[b][l][:, :4]) / whwh[0] * scale_fct

            for j, img_ in enumerate(ori_img):
                # anchor
                if 'anchors' in outputs.keys():
                    # all anchors
                    img = np.array(copy.deepcopy(img_))
                    img = torch.tensor(img).permute(2, 0, 1)
                    img = torchvision.utils.draw_bounding_boxes(img, torch.cat(output_anchors[j], dim=0))
                    img = img.permute(1, 2, 0).numpy()
                    img = Image.fromarray(img)
                    img.save(output_dir / ('%d_%d_%d_anchor.jpeg'%(epoch, i, j)))
                    # each level
                    num_level = len(output_anchors[b])
                    for l in range(num_level):
                        img = np.array(copy.deepcopy(img_))
                        img = torch.tensor(img).permute(2, 0, 1)
                        img = torchvision.utils.draw_bounding_boxes(img, output_anchors[j][l])
                        if l >= 2:
                            _, crop_pos, batch_idx = outputs['anchors']['anchors_for_loss'][l]
                            crop_patch = torch.cat([crop_pos, crop_pos+model.module.anchor_generator.patch_size], dim=1) * 2**(6-l) / whwh[0] * scale_fct
                            img = torchvision.utils.draw_bounding_boxes(img, crop_patch, colors='red', width=3)
                        img = img.permute(1, 2, 0).numpy()
                        img = Image.fromarray(img)
                        img.save(output_dir / ('%d_%d_%d_anchor_level_%d.jpeg'%(epoch, i, j, l)))
                # rpn
                if 'rpn' in outputs.keys():
                    img = np.array(copy.deepcopy(img_))
                    img = torch.tensor(img).permute(2, 0, 1)
                    box_color = [color(rpn_result[j]['labels'][l]) for l in range(len(rpn_result[j]['labels']))]
                    img = torchvision.utils.draw_bounding_boxes(img, rpn_result[j]['boxes'], colors=box_color)
                    img = img.permute(1, 2, 0).numpy()
                    img = Image.fromarray(img)
                    img.save(output_dir / ('%d_%d_%d_rpn.jpeg'%(epoch, i, j)))

                # inner layer
                if inner_result is not None:
                    for k, inner in enumerate(inner_result):
                        img = np.array(copy.deepcopy(img_))
                        img = torch.tensor(img).permute(2, 0, 1)
                        box_color = [color(inner[j]['labels'][l]) for l in range(len(inner[j]['labels']))]
                        img = torchvision.utils.draw_bounding_boxes(img, inner[j]['boxes'], colors=box_color)
                        img = img.permute(1, 2, 0).numpy()
                        img = Image.fromarray(img)
                        img.save(output_dir / ('%d_%d_%d_inner_%d.jpeg'%(epoch, i, j, k)))

                # final
                img = np.array(copy.deepcopy(img_))
                img = torch.tensor(img).permute(2, 0, 1)
                box_color = [color(results[j]['labels'][l]) for l in range(len(results[j]['labels']))]
                img = torchvision.utils.draw_bounding_boxes(img, results[j]['boxes'], colors=box_color)
                img = img.permute(1, 2, 0).numpy()
                img = Image.fromarray(img)
                img.save(output_dir / ('%d_%d_%d_final.jpeg'%(epoch, i, j)))
                
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if eval_recall:
            gt_boxes += [(box_ops.box_cxcywh_to_xyxy(target['boxes']) * whwh_i).cpu().numpy() for target, whwh_i in zip(targets, whwh)]
            total_anchors = outputs['anchors']['padded_output_anchors']
            total_anchors[..., :4] = box_ops.box_cxcywh_to_xyxy(total_anchors[..., :4])# * whwh
            recall_result += [an.cpu().numpy() for an in total_anchors]
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    # evalutate recall
    if eval_recall:
        fast_eval_recall(gt_boxes, recall_result, proposal_nums=[1000,])

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator




@torch.no_grad()
def evaluate_crowdhuman(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, show=False, epoch=0, eval_recall=False, used_head='main', coco_path=''):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'ori_img'} for t in targets]
        hw = [t["size"] for t in targets]
        hw = torch.stack(hw, dim=0)
        h, w = hw.unbind(-1)
        whwh = torch.stack([w, h, w, h], dim=-1)
        whwh = whwh[:, None, :]

        hw = list(hw.cpu().numpy())
        hw = [tuple(size) for size in hw]

        outputs = model(samples, whwh, hw)
        loss_dict = criterion(outputs, targets, whwh, False)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # since batch size is 1 during inference, valid mask can be discarded
        if used_head == "main":
            results = postprocessors['bbox'](outputs, orig_target_sizes)
        else:
            results = postprocessors['bbox'](outputs['aux_outputs'][int(used_head[-1])], orig_target_sizes)
                
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    gether_result = all_gather(coco_evaluator.coco_predictions)
    all_result = []
    for result in gether_result:
        all_result += result
    
    if utils.is_main_process():
        dt_path = output_dir / "results.odgt"
        gt_path = output_dir / 'annotation_val.odgt'
        
        # ########################
        #  predictions
        # ########################
        
        new_json_results = dict()
        for res in all_result:
            if res['image_id'] in new_json_results:
                new_json_results[res['image_id']]['dtboxes'].append({'box':res['bbox'], 'score':res['score']})
            else:
                new_json_results[res['image_id']] = dict()
                new_json_results[res['image_id']]['ID'] =  res['image_id']
                new_json_results[res['image_id']]['dtboxes'] = list()
                new_json_results[res['image_id']]['dtboxes'].append({'box':res['bbox'], 'score':res['score']})
                
        with open(dt_path, "w") as f:
            for db in new_json_results:
                line = json.dumps(new_json_results[db]) + '\n'
                f.write(line)
        
        # ########################
        #  gt
        # ########################
        gt_json = coco_path + 'annotations/val.json'
        json_results = json.load(open(gt_json, "r"))
        
        img_dict = dict()
        for img in json_results['images']:
            img_dict[img['id']] = {'height': img['height'], 'width': img['width']}
        
        new_json_results = dict()
        for res in json_results['annotations']:
            if res['image_id'] in new_json_results:
                new_json_results[res['image_id']]['gtboxes'].append({
                    'tag': 'person',
                    'fbox':res['bbox'], 
                    'hbox':res['bbox_vis'], # here is wrong, but no matter
                    'extra':{'ignore':res['iscrowd']}
                })
            else:
                new_json_results[res['image_id']] = dict()
                new_json_results[res['image_id']]['ID'] =  res['image_id']
                new_json_results[res['image_id']]['height'] =  img_dict[res['image_id']]['height']
                new_json_results[res['image_id']]['width'] =  img_dict[res['image_id']]['width']
                new_json_results[res['image_id']]['gtboxes'] = list()
                new_json_results[res['image_id']]['gtboxes'].append({
                    'tag': 'person',
                    'fbox':res['bbox'], 
                    'hbox':res['bbox_vis'], # here is wrong, but no matter
                    'extra':{'ignore':res['iscrowd']}
                })    
        with open(gt_path, "w") as f:
            for db in new_json_results:
                line = json.dumps(new_json_results[db]) + '\n'
                f.write(line)

        eval_results = _evaluate_predictions_on_crowdhuman(gt_path, dt_path)


        for i, metric in enumerate(["AP", "mMR", "Recall"]):
            print(metric,":",eval_results[i])

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if utils.is_main_process():
        stats["AP"] = eval_results[0]
        stats["mMR"] = eval_results[1]
        stats["Recall"] = eval_results[2]
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator



