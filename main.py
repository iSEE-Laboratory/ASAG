import argparse
import datetime
import json
from models.lr_scheduler import build_lr_scheduler
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, evaluate_crowdhuman
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # training setting
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--training_schedule', default='1x', type=str, choices=('1x', '3x'))
    parser.add_argument('--warmup_iter', default=500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true') # just eval, not train
    parser.add_argument('--eval_recall', action='store_true') # just eval, not train, using only one gpu
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--draw_pic', action='store_true',
                        help='visualize results on pictures during evaluate')

    # general setting
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--activation', default='gelu', type=str, choices=('relu', 'gelu', 'silu'))
    parser.add_argument('--num_query_pattern', default=1, type=int)
    parser.add_argument('--num_query', default='100', type=str, choices=('100', '300'))
    parser.add_argument('--used_head', default='main', type=str, help="main, aux_0, aux_1, aux_2")
    parser.add_argument('--used_inference_level', default='P3P6', type=str, help="used pyramid levels in anchor generator during inference", choices=('P3P6', 'P4P6', 'P5P6'))

    # dn
    parser.add_argument('--use_dn', action='store_true',
                        help="do not use denosing training")
    parser.add_argument('--fix_noise_scale', action='store_true',
                        help="box noise scale is fixed during training")
    parser.add_argument('--num_dn', default=200, type=int,
                        help="the number of noisy queries")
    parser.add_argument('--noise_scale', default=0.4, type=float,
                        help="box noise scale in denoising training")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, choices=('resnet50', 'resnet101'),
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--pretrained_checkpoint', default='', type=str,
                        help="Path to the used pretrained checkpoint")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--return_interm_layers', action='store_false',
                        help="return c2c3c4c5")
    parser.add_argument('--no_frozen_bn', dest='frozen_bn', action='store_false',
                        help="do not frozen backbone's batch normalization")

    # anchor
    parser.add_argument('--patch_size_interpolate', default=15, type=int,
                        help="croped patch size on the first feature map")
    parser.add_argument('--patch_size', default=15, type=int,
                        help="croped patch size on the latter feature map")
    parser.add_argument('--anchor_matcher', default='AnchorHungarianMatcher', type=str, choices=('AnchorHungarianMatcher'),
                        help="label assignment method used in generating anchors, which should be class agnostic")
    parser.add_argument('--compress_dim', default=8, type=int,
                        help="compress feature map to save computation")
    parser.add_argument('--num_pred_anchor1', default=50, type=int,
                        help="the number of predicted anchors on p5")
    parser.add_argument('--num_pred_anchor2', default=20, type=int,
                        help="the number of predicted anchors on patches of other feature pyramid levels")
    parser.add_argument('--threshold1', default=0.7, type=float,
                        help="the higher conf threshold for predicted anchors that need to be refined on the next feature pyramid level")
    parser.add_argument('--threshold2', default=0.1, type=float,
                        help="the lower conf threshold for predicted anchors that need to be refined on the next feature pyramid level, \
                        the anchors with scores lower than threshold2 are regarded as noisy anchors and will be discarded.")
    parser.add_argument('--patch_nms_threshold', default=0.25, type=float, help="Select patches to save computation.")
    parser.add_argument('--use_target_epoch', default=1, type=int, help="to stablize training.")
    parser.add_argument('--use_anchor_threshold', default=0.1, type=float)
    
    # rpn
    parser.add_argument('--rpn_matcher', default='HungarianMatcher', type=str, choices=('HungarianMatcher'),
                        help="label assignment method used in rpn")

    # decoder
    parser.add_argument('--decoder_type', default='AdaMixer', type=str, choices=('AdaMixer', 'SparseRCNN'))
    parser.add_argument('--num_decoder_layers', default=4, type=int,
                        help="Number of encoder")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--in_points', default=32, type=int)
    parser.add_argument('--out_points', default=128, type=int)
    parser.add_argument('--n_groups', default=4, type=int)
    parser.add_argument('--decoder_matcher', default='HungarianMatcher', type=str, choices=('HungarianMatcher'),
                        help="label assignment method used in decoder")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    
    # post processing
    parser.add_argument('--nms', action='store_true',
                        help='use nms in the post-processing, just for expriments')

    # * Loss coefficients
    parser.add_argument('--area_weight', default=4, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--ce_loss_coef', default=2, type=float)
    parser.add_argument('--anchor_loss_coef', default=4, type=float)
    parser.add_argument('--other_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--gamma1', default=0.4, type=float, help="coefficient for cls logits in label weighting")
    parser.add_argument('--gamma2', default=0.6, type=float, help="coefficient for box IoU in label weighting")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args, output_dir):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    if args.training_schedule == '1x':
        total_epoches = 12
    else:
        total_epoches = 36
    
    if args.dataset_file == 'crowdhuman':
        total_epoches = 50
    
    if (args.num_query == '300') or (args.dataset_file == 'crowdhuman'):
        args.num_pred_anchor1 = 150
        args.num_pred_anchor2 = 50
        args.use_anchor_threshold = 0.05
        args.nms = True

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers, collate_fn=utils.collate_fn)
    data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, num_workers=args.num_workers, collate_fn=utils.collate_fn)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if args.dataset_file == 'crowdhuman':
            test_stats, coco_evaluator = evaluate_crowdhuman(model, criterion, postprocessors, data_loader_val, 
                                              base_ds, device, output_dir, args.draw_pic, eval_recall=args.eval_recall, used_head=args.used_head, coco_path=args.coco_path)
        else:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, 
                                              base_ds, device, output_dir, args.draw_pic, eval_recall=args.eval_recall, used_head=args.used_head, coco_path=args.coco_path)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            utils.save_json(coco_evaluator.coco_predictions, output_dir / "results.json") # only for single GPU evaluation
        return

    assert args.pretrained_checkpoint != '', "ImageNet pretrained checkpoint should be provided for training"
    best_mAP = 0.
    best_epoch = -1
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, total_epoches):
        if args.fix_noise_scale:
            box_noise_scale = args.noise_scale
        else:
            box_noise_scale = 0.8 - epoch * 0.04
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, lr_scheduler, device, epoch,
            args.warmup_iter, box_noise_scale, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if args.dataset_file == 'crowdhuman':
            test_stats, coco_evaluator = evaluate_crowdhuman(
                model, criterion, postprocessors, data_loader_val, base_ds, device, output_dir, args.draw_pic, epoch, used_head=args.used_head, coco_path=args.coco_path
            )
        else:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, output_dir, args.draw_pic, epoch, used_head=args.used_head, coco_path=args.coco_path
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            ##################################################
            if 'test_coco_eval_bbox' in log_stats.keys() and log_stats['test_coco_eval_bbox'][0] > best_mAP:
                best_mAP = log_stats['test_coco_eval_bbox'][0]
                best_epoch = epoch
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch,
                    'mAP': best_mAP,
                }, output_dir / 'best.pth')
            ##################################################

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("best mAP is %f at %d epoch" % (best_mAP, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        dt = datetime.datetime.now()
        output_dir = output_dir / dt.strftime('%Y%m%d_%H%M%S')
        output_dir.mkdir(parents=True, exist_ok=True)
    main(args, output_dir)


