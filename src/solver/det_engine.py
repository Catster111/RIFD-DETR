"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
import random
from typing import Iterable, Dict

import torch
import torch.amp 
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..misc.rotation_utils import (
    rotate_image_tensor,
    rotate_targets,
    inverse_rotate_boxes_cxcywh,
    rotate_keypoints,
)
from ..zoo.rtdetr.rtdetr_keypoint_head import heatmap_expectation_xy, POLAR_RADIUS_SCALE


def _compute_angle_weights(
    targets,
    cfg: Dict | None,
    device: torch.device,
):
    """Attach angle-based weights to targets and return average scaling factor."""
    if not cfg or not cfg.get('enable', False):
        return torch.tensor(1.0, device=device)

    alpha = float(cfg.get('alpha', 1.0))
    max_angle = float(max(cfg.get('max_angle', 180.0), 1e-6))
    min_weight = float(cfg.get('min_weight', 1.0))
    max_weight = float(cfg.get('max_weight', 3.0))

    weights = []
    for tgt in targets:
        boxes = tgt.get('boxes')
        angle_tensor = tgt.get('rotation_angle', torch.tensor(0.0, device=device))
        if isinstance(angle_tensor, torch.Tensor):
            angle_val = float(angle_tensor.item())
        else:
            angle_val = float(angle_tensor)

        weight = 1.0 + alpha * abs(angle_val) / max_angle
        weight = max(min_weight, min(weight, max_weight))

        if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
            tgt['angle_weight'] = torch.full(
                (boxes.shape[0],),
                weight,
                dtype=boxes.dtype,
                device=boxes.device,
            )
        else:
            tgt['angle_weight'] = torch.zeros((0,), device=device)

        tgt['angle_weight_scalar'] = torch.tensor(weight, device=device)
        weights.append(weight)

    if not weights:
        return torch.tensor(1.0, device=device)

    return torch.tensor(sum(weights) / len(weights), device=device)


def _decode_keypoints_norm(
    boxes: torch.Tensor,
    heatmaps: torch.Tensor,
    offsets: torch.Tensor,
    domain: str = 'cartesian',
):
    """Decode normalized keypoint coordinates (x, y) from network outputs."""
    bs, num_queries, num_keypoints, _, _ = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype
    decoded = torch.zeros((bs, num_queries, num_keypoints, 2), device=device, dtype=dtype)

    for b in range(bs):
        for q in range(num_queries):
            bbox = boxes[b, q]
            cx, cy, w, h = bbox
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0

            query_heatmaps = heatmaps[b, q]
            query_offsets = offsets[b, q]
            for k in range(num_keypoints):
                heatmap = query_heatmaps[k]
                base_x, base_y = heatmap_expectation_xy(heatmap, domain, POLAR_RADIUS_SCALE)
                offset = query_offsets[k]
                final_x_norm = torch.clamp(base_x + offset[0], 0, 1)
                final_y_norm = torch.clamp(base_y + offset[1], 0, 1)
                decoded[b, q, k, 0] = x1 + final_x_norm * w
                decoded[b, q, k, 1] = y1 + final_y_norm * h

    return decoded


def _maybe_compute_rotational_loss(
    model: torch.nn.Module,
    samples: torch.Tensor,
    targets,
    outputs,
    device: torch.device,
    rot_cfg: Dict | None,
    angle_cfg: Dict | None,
    scaler_enabled: bool,
    amp_device_type: str,
):
    """Run an auxiliary rotated forward pass for consistency loss."""
    if not rot_cfg or not rot_cfg.get('enable', False):
        return {}

    prob = float(rot_cfg.get('prob', 1.0))
    if random.random() > prob:
        return {}

    angles = rot_cfg.get('angles', [90])
    if not angles:
        return {}
    angle = random.choice(angles)

    rot_samples = rotate_image_tensor(samples, angle)
    rot_targets = rotate_targets(targets, angle)

    # ensure tensors are on the correct device
    rot_targets = [
        {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in tgt.items()
        }
        for tgt in rot_targets
    ]

    # Optional: attach angle weights so denoising path can see consistent metadata
    _ = _compute_angle_weights(rot_targets, angle_cfg, device)

    if scaler_enabled:
        with torch.autocast(device_type=amp_device_type, cache_enabled=True):
            rot_outputs = model(rot_samples, targets=rot_targets)
    else:
        rot_outputs = model(rot_samples, targets=rot_targets)

    losses = {}
    with torch.autocast(device_type=amp_device_type, enabled=False):
        ref_boxes = outputs['pred_boxes'].detach()
        rot_boxes = inverse_rotate_boxes_cxcywh(rot_outputs['pred_boxes'], angle)
        loss_bbox = F.smooth_l1_loss(rot_boxes, ref_boxes, reduction='mean')

        ref_logits = outputs['pred_logits'].detach()
        rot_logits = rot_outputs['pred_logits']
        loss_cls = F.mse_loss(
            torch.sigmoid(rot_logits),
            torch.sigmoid(ref_logits),
            reduction='mean',
        )

    losses['loss_rot_bbox'] = loss_bbox * float(rot_cfg.get('loss_weight_bbox', 0.5))
    losses['loss_rot_cls'] = loss_cls * float(rot_cfg.get('loss_weight_cls', 0.1))

    kpt_weight = float(rot_cfg.get('loss_weight_kpt', 0.0))
    if (
        kpt_weight > 0
        and 'pred_keypoint_heatmaps' in outputs
        and 'pred_keypoint_heatmaps' in rot_outputs
    ):
        domain = rot_cfg.get('keypoint_domain', 'cartesian')
        orig_kpts = _decode_keypoints_norm(
            outputs['pred_boxes'].detach(),
            outputs['pred_keypoint_heatmaps'].detach(),
            outputs['pred_keypoint_offsets'].detach(),
            domain,
        )
        rot_kpts = _decode_keypoints_norm(
            rot_outputs['pred_boxes'],
            rot_outputs['pred_keypoint_heatmaps'],
            rot_outputs['pred_keypoint_offsets'],
            domain,
        )

        ones_vis = torch.ones((*rot_kpts.shape[:-1], 1), device=rot_kpts.device, dtype=rot_kpts.dtype)
        rot_kpts_full = torch.cat([rot_kpts, ones_vis], dim=-1)
        rot_kpts_back_full = rotate_keypoints(rot_kpts_full, -angle)
        rot_kpts_back = rot_kpts_back_full[..., :2]

        loss_kpt = F.smooth_l1_loss(rot_kpts_back, orig_kpts, reduction='mean')
        losses['loss_rot_kpt'] = loss_kpt * kpt_weight

    return losses


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    rot_cfg = kwargs.get('rotational_consistency_cfg', None)
    angle_cfg = kwargs.get('angle_balanced_cfg', None)
    amp_device_type = device.type if isinstance(device, torch.device) else str(device)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        angle_scale = _compute_angle_weights(targets, angle_cfg, device)
        rot_losses = {}

        if scaler is not None:
            with torch.autocast(device_type=amp_device_type, cache_enabled=True):
                outputs = model(samples, targets=targets)

            rot_losses = _maybe_compute_rotational_loss(
                model,
                samples,
                targets,
                outputs,
                device,
                rot_cfg,
                angle_cfg,
                True,
                amp_device_type,
            )
            
            with torch.autocast(device_type=amp_device_type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
                if angle_cfg and angle_cfg.get('enable', False):
                    for key in ('loss_bbox', 'loss_giou', 'loss_vfl'):
                        if key in loss_dict:
                            loss_dict[key] = loss_dict[key] * angle_scale
                for k, v in rot_losses.items():
                    if k in loss_dict:
                        loss_dict[k] = loss_dict[k] + v
                    else:
                        loss_dict[k] = v

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            rot_losses = _maybe_compute_rotational_loss(
                model,
                samples,
                targets,
                outputs,
                device,
                rot_cfg,
                angle_cfg,
                False,
                amp_device_type,
            )
            loss_dict = criterion(outputs, targets, **metas)
            if angle_cfg and angle_cfg.get('enable', False):
                for key in ('loss_bbox', 'loss_giou', 'loss_vfl'):
                    if key in loss_dict:
                        loss_dict[key] = loss_dict[key] * angle_scale
            for k, v in rot_losses.items():
                if k in loss_dict:
                    loss_dict[k] = loss_dict[k] + v
                else:
                    loss_dict[k] = v
            
            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

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

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator
