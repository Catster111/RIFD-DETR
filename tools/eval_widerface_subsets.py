#!/usr/bin/env python3

"""Evaluate RT-DETR model on WiderFace subsets (easy/medium/hard).

This script runs COCO-style evaluation on the WiderFace val set converted to COCO,
then recomputes AP on three difficulty buckets by face size (sqrt(area) in pixels):

- hard:   [0, medium_thr)
- medium: [medium_thr, easy_thr)
- easy:   [easy_thr, +inf)

Note: WiderFace official difficulty also considers blur/occlusion; here we use a
size-based proxy because those attributes are not present in the provided COCO JSON.

Usage examples
- Default thresholds (easy>=40px, medium 20–40px, hard<20px):
    python tools/eval_widerface_subsets.py \
        -c configs/rtdetr/rtdetr_v2_face.yaml \
        --checkpoint output/rtdetr_r50vd_widerface_keypoints_v2_corrected_augmentation/last.pth

- With ResNet101 override:
    python tools/eval_widerface_subsets.py \
        -c configs/rtdetr/rtdetr_v2_face.yaml \
        --checkpoint output/your_r101_ckpt/last.pth \
        -u RTDETR.backbone=PResNet PResNet.depth=101 HybridEncoder.in_channels=[512,1024,2048]

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.core.yaml_utils import parse_cli as parse_update
from src.data.dataset.coco_utils import get_coco_api_from_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate WiderFace subsets (COCO proxy)")
    p.add_argument('-c', '--config', required=True, type=str)
    p.add_argument('--checkpoint', required=True, type=str)
    p.add_argument('-u', '--update', nargs='+', default=None,
                   help='Override YAML, e.g. PResNet.depth=101')
    p.add_argument('--batch-size', type=int, default=None, help='Override eval batch size')
    p.add_argument('--easy-thr', type=float, default=40.0, help='Easy if sqrt(area)>=easy_thr')
    p.add_argument('--med-thr', type=float, default=20.0, help='Medium if med_thr<=sqrt(area)<easy_thr')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def load_model(cfg_path: str, ckpt_path: str, update_args) -> tuple[torch.nn.Module, torch.nn.Module, YAMLConfig]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    update = parse_update(update_args) if update_args else {}
    cfg = YAMLConfig(cfg_path, **update)
    model = cfg.model
    post = cfg.postprocessor

    state = torch.load(ckpt_path, map_location='cpu')

    def _try_load(module, sd):
        try:
            module.load_state_dict(sd)
        except Exception as e:
            print(f"⚠️  Strict load failed: {e}. Retrying with strict=False…")
            missing, unexpected = module.load_state_dict(sd, strict=False)
            print(f"   missing: {len(missing)}, unexpected: {len(unexpected)}")

    _try_load(model, state.get('model', state))
    if 'postprocessor' in state and post is not None:
        _try_load(post, state['postprocessor'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    if post is not None:
        post.to(device).eval()
    return model, post, cfg


@torch.no_grad()
def run_inference(model, post, data_loader) -> List[Dict]:
    device = next(model.parameters()).device
    all_results: List[Dict] = []
    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)
        orig_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(device)
        results = post(outputs, orig_sizes)
        for tgt, out in zip(targets, results):
            image_id = int(tgt['image_id'])
            boxes = out['boxes']  # xywh in pixels
            scores = out['scores']
            labels = out['labels']
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu()
                scores = scores.detach().cpu()
                labels = labels.detach().cpu()
            for b, s, c in zip(boxes, scores, labels):
                all_results.append({
                    'image_id': image_id,
                    'category_id': int(c),
                    'bbox': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    'score': float(s),
                })
    return all_results


def evaluate_subsets(coco_gt, results: List[Dict], easy_thr: float, med_thr: float):
    # Prefer faster_coco_eval if available, else fallback to pycocotools
    try:
        from faster_coco_eval.core.coco_eval import COCOeval
    except Exception:
        from pycocotools.cocoeval import COCOeval  # type: ignore

    coco_dt = coco_gt.loadRes(results) if len(results) else None

    def _eval_area(min_side: float, max_side: float | None, label: str):
        e = COCOeval(coco_gt, coco_dt, iouType='bbox')
        # areaRng is in pixels^2
        a_min = 0 if min_side <= 0 else (min_side * min_side)
        a_max = 1e10 if max_side is None else (max_side * max_side)
        # Use label 'all' so summarize() in faster_coco_eval/pycocotools doesn't crash
        # (it expects 'all' to be present). We still print our label externally.
        e.params.areaRng = [[a_min, a_max]]
        e.params.areaRngLbl = ['all']
        e.evaluate(); e.accumulate(); e.summarize()
        # Return AP at IoU=0.50:0.95 (stats[0]) and AP50 (stats[1])
        return e.stats[0], e.stats[1]

    hard_ap, hard_ap50 = _eval_area(0.0, med_thr, 'hard')
    med_ap, med_ap50 = _eval_area(med_thr, easy_thr, 'medium')
    easy_ap, easy_ap50 = _eval_area(easy_thr, None, 'easy')

    return {
        'easy': {'AP': easy_ap, 'AP50': easy_ap50},
        'medium': {'AP': med_ap, 'AP50': med_ap50},
        'hard': {'AP': hard_ap, 'AP50': hard_ap50},
    }


def main():
    args = parse_args()
    model, post, cfg = load_model(args.config, args.checkpoint, args.update)

    # Build val dataloader (optionally override batch_size)
    val_loader = cfg.val_dataloader
    if args.batch_size is not None and args.batch_size > 0:
        # rebuild with a different batch size
        from src.core.workspace import create
        g = cfg.global_cfg
        g['val_dataloader']['total_batch_size'] = args.batch_size
        val_loader = create('val_dataloader', g, batch_size=args.batch_size)

    # Prepare COCO GT
    coco_gt = get_coco_api_from_dataset(val_loader.dataset)

    # Run inference and collect COCO-format results
    print('Running inference on validation set…')
    results = run_inference(model, post, val_loader)
    print(f'Collected {len(results)} detections.')

    # Evaluate by size-based WiderFace subsets
    print(f'Evaluating subsets with thresholds: hard < {args.med_thr}px, '
          f'medium [{args.med_thr},{args.easy_thr})px, easy >= {args.easy_thr}px')
    subset_scores = evaluate_subsets(coco_gt, results, args.easy_thr, args.med_thr)

    def fmt(d):
        return f"AP={d['AP']*100:.2f}, AP50={d['AP50']*100:.2f}"

    print('Results (size-proxy for WiderFace):')
    print(f"  Easy   : {fmt(subset_scores['easy'])}")
    print(f"  Medium : {fmt(subset_scores['medium'])}")
    print(f"  Hard   : {fmt(subset_scores['hard'])}")


if __name__ == '__main__':
    main()
