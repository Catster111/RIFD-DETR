#!/usr/bin/env python3

"""Evaluate WiderFace val using per-face difficulty (approx. official protocol).

This expects a COCO JSON converted from the official .mat with attributes
stored per annotation: wf_blur, wf_occlusion, wf_pose, wf_invalid.

Subset rules (common in open-source repos):
  - Exclude invalid faces: wf_invalid == 0 only
  - Easy   : blur<=0 & occlusion<=0 & pose<=0
  - Medium : blur<=1 & occlusion<=1 & pose<=1
  - Hard   : blur<=2 & occlusion<=2 & pose<=2 (i.e., all valid)

Usage:
  python tools/eval_widerface_difficulty.py \
    -c configs/rtdetr/rtdetr_v2_face.yaml \
    --checkpoint output/.../last.pth \
    -u val_dataloader.dataset.ann_file='./dataset/annotations/widerface_val_coco.json' \
       val_dataloader.dataset.img_prefix='./dataset/' \
       RTDETRPostProcessor.remap_widerface_category=false
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.core.yaml_utils import parse_cli as parse_update
from src.data.dataset.coco_utils import get_coco_api_from_dataset


def parse_args():
    p = argparse.ArgumentParser('Evaluate WiderFace per-face difficulty')
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('-u', '--update', nargs='+', default=None)
    p.add_argument('--batch-size', type=int, default=None)
    # Visualization options
    p.add_argument('--viz-index', type=int, default=None, help='Visualize overlay for a single sample index')
    p.add_argument('--viz-output', type=str, default='viz_pred_gt.png', help='Output path for visualization')
    p.add_argument('--score-threshold', type=float, default=0.3, help='Score threshold for drawing predictions')
    p.add_argument('--skip-eval', action='store_true', help='Skip evaluation and only visualize one sample')
    p.add_argument('--show-top-fp', type=int, default=0, help='List top-N images with the most false positives (IoU>=0.5).')
    p.add_argument('--save-fp-list', type=str, default=None, help='Optional path to save FP ranking (csv/txt).')
    p.add_argument('--viz-top-fp', type=int, default=0, help='Automatically render overlays for top-K FP images.')
    p.add_argument('--fp-viz-dir', type=str, default='fp_viz', help='Directory to store FP overlays when --viz-top-fp > 0.')
    return p.parse_args()


def load_model(cfg_path: str, ckpt_path: str, update_args):
    update = parse_update(update_args) if update_args else {}
    cfg = YAMLConfig(cfg_path, **update)
    model = cfg.model
    post = cfg.postprocessor
    state = torch.load(ckpt_path, map_location='cpu')
    def _try_load(module, sd):
        try:
            module.load_state_dict(sd)
        except Exception:
            module.load_state_dict(sd, strict=False)
    _try_load(model, state.get('model', state))
    if 'postprocessor' in state and post is not None:
        _try_load(post, state['postprocessor'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    if post is not None:
        post.to(device).eval()
    return model, post, cfg


@torch.no_grad()
def run_inference(model, post, data_loader):
    device = next(model.parameters()).device
    all_results: List[Dict] = []
    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)
        orig_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(device)
        results = post(outputs, orig_sizes)
        for tgt, out in zip(targets, results):
            image_id = int(tgt['image_id'])
            boxes = out['boxes']
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


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _bbox_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def compute_fp_ranking(
    coco_gt,
    results: List[Dict],
    top_k: int = 20,
    iou_thr: float = 0.5,
    score_thr: float = 0.0,
    preds_by_image: Optional[Dict[int, List[Dict]]] = None,
) -> List[Tuple[int, int, int]]:
    """Return list of (image_id, fp_count, total_preds)."""
    gt_boxes = defaultdict(list)
    for ann in coco_gt.dataset.get('annotations', []):
        if ann.get('wf_invalid', 0) != 0:
            continue
        gt_boxes[ann['image_id']].append(_xywh_to_xyxy(ann['bbox']))

    if preds_by_image is None:
        preds_dict = defaultdict(list)
        for det in results:
            if det['score'] < score_thr:
                continue
            preds_dict[det['image_id']].append(det)
    else:
        preds_dict = defaultdict(list)
        for img_id, dets in preds_by_image.items():
            filtered = [d for d in dets if d['score'] >= score_thr]
            if filtered:
                preds_dict[img_id].extend(filtered)

    ranking = []
    for img_id, dets in preds_dict.items():
        pred_list = sorted(dets, key=lambda x: x['score'], reverse=True)
        gts = gt_boxes.get(img_id, [])
        matched = [False] * len(gts)
        fp = 0
        total_preds = len(pred_list)
        for det in pred_list:
            box = _xywh_to_xyxy(det['bbox'])
            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(gts):
                if matched[idx]:
                    continue
                iou = _bbox_iou_xyxy(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_thr and best_idx >= 0:
                matched[best_idx] = True
            else:
                fp += 1
        ranking.append((img_id, fp, total_preds))

    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking[:top_k]


@torch.no_grad()
def visualize_one(
    model,
    post,
    dataset,
    coco_gt,
    index: Optional[int],
    out_path: str,
    score_thr: float = 0.3,
    *,
    image_id: Optional[int] = None,
    precomputed: Optional[List[Dict]] = None,
    fp_only: bool = False,
    iou_thr: float = 0.5,
):
    """Visualize prediction vs GT on one sample.

    Provide either dataset index or image_id.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np
    from torchvision import transforms

    # Resolve image_id and file path
    if image_id is None:
        if index is None:
            raise ValueError('Either index or image_id must be provided for visualization')
        if hasattr(dataset, 'image_ids'):
            image_id = int(dataset.image_ids[index])
        else:
            image_id = coco_gt.getImgIds()[index]

    img_info_list = coco_gt.loadImgs([image_id]) or [{}]
    img_info = img_info_list[0] if isinstance(img_info_list, list) else img_info_list
    file_name = img_info.get('file_name')
    # Fallback to dataset's image_dict if COCO API lacks file_name
    if not file_name and hasattr(dataset, 'image_dict') and image_id in getattr(dataset, 'image_dict'):
        file_name = dataset.image_dict[image_id].get('file_name')
    if not file_name:
        raise KeyError('file_name')

    img_prefix = getattr(dataset, 'img_prefix', '')
    img_path = os.path.join(img_prefix, file_name) if img_prefix else file_name

    # Load original image
    image = Image.open(img_path).convert('RGB')
    width, height = image.size

    device = next(model.parameters()).device

    detections: List[Dict]
    if precomputed is None:
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        tensor = transform(image).unsqueeze(0).to(device)
        orig_sizes = torch.tensor([[height, width]], dtype=torch.float32, device=device)
        outputs = model(tensor)
        det = post(outputs, orig_sizes)[0]
        boxes = det.get('boxes', torch.empty(0))
        scores = det.get('scores', torch.empty(0))
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu()
        detections = [
            {
                'bbox': [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])],
                'score': float(scores[i]),
            }
            for i in range(len(boxes))
        ]
    else:
        detections = [
            {
                'bbox': [float(v) for v in det['bbox']],
                'score': float(det['score']),
            }
            for det in precomputed
        ]

    # Prepare figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(np.array(image))
    ax.axis('off')

    # Draw GT boxes (green)
    ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
    anns = coco_gt.loadAnns(ann_ids)
    for a in anns:
        x, y, w, h = a['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    # Draw predictions (red)
    filtered = [det for det in detections if det['score'] >= score_thr]
    gt_boxes_xyxy = [_xywh_to_xyxy(a['bbox']) for a in anns]
    fp_flags = []
    pred_xyxy = []
    matched = [False] * len(gt_boxes_xyxy)
    for det in filtered:
        pred_xyxy.append(_xywh_to_xyxy(det['bbox']))

    for box in pred_xyxy:
        best_iou = 0.0
        best_idx = -1
        for idx, gt_box in enumerate(gt_boxes_xyxy):
            if matched[idx]:
                continue
            iou = _bbox_iou_xyxy(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_thr and best_idx >= 0:
            matched[best_idx] = True
            fp_flags.append(False)
        else:
            fp_flags.append(True)

    if fp_only and not any(fp_flags):
        print(f'ℹ️  No false positives above threshold {score_thr} for image {image_id}.')

    for det, is_fp in zip(filtered, fp_flags):
        if fp_only and not is_fp:
            continue
        x, y, w, h = det['bbox']
        color = 'red' if is_fp else 'cyan'
        rect = patches.Rectangle((x, y), float(w), float(h), linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, max(y - 4, 0), f"{det['score']:.2f}", color=color, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    title_suffix = 'FP only' if fp_only else 'Predictions'
    ax.set_title(f'Image ID {image_id} — GT (green) vs {title_suffix} (thr={score_thr})')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'✅ Saved visualization: {os.path.abspath(out_path)}')


def coco_from_dataset_dict(dataset_dict: Dict):
    try:
        from faster_coco_eval import COCO
    except Exception:
        from pycocotools.coco import COCO  # type: ignore
    coco = COCO()
    coco.dataset = dataset_dict
    coco.createIndex()
    return coco


def filter_by_difficulty(coco_gt, subset: str) -> 'COCO':
    assert subset in {'easy', 'medium', 'hard'}
    ds = copy.deepcopy(coco_gt.dataset)
    anns = ds['annotations']
    filtered = []
    for a in anns:
        # Require our attributes
        inv = a.get('wf_invalid', 0)
        if inv != 0:
            continue
        blur = a.get('wf_blur', 0)
        occ = a.get('wf_occlusion', 0)
        pose = a.get('wf_pose', 0)
        if subset == 'easy':
            keep = (blur <= 0 and occ <= 0 and pose <= 0)
        elif subset == 'medium':
            keep = (blur <= 1 and occ <= 1 and pose <= 1)
        else:
            keep = (blur <= 2 and occ <= 2 and pose <= 2)
        if keep:
            filtered.append(a)
    ds['annotations'] = filtered
    return coco_from_dataset_dict(ds)


def evaluate_subset(coco_gt, results, subset_label: str):
    # Prefer faster_coco_eval COCOeval
    try:
        from faster_coco_eval.core.coco_eval import COCOeval
    except Exception:
        from pycocotools.cocoeval import COCOeval  # type: ignore

    sub_gt = filter_by_difficulty(coco_gt, subset_label)
    coco_dt = sub_gt.loadRes(results) if len(results) else None
    e = COCOeval(sub_gt, coco_dt, iouType='bbox')

    # Restrict evaluation to images that actually have GT for this subset
    img_ids_with_gt = sorted({ann['image_id'] for ann in sub_gt.dataset.get('annotations', [])})
    if img_ids_with_gt:
        e.params.imgIds = img_ids_with_gt

    # Log basic counts for transparency
    print(f"Subset '{subset_label}': images={len(img_ids_with_gt)}, gt={len(sub_gt.dataset.get('annotations', []))}")

    e.evaluate(); e.accumulate(); e.summarize()
    return e.stats[0], e.stats[1]


def main():
    args = parse_args()
    model, post, cfg = load_model(args.config, args.checkpoint, args.update)
    val_loader = cfg.val_dataloader
    if args.batch_size:
        from src.core.workspace import create
        g = cfg.global_cfg
        g['val_dataloader']['total_batch_size'] = args.batch_size
        val_loader = create('val_dataloader', g, batch_size=args.batch_size)

    dataset = val_loader.dataset
    if hasattr(dataset, 'coco'):
        coco_dict = copy.deepcopy(dataset.coco)
        coco_gt = coco_from_dataset_dict(coco_dict)
    else:
        coco_gt = get_coco_api_from_dataset(dataset)
    # Fast path: only visualize a single sample
    # argparse converts '--skip-eval' to attribute 'skip_eval'
    if args.viz_index is not None and args.skip_eval:
        try:
            visualize_one(
                model,
                post,
                val_loader.dataset,
                coco_gt,
                args.viz_index,
                args.viz_output,
                args.score_threshold,
            )
            return
        except Exception as e:
            print(f'⚠️  Visualization failed: {e}')

    print('Running inference on validation set…')
    results = run_inference(model, post, val_loader)
    print(f'Collected {len(results)} detections.')
    preds_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for det in results:
        preds_by_image[det['image_id']].append(det)

    ranking_info: List[Tuple[int, str, int, int]] = []
    need_fp_ranking = (
        args.show_top_fp > 0
        or args.viz_top_fp > 0
        or (args.save_fp_list is not None)
    )
    if need_fp_ranking:
        top_k = max(args.show_top_fp, args.viz_top_fp, 0)
        if args.save_fp_list and top_k == 0:
            top_k = 50
        if top_k > 0:
            ranking = compute_fp_ranking(
                coco_gt,
                results,
                top_k=top_k,
                iou_thr=0.5,
                score_thr=args.score_threshold,
                preds_by_image=preds_by_image,
            )
        else:
            ranking = []

        if ranking:
            id_to_name = {img['id']: img.get('file_name', '') for img in coco_gt.dataset.get('images', [])}
            for idx, (img_id, fp_count, total_preds) in enumerate(ranking, 1):
                name = id_to_name.get(img_id, f'image_{img_id}')
                ranking_info.append((img_id, name, fp_count, total_preds))

            if args.show_top_fp > 0:
                display = ranking_info[:args.show_top_fp]
                print(f'Top {len(display)} images with most predicted false positives (score >= {args.score_threshold}):')
                for rank_idx, (img_id, name, fp_count, total_preds) in enumerate(display, 1):
                    print(f'  {rank_idx:02d}. {name} — FP {fp_count}/{total_preds}')
        else:
            print('No predictions above score threshold; skipping FP ranking.')

        if args.save_fp_list and ranking_info:
            os.makedirs(os.path.dirname(args.save_fp_list) or '.', exist_ok=True)
            with open(args.save_fp_list, 'w') as f:
                f.write('rank,image_id,file_name,false_positives,total_predictions\n')
                for idx, (img_id, name, fp_count, total_preds) in enumerate(ranking_info, 1):
                    f.write(f'{idx},{img_id},{name},{fp_count},{total_preds}\n')
            print(f'✅ Saved FP ranking to {os.path.abspath(args.save_fp_list)}')

        if args.viz_top_fp > 0 and ranking_info:
            os.makedirs(args.fp_viz_dir, exist_ok=True)
            if hasattr(dataset, 'image_ids'):
                id_to_index = {int(img_id): idx for idx, img_id in enumerate(dataset.image_ids)}
            else:
                id_to_index = {img['id']: idx for idx, img in enumerate(coco_gt.dataset.get('images', []))}

            for rank_idx, (img_id, name, fp_count, total_preds) in enumerate(ranking_info[:args.viz_top_fp], 1):
                idx = id_to_index.get(img_id)
                out_name = os.path.join(args.fp_viz_dir, f'fp_{rank_idx:02d}_{os.path.basename(name)}')
                try:
                    visualize_one(
                        model,
                        post,
                        dataset,
                        coco_gt,
                        idx,
                        out_name,
                        args.score_threshold,
                        image_id=img_id,
                        precomputed=preds_by_image.get(img_id, []),
                        fp_only=True,
                        iou_thr=0.5,
                    )
                except Exception as e:
                    print(f'⚠️  Failed to visualize {name}: {e}')

    for subset in ['easy', 'medium', 'hard']:
        print(f'===== {subset} subset (difficulty-aware) =====')
        ap, ap50 = evaluate_subset(coco_gt, results, subset)
        print(f'AP={ap*100:.2f}, AP50={ap50*100:.2f}')

    # Optional single-image visualization
    if args.viz_index is not None:
        try:
            visualize_one(model, post, val_loader.dataset, coco_gt, args.viz_index, args.viz_output, args.score_threshold)
        except Exception as e:
            print(f'⚠️  Visualization failed: {e}')


if __name__ == '__main__':
    main()
