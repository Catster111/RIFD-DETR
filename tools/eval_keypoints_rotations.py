#!/usr/bin/env python3
"""Evaluate keypoint accuracy under image rotations (AFLW / COCO-style)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms

# make repo root importable when script is run from rtdetrv2_pytorch/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig

# allow slightly truncated JPEGs (AFLW has some)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Keypoint rotation evaluation")
    parser.add_argument('-c', '--config', required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint (.pth)')
    parser.add_argument('--coco-json', required=True, help='COCO-style annotation json (with keypoints)')
    parser.add_argument('--image-root', required=True, help='Root directory of original images')
    parser.add_argument('--rotations', nargs='+', default=['right', 'left', 'up', 'down'],
                        help='Subset of {right,left,up,down}')
    parser.add_argument('--category-id', type=int, default=None,
                        help='Optional category_id filter (e.g., 1 for face)')
    parser.add_argument('--score-thr', type=float, default=0.1,
                        help='Score threshold for predictions')
    parser.add_argument('--iou-thr', type=float, default=0.5,
                        help='IoU threshold to match predictions with GT boxes')
    parser.add_argument('--pck-thr', type=float, default=0.1,
                        help='PCK threshold on normalized error (<= threshold)')
    parser.add_argument('--norm', type=str, default='sqrtwh', choices=['sqrtwh', 'iod'],
                        help='Normalization method: sqrtwh (√(w·h)) or iod (inter-ocular distance)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Limit number of images per orientation (debug)')
    parser.add_argument('--keypoint-names', type=str, nargs='+',
                        default=['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'],
                        help='Names of keypoints for per-keypoint stats (default matches AFLW)')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Randomly sample this many images (with GT) and reuse for all rotations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for image sampling')
    parser.add_argument('--log-high-error', type=str, default=None,
                        help='Optional CSV file to log keypoints with normalized error >= error-thr')
    parser.add_argument('--error-thr', type=float, default=0.5,
                        help='Threshold for logging high-error keypoints')
    parser.add_argument('--exclude-list', type=str, default=None,
                        help='Path to CSV/TXT with image file_name to exclude from evaluation')
    return parser.parse_args()


def load_model(cfg_path: str, checkpoint: str, device: torch.device):
    cfg = YAMLConfig(cfg_path, resume=checkpoint)
    state = torch.load(checkpoint, map_location='cpu')
    weights = state.get('ema', {}).get('module') if 'ema' in state else state.get('model', state)
    model = cfg.model.to(device)
    model.load_state_dict(weights)
    model.eval()
    post = cfg.postprocessor
    post.to(device)
    post.eval()
    return model, post, cfg


def preprocess_image(img: Image.Image, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)


def run_inference(model, postprocessor, img_tensor, orig_size, device):
    with torch.no_grad():
        outputs = model(img_tensor)
        orig_sizes = torch.tensor([orig_size], dtype=torch.float32, device=device)
        processed = postprocessor(outputs, orig_sizes)
    return processed[0]


def iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    if inter <= 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def rotate_points(points, width: float, height: float, angle_deg: int):
    angle_rad = math.radians(angle_deg % 360)
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)
    cx, cy = width / 2.0, height / 2.0
    if angle_deg % 180 == 0:
        new_cx, new_cy = cx, cy
    else:
        new_cx, new_cy = height / 2.0, width / 2.0

    rotated = []
    for x, y in points:
        x0 = x - cx
        y0 = y - cy
        xr = x0 * cos_a + y0 * sin_a
        yr = -x0 * sin_a + y0 * cos_a
        rotated.append((xr + new_cx, yr + new_cy))
    return rotated


def rotate_box(box, width: float, height: float, angle_deg: int):
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    rotated = rotate_points(corners, width, height, angle_deg)
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    return [min(xs), min(ys), max(xs), max(ys)]


def clamp_box(box, width: float, height: float):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(width, x1))
    y1 = max(0.0, min(height, y1))
    x2 = max(0.0, min(width, x2))
    y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def rotate_keypoints(kpts, width: float, height: float, angle_deg: int):
    pts = []
    coords = [(kpts[i], kpts[i + 1], kpts[i + 2]) for i in range(0, len(kpts), 3)]
    xy = [(x, y) for x, y, v in coords]
    rotated_xy = rotate_points(xy, width, height, angle_deg)
    for (x, y, v), (rx, ry) in zip(coords, rotated_xy):
        pts.extend([rx, ry, v])
    return pts


def open_image_safe(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert('RGB')
    except Exception as e:
        print(f"⚠️  Skipping image {path}: {e}")
        return None


def load_coco_with_keypoints(json_path: str, category_id: int | None):
    with open(json_path, 'r') as f:
        data = json.load(f)
    id_to_image = {img['id']: img for img in data['images']}
    per_image = defaultdict(list)
    for ann in data['annotations']:
        if category_id is not None and ann.get('category_id') != category_id:
            continue
        if ann.get('iscrowd', 0):
            continue
        if 'keypoints' not in ann or not ann['keypoints']:
            continue
        img_info = id_to_image.get(ann['image_id'])
        if not img_info:
            continue
        box_xywh = ann['bbox']
        box_xyxy = [box_xywh[0], box_xywh[1],
                    box_xywh[0] + box_xywh[2], box_xywh[1] + box_xywh[3]]
        per_image[ann['image_id']].append({
            'bbox': box_xyxy,
            'keypoints': ann['keypoints'],
        })
    return data['images'], per_image


def boxes_xywh_to_xyxy_torch(boxes_xywh: torch.Tensor) -> torch.Tensor:
    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    x2 = x1 + boxes_xywh[:, 2]
    y2 = y1 + boxes_xywh[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, post, cfg = load_model(args.config, args.checkpoint, device)
    images, anns_per_image = load_coco_with_keypoints(args.coco_json, args.category_id)
    print(f"Loaded {len(images)} images from {args.coco_json}; {len(anns_per_image)} have keypoints")

    name_to_id = {img['file_name']: img['id'] for img in images}

    exclude_ids: set[int] = set()
    if args.exclude_list:
        exclude_path = Path(args.exclude_list)
        if not exclude_path.is_file():
            print(f"⚠️  Exclude list not found: {exclude_path}")
        else:
            print(f"Reading exclude list from {exclude_path}")
            if exclude_path.suffix.lower() == '.csv':
                with exclude_path.open('r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not row:
                            continue
                        name = row.get('image') or row.get('file_name')
                        if not name and len(row) > 0:
                            name = list(row.values())[0]
                        if not name:
                            continue
                        name = name.strip()
                        img_id = name_to_id.get(name)
                        if img_id is not None:
                            exclude_ids.add(img_id)
            else:
                with exclude_path.open('r') as f:
                    for line in f:
                        name = line.strip()
                        if not name or name.startswith('#'):
                            continue
                        img_id = name_to_id.get(name)
                        if img_id is not None:
                            exclude_ids.add(img_id)
            if exclude_ids:
                print(f"Excluding {len(exclude_ids)} images based on exclude list")

    # Build list of images that have keypoints and are not excluded
    valid_images = [img for img in images if img['id'] in anns_per_image and img['id'] not in exclude_ids]
    selected_ids = None
    if args.num_images is not None:
        rng = random.Random(args.seed)
        valid_ids = [img['id'] for img in valid_images]
        if args.num_images < len(valid_ids):
            sampled_ids = rng.sample(valid_ids, args.num_images)
        else:
            sampled_ids = valid_ids
        selected_ids = set(sampled_ids)
        print(f"Sampling {len(sampled_ids)} images (seed={args.seed})")
    else:
        print(f"Evaluating all {len(valid_images)} images with keypoints")

    high_error_entries = []

    if selected_ids is None:
        target_images = valid_images
    else:
        target_images = [img for img in valid_images if img['id'] in selected_ids]

    orientation_map = {
        'right': 0,
        'left': 180,
        'up': 90,
        'down': 270,
    }

    rotations = [o.lower() for o in args.rotations]
    kp_names = args.keypoint_names
    results = {}

    for orientation in rotations:
        if orientation not in orientation_map:
            print(f"⚠️  Skip invalid orientation: {orientation}")
            continue
        angle = orientation_map[orientation]
        total_kpts = 0
        matched_kpts = 0
        sum_norm_error = 0.0
        pck_hits = 0
        matched_boxes = 0
        total_boxes = 0

        per_kp_sum = np.zeros(len(kp_names), dtype=np.float64)
        per_kp_cnt = np.zeros(len(kp_names), dtype=np.int64)
        per_kp_pck = np.zeros(len(kp_names), dtype=np.int64)

        print(f"\n=== Orientation: {orientation.upper()} (angle={angle}°) ===")

        for img_idx, img_info in enumerate(target_images):
            if args.max_images is not None and img_idx >= args.max_images:
                break
            ann_list = anns_per_image.get(img_info['id'])
            if not ann_list:
                continue

            img_path = Path(args.image_root) / img_info['file_name']
            if not img_path.is_file():
                print(f"⚠️  Missing image: {img_path}")
                continue

            image = open_image_safe(img_path)
            if image is None:
                continue

            width, height = image.size
            rot_image = image
            if angle == 180:
                rot_image = image.transpose(Image.ROTATE_180)
            elif angle == 90:
                rot_image = image.transpose(Image.ROTATE_90)
            elif angle == 270:
                rot_image = image.transpose(Image.ROTATE_270)

            new_w, new_h = rot_image.size

            # Rotate GT boxes and keypoints
            gt_boxes = []
            gt_kpts = []
            for ann in ann_list:
                rot_box = rotate_box(ann['bbox'], width, height, angle)
                rot_box = clamp_box(rot_box, new_w, new_h)
                if rot_box is None:
                    continue
                rot_kp = rotate_keypoints(ann['keypoints'], width, height, angle)
                gt_boxes.append(rot_box)
                gt_kpts.append(rot_kp)

            if not gt_boxes:
                continue

            total_boxes += len(gt_boxes)

            img_tensor = preprocess_image(rot_image, device)
            prediction = run_inference(model, post, img_tensor, (new_h, new_w), device)

            pred_boxes_xywh = prediction.get('boxes', torch.empty(0, 4))
            pred_scores = prediction.get('scores', torch.empty(0))
            pred_kpts = prediction.get('keypoints', None)

            if pred_boxes_xywh.numel() == 0 or pred_kpts is None:
                continue

            pred_boxes_xyxy = boxes_xywh_to_xyxy_torch(pred_boxes_xywh)
            scores = pred_scores.detach().cpu().numpy()
            boxes_pred = pred_boxes_xyxy.detach().cpu().numpy()
            kpts_pred = [kp.cpu().numpy() for kp in pred_kpts]

            # filter by score
            keep = scores >= args.score_thr
            boxes_pred = boxes_pred[keep]
            kpts_pred = [k for k, m in zip(kpts_pred, keep) if m]
            scores_filtered = scores[keep]

            if not len(boxes_pred):
                continue

            # Greedy matching
            assigned_pred = set()
            for gt_box, gt_kp in zip(gt_boxes, gt_kpts):
                gt_box_arr = np.array(gt_box)
                best_iou = 0.0
                best_idx = -1
                for idx_pred, box_pred in enumerate(boxes_pred):
                    if idx_pred in assigned_pred:
                        continue
                    iou = iou_xyxy(gt_box_arr, box_pred)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx_pred
                if best_idx >= 0 and best_iou >= args.iou_thr:
                    assigned_pred.add(best_idx)
                    matched_boxes += 1

                    kp_pred = kpts_pred[best_idx]
                    kp_gt = np.array(gt_kp).reshape(-1, 3)
                    kp_pred = np.array(kp_pred).reshape(-1, 3)
                    score_val = float(scores_filtered[best_idx])
                    iou_value = float(best_iou)

                    if args.norm == 'iod' and kp_gt.shape[0] >= 2:
                        lx, ly = kp_gt[0, :2]
                        rx, ry = kp_gt[1, :2]
                        norm = math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)
                        norm = max(norm, 1.0)
                    else:
                        box_width = max(1.0, gt_box_arr[2] - gt_box_arr[0])
                        box_height = max(1.0, gt_box_arr[3] - gt_box_arr[1])
                        norm = math.sqrt(box_width * box_height)

                    for idx_kp, (p_gt, p_pred) in enumerate(zip(kp_gt, kp_pred)):
                        if p_gt[2] <= 0:
                            continue
                        total_kpts += 1
                        dx = p_pred[0] - p_gt[0]
                        dy = p_pred[1] - p_gt[1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        norm_err = dist / norm
                        sum_norm_error += norm_err
                        if norm_err <= args.pck_thr:
                            pck_hits += 1
                        matched_kpts += 1

                        if idx_kp < len(kp_names):
                            per_kp_sum[idx_kp] += norm_err
                            per_kp_cnt[idx_kp] += 1
                            if norm_err <= args.pck_thr:
                                per_kp_pck[idx_kp] += 1
                        if norm_err >= args.error_thr:
                            kp_name = kp_names[idx_kp] if idx_kp < len(kp_names) else f'kp_{idx_kp}'
                            high_error_entries.append({
                                'orientation': orientation,
                                'angle': angle,
                                'image': img_info['file_name'],
                                'keypoint': kp_name,
                                'norm_err': norm_err,
                                'score': score_val,
                                'iou': iou_value
                            })

        if matched_kpts == 0:
            print("No matched keypoints for", orientation)
            results[orientation] = {
                'matched_boxes': matched_boxes,
                'total_boxes': total_boxes,
                'matched_keypoints': matched_kpts,
                'total_keypoints': total_kpts,
                'nme': float('nan'),
                'pck': 0.0,
            }
            continue

        avg_nme = sum_norm_error / matched_kpts
        pck = pck_hits / matched_kpts
        print(f"Matched boxes: {matched_boxes}/{total_boxes}")
        print(f"Matched keypoints: {matched_kpts}/{total_kpts}")
        print(f"Mean normalized error (NME): {avg_nme:.4f}")
        print(f"PCK@{args.pck_thr:.2f}: {pck * 100:.2f}%")

        for idx_kp, name in enumerate(kp_names):
            cnt = per_kp_cnt[idx_kp]
            if cnt == 0:
                print(f"  - {name}: no valid points")
                continue
            nme_kp = per_kp_sum[idx_kp] / cnt
            pck_kp = per_kp_pck[idx_kp] / cnt
            print(f"  - {name}: NME={nme_kp:.4f}, PCK@{args.pck_thr:.2f}={pck_kp * 100:.2f}% (count={cnt})")

        results[orientation] = {
            'matched_boxes': matched_boxes,
            'total_boxes': total_boxes,
            'matched_keypoints': matched_kpts,
            'total_keypoints': total_kpts,
            'nme': avg_nme,
            'pck': pck,
            'per_kp_cnt': per_kp_cnt.tolist(),
            'per_kp_sum': per_kp_sum.tolist(),
            'per_kp_pck': per_kp_pck.tolist(),
        }

    if results:
        print("\n=== Summary ===")
        for orientation, stats in results.items():
            nme = stats['nme']
            pck = stats['pck']
            if math.isnan(nme):
                print(f"  {orientation.capitalize():>5}: no matches")
            else:
                print(f"  {orientation.capitalize():>5}: NME={nme:.4f}  PCK@{args.pck_thr:.2f}={pck*100:.2f}%")

    if args.log_high_error:
        log_path = Path(args.log_high_error)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if high_error_entries:
            fieldnames = ['orientation', 'angle', 'image', 'keypoint', 'norm_err', 'score', 'iou']
            high_error_entries.sort(key=lambda x: x['norm_err'], reverse=True)
            with log_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in high_error_entries:
                    writer.writerow(row)
            print(f"Logged {len(high_error_entries)} keypoints with norm_err >= {args.error_thr} to {log_path}")
        else:
            if log_path.exists():
                log_path.unlink()
            print(f"No keypoints exceeded error threshold {args.error_thr}; nothing logged")


if __name__ == '__main__':
    main()
