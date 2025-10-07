#!/usr/bin/env python3

"""Visualise rotational consistency by comparing detections across rotations."""

from __future__ import annotations

import argparse
import colorsys
import os
import sys
from pathlib import Path
import math
from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

# Ensure package imports work when executed from repository root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.misc.rotation_utils import inverse_rotate_boxes_cxcywh


def rotate_point_xy(x: float, y: float, width: float, height: float, angle_deg: float) -> tuple[float, float]:
    """Rotate absolute pixel coords (x,y) around image centre by angle_deg.

    Positive angles follow PIL/torch.rot90: counter-clockwise, y-axis down.
    """
    cx, cy = width / 2.0, height / 2.0
    a = math.radians(angle_deg % 360)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    px, py = x - cx, y - cy
    xr = px * cos_a + py * sin_a
    yr = -px * sin_a + py * cos_a
    return xr + cx, yr + cy


def build_palette(n: int) -> List[tuple[int, int, int]]:
    base_hues = [(i / max(n, 1)) for i in range(n)]
    return [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, 0.7, 0.9)) for h in base_hues]


def load_model(cfg_path: str, checkpoint: str, device: torch.device):
    cfg = YAMLConfig(cfg_path, resume=checkpoint)

    state = torch.load(checkpoint, map_location='cpu')
    if 'ema' in state:
        weights = state['ema']['module']
    else:
        weights = state['model']

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


def boxes_xywh_to_cxcywh(boxes_xywh: torch.Tensor, width: float, height: float) -> torch.Tensor:
    cx = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0) / width
    cy = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0) / height
    w = boxes_xywh[:, 2] / width
    h = boxes_xywh[:, 3] / height
    return torch.stack([cx, cy, w, h], dim=-1)


def boxes_xywh_to_xyxy(boxes_xywh: torch.Tensor) -> torch.Tensor:
    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    x2 = x1 + boxes_xywh[:, 2]
    y2 = y1 + boxes_xywh[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def boxes_cxcywh_to_xyxy(boxes_cxcywh: torch.Tensor, width: float, height: float) -> torch.Tensor:
    cx = boxes_cxcywh[:, 0] * width
    cy = boxes_cxcywh[:, 1] * height
    w = boxes_cxcywh[:, 2] * width
    h = boxes_cxcywh[:, 3] * height
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def draw_boxes(image: Image.Image, boxes: torch.Tensor, scores: torch.Tensor, label: str, color: tuple[int, int, int], score_thr: float = 0.4):
    draw = ImageDraw.Draw(image)
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for box, score in zip(boxes, scores):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{label}: {score:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)
        text_w = text_size[2] - text_size[0]
        text_h = text_size[3] - text_size[1]
        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
        draw.text((x1, y1 - text_h), text, fill=(0, 0, 0), font=font)


def draw_keypoints(image: Image.Image, keypoints, color: tuple[int, int, int], radius: int = 3, score_thr: float = 0.0):
    """Draw keypoints list returned by postprocessor (list of [K,3] tensors)."""
    if keypoints is None:
        return
    draw = ImageDraw.Draw(image)
    for kp in keypoints:
        if kp is None:
            continue
        # kp: [K, 3] with (x, y, conf)
        for k in range(kp.shape[0]):
            x, y = float(kp[k, 0]), float(kp[k, 1])
            conf = float(kp[k, 2]) if kp.shape[1] > 2 else 1.0
            if conf < score_thr:
                continue
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)


def main():
    parser = argparse.ArgumentParser("Visualise rotational consistency outputs")
    parser.add_argument('-c', '--config', required=True, help='Path to YAML config')
    parser.add_argument('-r', '--resume', required=True, help='Checkpoint path (.pth)')
    parser.add_argument('-i', '--image', required=True, help='Input image path')
    parser.add_argument('--angles', nargs='+', type=float, default=[90, 180, 270], help='Angles to evaluate')
    parser.add_argument('--score-thr', type=float, default=0.4, help='Score threshold for drawing')
    parser.add_argument('--output-dir', type=str, default='rot_consistency_vis', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, post, _ = load_model(args.config, args.resume, device)

    img_path = Path(args.image)
    image = Image.open(img_path).convert('RGB')
    width, height = image.size

    img_tensor = preprocess_image(image, device)
    orig_prediction = run_inference(model, post, img_tensor, (height, width), device)

    orig_boxes_xywh = orig_prediction['boxes']
    orig_scores = orig_prediction['scores']
    orig_kps = orig_prediction.get('keypoints', None)
    orig_boxes_xyxy = boxes_cxcywh_to_xyxy(
        boxes_xywh_to_cxcywh(orig_boxes_xywh, width, height),
        width,
        height,
    )

    palette = build_palette(len(args.angles) + 1)
    base_color = palette[0]

    # Prepare canvas that accumulates all boxes on the original image
    base_overlay = image.copy()
    draw_boxes(base_overlay, orig_boxes_xyxy, orig_scores, 'orig', base_color, args.score_thr)
    if orig_kps is not None:
        draw_keypoints(base_overlay, orig_kps, base_color)

    combined_inverse = base_overlay.copy()

    outputs: List[tuple[Image.Image, str]] = []
    outputs.append((base_overlay, 'original.png'))
    inverse_overlays: List[Image.Image] = [base_overlay]
    rotated_raw_views: List[Image.Image] = []

    for idx, angle in enumerate(args.angles, start=1):
        angle_color = palette[idx]
        k = int(angle // 90) % 4
        rot_tensor = torch.rot90(img_tensor, k, dims=(-2, -1))
        rot_prediction = run_inference(model, post, rot_tensor, (height, width), device)

        rot_boxes_xywh = rot_prediction['boxes']
        rot_scores = rot_prediction['scores']
        rot_kps = rot_prediction.get('keypoints', None)

        # Visualise predictions in the rotated frame
        rot_img_pil = transforms.functional.to_pil_image(rot_tensor.squeeze(0).cpu().clamp(0, 1))
        rot_overlay = rot_img_pil.copy()
        draw_boxes(rot_overlay, boxes_xywh_to_xyxy(rot_boxes_xywh), rot_scores, f'rot{int(angle)}_raw', angle_color, args.score_thr)
        if rot_kps is not None:
            draw_keypoints(rot_overlay, rot_kps, angle_color)
        outputs.append((rot_overlay, f'rot_{int(angle)}_raw.png'))
        rotated_raw_views.append(rot_overlay)

        # Convert to normalized cxcywh then inverse rotate back to original frame
        rot_boxes_cxcywh = boxes_xywh_to_cxcywh(rot_boxes_xywh, width, height)
        rot_boxes_cxcywh_orig = inverse_rotate_boxes_cxcywh(rot_boxes_cxcywh, angle)
        rot_boxes_xyxy = boxes_cxcywh_to_xyxy(rot_boxes_cxcywh_orig, width, height)

        rotated_overlay = image.copy()
        draw_boxes(rotated_overlay, rot_boxes_xyxy, rot_scores, f'rot{int(angle)}', angle_color, args.score_thr)
        # Inverse-rotate predicted keypoints back to original frame and draw
        if rot_kps is not None:
            inv_kps = []
            for kp in rot_kps:
                coords = []
                for k in range(kp.shape[0]):
                    x, y = float(kp[k, 0]), float(kp[k, 1])
                    xi, yi = rotate_point_xy(x, y, width, height, -angle)
                    coords.append((xi, yi, float(kp[k, 2]) if kp.shape[1] > 2 else 1.0))
                inv_kps.append(torch.tensor(coords))
            draw_keypoints(rotated_overlay, inv_kps, angle_color)
        outputs.append((rotated_overlay, f'rot_{int(angle)}.png'))
        inverse_overlays.append(rotated_overlay)

        # add to combined canvas
        draw_boxes(combined_inverse, rot_boxes_xyxy, rot_scores, f'rot{int(angle)}', angle_color, args.score_thr)
        if rot_kps is not None:
            draw_keypoints(combined_inverse, inv_kps, angle_color)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a composite panel: top row inverse overlays, bottom row raw rotated views
    panel_columns = max(len(inverse_overlays), len(rotated_raw_views))
    panel_width = panel_columns * width
    panel_height = height * 2
    composite = Image.new('RGB', (panel_width, panel_height), color=(30, 30, 30))

    for idx, img in enumerate(inverse_overlays):
        composite.paste(img, (idx * width, 0))

    for idx, img in enumerate(rotated_raw_views):
        composite.paste(img, (idx * width, height))

    outputs.append((combined_inverse, 'combined_overlay.png'))
    outputs.append((composite, 'panel_summary.png'))

    for img, name in outputs:
        save_path = out_dir / name
        img.save(save_path)
        print(f'Saved {save_path}')


if __name__ == '__main__':
    main()
