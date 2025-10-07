#!/usr/bin/env python3

"""Create a synthetic example to illustrate rotational consistency operations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.misc.rotation_utils import (
    rotate_boxes_cxcywh,
    inverse_rotate_boxes_cxcywh,
)


def make_canvas(size: int = 512) -> Image.Image:
    image = Image.new('RGB', (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    step = size // 8
    for x in range(0, size, step):
        draw.line([(x, 0), (x, size)], fill=(220, 220, 220), width=1)
    for y in range(0, size, step):
        draw.line([(0, y), (size, y)], fill=(220, 220, 220), width=1)
    return image


def draw_face(draw: ImageDraw.ImageDraw, center: Tuple[float, float], radius: float, color: Tuple[int, int, int]):
    cx, cy = center
    left = cx - radius
    top = cy - radius
    right = cx + radius
    bottom = cy + radius
    draw.ellipse([left, top, right, bottom], fill=(255, 255, 255), outline=color, width=3)

    eye_radius = radius * 0.15
    eye_offset_x = radius * 0.35
    eye_offset_y = radius * 0.35
    draw.ellipse([cx - eye_offset_x - eye_radius, cy - eye_offset_y - eye_radius,
                  cx - eye_offset_x + eye_radius, cy - eye_offset_y + eye_radius], fill=color)
    draw.ellipse([cx + eye_offset_x - eye_radius, cy - eye_offset_y - eye_radius,
                  cx + eye_offset_x + eye_radius, cy - eye_offset_y + eye_radius], fill=color)

    mouth_width = radius * 0.6
    mouth_height = radius * 0.25
    draw.arc([cx - mouth_width, cy + radius * 0.15 - mouth_height,
              cx + mouth_width, cy + radius * 0.15 + mouth_height], start=20, end=160, fill=color, width=3)


def draw_boxes(image: Image.Image, boxes_xyxy: torch.Tensor, labels: List[str], color: Tuple[int, int, int]):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    label_bg = tuple(min(255, c + 120) for c in color)
    luminance = (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114)
    text_fill = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    for box, label in zip(boxes_xyxy.tolist(), labels):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(bbox, fill=label_bg)
        draw.text((x1, y1), label, fill=text_fill, font=font)


def boxes_cxcywh_to_xyxy(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return torch.stack([x1, y1, x2, y2], dim=-1)


def boxes_xyxy_to_cxcywh(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return torch.stack([cx, cy, w, h], dim=-1)


def main():
    parser = argparse.ArgumentParser('Synthetic rotational consistency demo')
    parser.add_argument('--size', type=int, default=512, help='Canvas size (square)')
    parser.add_argument('--angles', nargs='+', type=float, default=[90, 180, 270], help='Angles to illustrate')
    parser.add_argument('--output-dir', type=str, default='rot_consistency_demo', help='Output directory')
    parser.add_argument('--step-angle', type=float, default=None, help='Generate step-by-step panel for this angle (e.g. 90)')
    args = parser.parse_args()

    size = args.size
    width = height = float(size)

    base = make_canvas(size)
    # Define synthetic boxes (absolute coordinates in pixels)
    raw_boxes_xyxy = torch.tensor([
        [60, 100, 200, 240],
        [220, 140, 360, 280],
        [150, 310, 310, 450],
        [330, 220, 460, 350],
    ], dtype=torch.float32)
    labels = ["A", "B", "C", "D"]

    base_boxes_cxcywh = boxes_xyxy_to_cxcywh(raw_boxes_xyxy, width, height)

    predicted_boxes_xyxy = raw_boxes_xyxy + torch.tensor([
        [-8, -6, -8, -6],
        [-6, 12, -6, 12],
        [12, -8, 12, -8],
        [10, 6, 10, 6],
    ], dtype=torch.float32)
    pred_boxes_cxcywh = boxes_xyxy_to_cxcywh(predicted_boxes_xyxy, width, height)

    base_faces = base.copy()
    draw_base = ImageDraw.Draw(base_faces)
    face_colors = [(220, 30, 30), (76, 175, 80), (0, 150, 200), (155, 85, 245)]
    for box, color in zip(raw_boxes_xyxy.tolist(), face_colors):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        radius = min((x2 - x1), (y2 - y1)) / 2 * 0.7
        draw_face(draw_base, (cx, cy), radius, color)
    base = base_faces

    base_orig_only = base_faces.copy()
    draw_boxes(base_orig_only, raw_boxes_xyxy, [f'orig {lbl}' for lbl in labels], color=(220, 30, 30))

    base_pred_only = base_faces.copy()
    draw_boxes(base_pred_only, predicted_boxes_xyxy, [f'pred {lbl}' for lbl in labels], color=(0, 0, 0))

    base_overlay = base_faces.copy()
    draw_boxes(base_overlay, raw_boxes_xyxy, [f'orig {lbl}' for lbl in labels], color=(220, 30, 30))
    draw_boxes(base_overlay, predicted_boxes_xyxy, [f'pred {lbl}' for lbl in labels], color=(0, 0, 0))

    outputs: List[Tuple[Image.Image, str]] = []
    outputs.append((base_orig_only, '00_original_boxes.png'))
    outputs.append((base_pred_only, '01_pred_boxes.png'))
    outputs.append((base_overlay.copy(), '02_overlay_loss.png'))

    tensor_transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    base_tensor = tensor_transform(base)

    combined_inverse = base_overlay.copy()
    panel_rows = 2
    panel_cols = max(len(args.angles) + 1, 2)
    panel = Image.new('RGB', (int(width * panel_cols), int(height * panel_rows)), color=(30, 30, 30))
    panel.paste(base_overlay, (0, 0))

    colors = [
        (76, 175, 80),
        (0, 150, 200),
        (155, 85, 245),
    ]

    for idx, angle in enumerate(args.angles, start=1):
        k = int(angle // 90) % 4
        rot_tensor = torch.rot90(base_tensor, -k, dims=(-2, -1))
        rot_image = to_pil(rot_tensor)

        # Rotate boxes forward and backward
        rot_boxes_fwd = rotate_boxes_cxcywh(base_boxes_cxcywh, angle)
        rot_boxes_xyxy = boxes_cxcywh_to_xyxy(rot_boxes_fwd, width, height)
        rot_pred_boxes_fwd = rotate_boxes_cxcywh(pred_boxes_cxcywh, angle)
        rot_pred_boxes_xyxy = boxes_cxcywh_to_xyxy(rot_pred_boxes_fwd, width, height)

        rot_overlay = rot_image.copy()
        draw_boxes(rot_overlay, rot_boxes_xyxy, [f'rot{int(angle)} {lbl}' for lbl in labels], colors[(idx - 1) % len(colors)])
        draw_boxes(rot_overlay, rot_pred_boxes_xyxy, [f'rot{int(angle)} pred {lbl}' for lbl in labels], (0, 0, 0))
        outputs.append((rot_overlay, f'01_rot{int(angle)}_raw.png'))
        panel.paste(rot_overlay, (int(width * idx), int(height)))

        # Rotate predictions back to the original frame
        inverse_boxes = inverse_rotate_boxes_cxcywh(rot_boxes_fwd, angle)
        inverse_xyxy = boxes_cxcywh_to_xyxy(inverse_boxes, width, height)
        inverse_overlay = base.copy()
        draw_boxes(inverse_overlay, inverse_xyxy, [f'rot{int(angle)}->orig {lbl}' for lbl in labels], colors[(idx - 1) % len(colors)])
        inverse_pred_boxes = inverse_rotate_boxes_cxcywh(rot_pred_boxes_fwd, angle)
        inverse_pred_xyxy = boxes_cxcywh_to_xyxy(inverse_pred_boxes, width, height)
        draw_boxes(inverse_overlay, inverse_pred_xyxy, [f'rot{int(angle)}->orig pred {lbl}' for lbl in labels], (0, 0, 0))

        arrow_color = colors[(idx - 1) % len(colors)]
        arrow = ImageDraw.Draw(inverse_overlay)
        arrow.line([(width * 0.88, height * 0.15), (width * 0.72, height * 0.3)], fill=arrow_color, width=5)
        arrow.polygon([
            (width * 0.88, height * 0.15),
            (width * 0.83, height * 0.14),
            (width * 0.86, height * 0.19)
        ], fill=arrow_color)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        arrow.text((width * 0.7, height * 0.05), f"rotate {int(angle)}°", fill=arrow_color, font=font)

        outputs.append((inverse_overlay, f'02_rot{int(angle)}_inverse.png'))

        draw_boxes(combined_inverse, inverse_xyxy, [f'rot{int(angle)} {lbl}' for lbl in labels], colors[(idx - 1) % len(colors)])
        draw_boxes(combined_inverse, inverse_pred_xyxy, [f'rot{int(angle)} pred {lbl}' for lbl in labels], (0, 0, 0))
        panel.paste(inverse_overlay, (int(width * idx), 0))
        panel_arrow = ImageDraw.Draw(panel)
        panel_arrow.line([(int(width * idx + width * 0.88), int(height * 0.15)), (int(width * idx + width * 0.72), int(height * 0.3))], fill=arrow_color, width=5)
        panel_arrow.polygon([
            (int(width * idx + width * 0.88), int(height * 0.15)),
            (int(width * idx + width * 0.83), int(height * 0.14)),
            (int(width * idx + width * 0.86), int(height * 0.19))
        ], fill=arrow_color)
        panel_arrow.text((int(width * idx + width * 0.7), int(height * 0.05)), f"rotate {int(angle)}°", fill=arrow_color, font=font)

        if args.step_angle is not None and abs(angle - args.step_angle) < 1e-3:
            stage_font = font
            stage_cols = 3
            stage_panel = Image.new('RGB', (int(width * stage_cols), int(height + height * 0.15)), color=(250, 250, 250))
            stage_panel.paste(base_overlay, (0, 0))
            stage_panel.paste(rot_overlay, (int(width), 0))
            stage_panel.paste(inverse_overlay, (int(width * 2), 0))
            stage_draw = ImageDraw.Draw(stage_panel)
            titles = ['Stage 1: Original', f'Stage 2: Rotate {int(angle)}°', 'Stage 3: Rotate back']
            for col, title in enumerate(titles):
                stage_draw.text((int(width * col + width * 0.03), int(height * 1.03)), title, fill=(60, 60, 60), font=stage_font)
            outputs.append((stage_panel, f'step_panel_{int(angle)}.png'))

    outputs.append((combined_inverse, '03_combined_overlay.png'))
    outputs.append((panel, '04_panel_summary.png'))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img, name in outputs:
        path = out_dir / name
        img.save(path)
        print(f'Saved {path}')


if __name__ == '__main__':
    main()
