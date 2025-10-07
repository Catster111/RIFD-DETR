#!/usr/bin/env python3
"""Visualize keypoint heatmaps (polar or cartesian) for a single image."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import YAMLConfig
from src.solver import TASKS
from src.zoo.rtdetr.rtdetr_keypoint_head import POLAR_RADIUS_SCALE, heatmap_expectation_xy

KEYPOINT_NAMES = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
OVERLAY_CMAPS = ['Reds', 'Blues', 'Oranges', 'Purples', 'Greens']
POINT_COLORS = ['red', 'blue', 'orange', 'purple', 'green']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize keypoint heatmaps from a checkpoint")
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', type=str, default=None, help='Output image (png)')
    parser.add_argument('--topk', type=int, default=1, help='Number of detections to visualize after postprocess')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Score threshold for detections')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--query-thr', type=float, default=0.5, help='Sigmoid threshold on raw query logits')
    return parser.parse_args()


def load_model(cfg_path: str, checkpoint: str, device: torch.device):
    cfg = YAMLConfig(cfg_path, resume=checkpoint)
    task_name = cfg.yaml_cfg['task']
    solver = TASKS[task_name](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    model.to(device).eval()
    post = solver.postprocessor
    domain = cfg.yaml_cfg.get('RTDETRTransformerv2', {}).get('keypoint_heatmap_domain', 'cartesian')
    return model, post, domain


def decode_topk(outputs, postprocessor, orig_target_sizes):
    logits = outputs['pred_logits']
    boxes = outputs['pred_boxes']
    bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xywh')
    img_h, img_w = orig_target_sizes.unbind(1)
    scale_wh = torch.stack([img_w, img_h, img_w, img_h], dim=1).unsqueeze(1)
    bbox_pred = bbox_pred * scale_wh
    bbox_pred[..., 0] = bbox_pred[..., 0].clamp(min=0)
    bbox_pred[..., 1] = bbox_pred[..., 1].clamp(min=0)
    bbox_pred[..., 2] = bbox_pred[..., 2].clamp(min=1)
    bbox_pred[..., 3] = bbox_pred[..., 3].clamp(min=1)

    scores = torch.sigmoid(logits)
    k = min(postprocessor.num_top_queries, scores.flatten(1).shape[1])
    scores_top, index = torch.topk(scores.flatten(1), k, dim=-1)
    indices = index // postprocessor.num_classes
    boxes_sel = bbox_pred.gather(1, indices.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
    return boxes_sel, scores_top, indices


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, postprocessor, domain = load_model(args.config, args.checkpoint, device)

    img_path = Path(args.image)
    image = Image.open(img_path).convert('RGB')
    orig_w, orig_h = image.size

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    orig_sizes = torch.tensor([[orig_h, orig_w]], dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(tensor)

    boxes_sel, scores_sel, query_indices = decode_topk(outputs, postprocessor, orig_sizes)
    heatmaps = outputs['pred_keypoint_heatmaps'][0]
    offsets = outputs['pred_keypoint_offsets'][0]
    query_logits = torch.sigmoid(outputs['pred_logits'][0]).squeeze(-1)

    scores = scores_sel[0].detach().cpu().numpy()
    boxes = boxes_sel[0].detach().cpu().numpy()
    queries = query_indices[0].detach().cpu().numpy()

    keep = scores >= args.score_thr
    scores = scores[keep]
    boxes = boxes[keep]
    queries = queries[keep]

    if len(scores) == 0:
        print("No detections above threshold")
        return

    order = np.argsort(-scores)[: args.topk]

    for idx in order:
        score = scores[idx]
        box = boxes[idx]
        q_idx = int(queries[idx])
        if float(query_logits[q_idx].item()) < args.query_thr:
            continue

        heatmap = heatmaps[q_idx][:len(KEYPOINT_NAMES)]
        offset = offsets[q_idx][:len(KEYPOINT_NAMES)]

        fig = plt.figure(figsize=(18, 9))
        grid = fig.add_gridspec(2, 3)

        ax_img = fig.add_subplot(grid[:, 0])
        ax_heatmap = fig.add_subplot(grid[:, 1:])

        ax_img.imshow(image)
        ax_img.set_title(f'Detection (score={score:.2f})')
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax_img.add_patch(rect)

        for k, color in enumerate(POINT_COLORS):
            bx, by = heatmap_expectation_xy(heatmap[k], domain, POLAR_RADIUS_SCALE)
            fx = float(torch.clamp(bx + offset[k][0], 0, 1).item())
            fy = float(torch.clamp(by + offset[k][1], 0, 1).item())
            kp_x = x + fx * w
            kp_y = y + fy * h
            ax_img.plot(kp_x, kp_y, marker='o', markersize=6, color=color)
            ax_img.text(kp_x + 3, kp_y, KEYPOINT_NAMES[k], color='yellow', fontsize=8)

        ax_heatmap.clear()
        ax_heatmap.set_title('Combined Keypoint Heatmaps')
        if domain == 'polar':
            extent = [0, 360, 0, 1]
            origin = 'lower'
            aspect = 'auto'
            ax_heatmap.set_xlabel('Angle (deg)')
            ax_heatmap.set_ylabel('Radius')
        else:
            extent = None
            origin = 'upper'
            aspect = 'equal'
            ax_heatmap.set_xlabel('X (grid)')
            ax_heatmap.set_ylabel('Y (grid)')

        legend_handles = []
        for k, (cmap, point_color) in enumerate(zip(OVERLAY_CMAPS, POINT_COLORS)):
            hm = heatmap[k].sigmoid().cpu().numpy()
            ax_heatmap.imshow(
                hm,
                cmap=cmap,
                alpha=0.35,
                origin=origin,
                extent=extent,
                aspect=aspect,
            )
            legend_handles.append(mpatches.Patch(color=plt.get_cmap(cmap)(0.85), label=KEYPOINT_NAMES[k]))

        if legend_handles:
            ax_heatmap.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=False)

        fig.tight_layout()
        out_path = args.output if args.output else img_path.with_suffix('.heatmaps.png')
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved visualization to {out_path}')


if __name__ == '__main__':
    main()
