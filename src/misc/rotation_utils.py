"""Utility helpers for rotating images, boxes, keypoints, and targets."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F

__all__ = [
    "rotate_image_tensor",
    "rotate_boxes_cxcywh",
    "rotate_keypoints",
    "rotate_targets",
    "inverse_rotate_boxes_cxcywh",
]


def _rotation_affine_matrix(angle: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    theta = math.radians(float(angle))
    cos_a = math.cos(theta)
    sin_a = math.sin(theta)
    matrix = torch.tensor([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
    ], dtype=dtype, device=device)
    return matrix


def rotate_image_tensor(images: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate a batch of images by an arbitrary angle in degrees."""
    if abs(angle) % 360 < 1e-6:
        return images

    b, c, h, w = images.shape
    device = images.device

    theta = _rotation_affine_matrix(angle, device, torch.float32)
    theta = theta.unsqueeze(0).repeat(b, 1, 1)
    # grid_sample expects float32/64
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=False)
    rotated = F.grid_sample(images.to(torch.float32), grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    return rotated.to(images.dtype)


def rotate_boxes_cxcywh(boxes: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate normalized bounding boxes by arbitrary angle."""
    if boxes.numel() == 0:
        return boxes.clone()

    theta = math.radians(float(angle))
    cos_a = boxes.new_tensor(math.cos(theta))
    sin_a = boxes.new_tensor(math.sin(theta))

    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - w / 2.0
    x2 = cx + w / 2.0
    y1 = cy - h / 2.0
    y2 = cy + h / 2.0

    corners_x = torch.stack([x1, x2, x1, x2], dim=-1)
    corners_y = torch.stack([y1, y1, y2, y2], dim=-1)

    x_shift = corners_x - 0.5
    y_shift = corners_y - 0.5
    rot_x = x_shift * cos_a + y_shift * sin_a + 0.5
    rot_y = -x_shift * sin_a + y_shift * cos_a + 0.5

    new_x1 = rot_x.min(dim=-1).values.clamp(0.0, 1.0)
    new_y1 = rot_y.min(dim=-1).values.clamp(0.0, 1.0)
    new_x2 = rot_x.max(dim=-1).values.clamp(0.0, 1.0)
    new_y2 = rot_y.max(dim=-1).values.clamp(0.0, 1.0)

    new_w = (new_x2 - new_x1).clamp_min(1e-6)
    new_h = (new_y2 - new_y1).clamp_min(1e-6)
    new_cx = (new_x1 + new_x2) / 2.0
    new_cy = (new_y1 + new_y2) / 2.0

    return torch.stack([new_cx, new_cy, new_w, new_h], dim=-1)


def inverse_rotate_boxes_cxcywh(boxes: torch.Tensor, angle: float) -> torch.Tensor:
    if abs(angle) % 360 < 1e-6:
        return boxes.clone()
    return rotate_boxes_cxcywh(boxes, -angle)


def rotate_keypoints(keypoints: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate keypoints in normalized coordinates (x, y, v)."""
    if keypoints.numel() == 0:
        return keypoints.clone()

    theta = math.radians(float(angle))
    cos_a = keypoints.new_tensor(math.cos(theta))
    sin_a = keypoints.new_tensor(math.sin(theta))

    xy = keypoints[..., :2]
    x = xy[..., 0]
    y = xy[..., 1]
    x_shift = x - 0.5
    y_shift = y - 0.5
    new_x = x_shift * cos_a + y_shift * sin_a + 0.5
    new_y = -x_shift * sin_a + y_shift * cos_a + 0.5

    rotated = torch.stack([new_x, new_y], dim=-1)
    if keypoints.shape[-1] > 2:
        rotated = torch.cat([rotated, keypoints[..., 2:]], dim=-1)
    return rotated


def rotate_targets(targets: List[dict], angle: float) -> List[dict]:
    rotated_targets: List[dict] = []
    for tgt in targets:
        new_tgt = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in tgt.items()
        }
        device = None
        dtype = None
        if isinstance(new_tgt.get('boxes'), torch.Tensor):
            device = new_tgt['boxes'].device
            dtype = new_tgt['boxes'].dtype

        if 'boxes' in new_tgt and isinstance(new_tgt['boxes'], torch.Tensor) and new_tgt['boxes'].numel() > 0:
            new_tgt['boxes'] = rotate_boxes_cxcywh(new_tgt['boxes'], angle)

        if 'keypoints' in new_tgt and isinstance(new_tgt['keypoints'], torch.Tensor) and new_tgt['keypoints'].numel() > 0:
            new_tgt['keypoints'] = rotate_keypoints(new_tgt['keypoints'], angle)

        if 'angle_weight' in new_tgt and isinstance(new_tgt['angle_weight'], torch.Tensor) and new_tgt['angle_weight'].numel() > 0:
            new_tgt['angle_weight'] = new_tgt['angle_weight'].clone()

        current_angle = 0.0
        if 'rotation_angle' in new_tgt:
            rotation_value = new_tgt['rotation_angle']
            if isinstance(rotation_value, torch.Tensor):
                current_angle = float(rotation_value.item())
            else:
                current_angle = float(rotation_value)
        new_angle = current_angle + angle

        if device is None:
            if isinstance(new_tgt.get('keypoints'), torch.Tensor):
                device = new_tgt['keypoints'].device
            else:
                device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float32

        new_tgt['rotation_angle'] = torch.tensor(new_angle, device=device, dtype=dtype)

        rotated_targets.append(new_tgt)
    return rotated_targets
