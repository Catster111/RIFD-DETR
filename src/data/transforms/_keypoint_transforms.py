"""
Keypoint-specific transforms and helpers for RT-DETRv2.
"""

import random
from typing import Any, Dict, Optional, Tuple

import torch
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T
from torch import nn
from torchvision import tv_tensors
from torchvision.ops import box_iou
from PIL import Image

from ...core import register


class ConvertKeypoints(T.Transform):
    """Normalize keypoints to [0, 1] range relative to image size"""

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not isinstance(inpt, torch.Tensor):
            return inpt

        if 'spatial_size' in params:
            height, width = params['spatial_size']
        else:
            height, width = 640, 640

        if self.normalize and inpt.numel() > 0:
            out = inpt.clone()
            out[:, :, 0] = inpt[:, :, 0] / width
            out[:, :, 1] = inpt[:, :, 1] / height
            out[:, :, 0] = torch.clamp(out[:, :, 0], 0, 1)
            out[:, :, 1] = torch.clamp(out[:, :, 1], 0, 1)
            return out

        return inpt


@register()
class NormalizeKeypoints(T.Transform):
    """Normalize keypoints to [0,1] using current spatial size"""

    def forward(self, *inputs: Any):
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target, dataset = inputs

        if hasattr(image, 'shape'):
            _, height, width = image.shape
        else:
            width, height = image.size

        if 'keypoints' in target and target['keypoints'].numel() > 0:
            kp_abs = target['keypoints']

            if 'boxes' in target and target['boxes'].numel() > 0:
                boxes = target['boxes']
                if isinstance(boxes, tv_tensors.BoundingBoxes):
                    boxes_tensor = boxes.as_subclass(torch.Tensor).clone()
                    fmt = boxes.format
                    canvas = getattr(boxes, 'canvas_size', None) or getattr(boxes, 'spatial_size', None)
                else:
                    boxes_tensor = boxes.clone()
                    fmt = None
                    canvas = (height, width)

                if fmt is not None and fmt != tv_tensors.BoundingBoxFormat.XYXY:
                    boxes_xyxy = tv_tensors.convert_format(
                        boxes_tensor, fmt, tv_tensors.BoundingBoxFormat.XYXY
                    )
                else:
                    boxes_xyxy = boxes_tensor

                vis = kp_abs[..., 2] > 0
                if vis.any():
                    kp_x = kp_abs[..., 0]
                    kp_y = kp_abs[..., 1]
                    large = torch.full_like(kp_x, float('inf'))
                    neg_large = torch.full_like(kp_x, float('-inf'))

                    kp_x_min = torch.where(vis, kp_x, large).min(dim=1).values
                    kp_y_min = torch.where(vis, kp_y, large).min(dim=1).values
                    kp_x_max = torch.where(vis, kp_x, neg_large).max(dim=1).values
                    kp_y_max = torch.where(vis, kp_y, neg_large).max(dim=1).values

                    has_vis = vis.any(dim=1)
                    boxes_xyxy[:, 0] = torch.where(has_vis, torch.minimum(boxes_xyxy[:, 0], kp_x_min), boxes_xyxy[:, 0])
                    boxes_xyxy[:, 1] = torch.where(has_vis, torch.minimum(boxes_xyxy[:, 1], kp_y_min), boxes_xyxy[:, 1])
                    boxes_xyxy[:, 2] = torch.where(has_vis, torch.maximum(boxes_xyxy[:, 2], kp_x_max), boxes_xyxy[:, 2])
                    boxes_xyxy[:, 3] = torch.where(has_vis, torch.maximum(boxes_xyxy[:, 3], kp_y_max), boxes_xyxy[:, 3])

                boxes_xyxy[:, 0] = torch.clamp(boxes_xyxy[:, 0] - 2.0, 0, width)
                boxes_xyxy[:, 1] = torch.clamp(boxes_xyxy[:, 1] - 2.0, 0, height)
                boxes_xyxy[:, 2] = torch.clamp(boxes_xyxy[:, 2] + 2.0, 0, width)
                boxes_xyxy[:, 3] = torch.clamp(boxes_xyxy[:, 3] + 2.0, 0, height)

                if fmt is not None and fmt != tv_tensors.BoundingBoxFormat.XYXY:
                    boxes_tensor = tv_tensors.convert_format(
                        boxes_xyxy, tv_tensors.BoundingBoxFormat.XYXY, fmt
                    )
                    target['boxes'] = tv_tensors.BoundingBoxes(
                        boxes_tensor,
                        format=fmt,
                        canvas_size=canvas,
                    )
                elif fmt is not None:
                    target['boxes'] = tv_tensors.BoundingBoxes(
                        boxes_xyxy,
                        format=fmt,
                        canvas_size=canvas,
                    )
                else:
                    target['boxes'] = boxes_xyxy

            kp = kp_abs.clone()
            kp[:, :, 0] = kp[:, :, 0] / width
            kp[:, :, 1] = kp[:, :, 1] / height
            kp[:, :, 0] = torch.clamp(kp[:, :, 0], 0, 1)
            kp[:, :, 1] = torch.clamp(kp[:, :, 1], 0, 1)
            target['keypoints'] = kp

        return image, target, dataset


def _get_image_size(image) -> Tuple[int, int]:
    if hasattr(image, 'size'):
        return image.size  # type: ignore[return-value]
    if isinstance(image, torch.Tensor):
        *_, h, w = image.shape
        return w, h
    raise TypeError('Unsupported image type: cannot retrieve size')


def _clone_boxes(boxes):
    if isinstance(boxes, tv_tensors.BoundingBoxes):
        tensor = boxes.as_subclass(torch.Tensor).clone()
        fmt = boxes.format
        canvas = getattr(boxes, 'canvas_size', None) or getattr(boxes, 'spatial_size', None)
        return tensor, fmt, canvas
    return boxes.clone(), None, None


def _restore_boxes(tensor, fmt, canvas):
    if fmt is None:
        return tensor
    return tv_tensors.BoundingBoxes(tensor, format=fmt, canvas_size=canvas)


def _keypoints_visible_mask(kps: torch.Tensor) -> torch.Tensor:
    return kps[..., 2] > 0


def _update_visibility(kps: torch.Tensor, width: int, height: int) -> torch.Tensor:
    visible = _keypoints_visible_mask(kps)
    valid_x = (kps[..., 0] >= 0) & (kps[..., 0] <= width)
    valid_y = (kps[..., 1] >= 0) & (kps[..., 1] <= height)
    still_visible = visible & valid_x & valid_y
    kps[..., 2] = torch.where(still_visible, kps[..., 2], torch.zeros_like(kps[..., 2]))
    return kps


@register()
class FilterDegenerateBoxesWithKeypoints(T.Transform):
    """Remove boxes (and associated keypoints/attributes) whose width or height falls below a threshold."""

    def __init__(self, min_size: float = 1.0) -> None:
        super().__init__()
        self.min_size = float(min_size)

    def forward(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target, dataset = inputs

        if 'boxes' not in target or target['boxes'].numel() == 0:
            return image, target, dataset

        boxes_tensor, fmt, canvas = _clone_boxes(target['boxes'])
        boxes_xyxy = boxes_tensor
        if fmt is not None and fmt != tv_tensors.BoundingBoxFormat.XYXY:
            boxes_xyxy = tv_tensors.convert_format(boxes_tensor, fmt, tv_tensors.BoundingBoxFormat.XYXY)

        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        valid = (widths >= self.min_size) & (heights >= self.min_size)

        if not valid.any():
            empty_boxes = boxes_xyxy.new_zeros((0, 4))
            target['boxes'] = _restore_boxes(empty_boxes, fmt, canvas)

            if 'keypoints' in target and isinstance(target['keypoints'], torch.Tensor):
                target['keypoints'] = target['keypoints'].new_zeros((0,) + target['keypoints'].shape[1:])

            for key in ('labels', 'area', 'iscrowd', 'keypoints_visibility', 'bbox_offsets'):
                if key in target and isinstance(target[key], torch.Tensor):
                    target[key] = target[key].new_zeros((0,) + target[key].shape[1:])

            return image, target, dataset

        filtered_boxes = boxes_xyxy[valid]
        target['boxes'] = _restore_boxes(filtered_boxes, fmt, canvas)

        def _mask_tensor(value):
            if isinstance(value, torch.Tensor) and value.shape[0] == valid.shape[0]:
                return value[valid]
            return value

        for key in ('labels', 'area', 'iscrowd', 'keypoints_visibility', 'bbox_offsets'):
            if key in target:
                target[key] = _mask_tensor(target[key])

        if 'keypoints' in target and isinstance(target['keypoints'], torch.Tensor) and target['keypoints'].numel() > 0:
            target['keypoints'] = target['keypoints'][valid]

        return image, target, dataset


@register()
class RandomHorizontalFlipWithKeypoints(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target, dataset = inputs

        if random.random() > self.p:
            return image, target, dataset

        width, height = _get_image_size(image)
        image = F.hflip(image)

        if 'boxes' in target and target['boxes'].numel() > 0:
            boxes_tensor, fmt, canvas = _clone_boxes(target['boxes'])
            boxes_xyxy = boxes_tensor
            if fmt is not None and fmt != tv_tensors.BoundingBoxFormat.XYXY:
                boxes_xyxy = tv_tensors.convert_format(boxes_tensor, fmt, tv_tensors.BoundingBoxFormat.XYXY)

            x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
            flipped_xyxy = torch.stack([width - x2, y1, width - x1, y2], dim=1)

            if fmt is not None and fmt != tv_tensors.BoundingBoxFormat.XYXY:
                flipped = tv_tensors.convert_format(flipped_xyxy, tv_tensors.BoundingBoxFormat.XYXY, fmt)
            else:
                flipped = flipped_xyxy

            target['boxes'] = _restore_boxes(flipped, fmt, canvas or (height, width))

        if 'keypoints' in target and target['keypoints'].numel() > 0:
            kps = target['keypoints'].clone()  # [N, K, 3]
            mask = _keypoints_visible_mask(kps)
            kps[..., 0][mask] = width - kps[..., 0][mask]

            # Swap left/right symmetric keypoints after flip
            swap_pairs = [(0, 1), (3, 4)]  # (left_eye ↔ right_eye), (left_mouth ↔ right_mouth)
            for a, b in swap_pairs:
                tmp = kps[:, a, :].clone()
                kps[:, a, :] = kps[:, b, :]
                kps[:, b, :] = tmp

            target['keypoints'] = _update_visibility(kps, width, height)

        return image, target, dataset


@register()
class RandomZoomOutWithKeypoints(nn.Module):
    def __init__(self, fill: int = 0, side_range: Tuple[float, float] = (1.0, 4.0), p: float = 0.5) -> None:
        super().__init__()
        self.fill = fill
        self.side_range = side_range
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target, dataset = inputs

        if random.random() > self.p:
            return image, target, dataset

        width, height = _get_image_size(image)
        ratio = random.uniform(self.side_range[0], self.side_range[1])
        canvas_w = int(width * ratio)
        canvas_h = int(height * ratio)

        if canvas_w == width and canvas_h == height:
            return image, target, dataset

        left = random.randint(0, canvas_w - width)
        top = random.randint(0, canvas_h - height)

        if isinstance(image, torch.Tensor):
            c = image.shape[0]
            new_image = torch.full((c, canvas_h, canvas_w), self.fill, dtype=image.dtype)
            new_image[:, top:top + height, left:left + width] = image
        else:
            new_image = Image.new(image.mode, (canvas_w, canvas_h), color=self.fill)
            new_image.paste(image, (left, top))

        if 'boxes' in target and target['boxes'].numel() > 0:
            boxes_tensor, fmt, canvas = _clone_boxes(target['boxes'])
            boxes_xyxy = boxes_tensor
            if fmt is not None and fmt != tv_tensors.BoundingBoxFormat.XYXY:
                boxes_xyxy = tv_tensors.convert_format(boxes_tensor, fmt, tv_tensors.BoundingBoxFormat.XYXY)
            boxes_xyxy[:, 0::2] += left
            boxes_xyxy[:, 1::2] += top
            if fmt is not None and fmt != tv_tensors.BoundingBoxFormat.XYXY:
                boxes_tensor = tv_tensors.convert_format(boxes_xyxy, tv_tensors.BoundingBoxFormat.XYXY, fmt)
            else:
                boxes_tensor = boxes_xyxy
            target['boxes'] = _restore_boxes(boxes_tensor, fmt, (canvas_h, canvas_w))

        if 'keypoints' in target and target['keypoints'].numel() > 0:
            kps = target['keypoints'].clone()
            mask = _keypoints_visible_mask(kps)
            kps[..., 0][mask] += left
            kps[..., 1][mask] += top
            target['keypoints'] = _update_visibility(kps, canvas_w, canvas_h)

        return new_image, target, dataset


@register()
class RandomIoUCropWithKeypoints(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[Tuple[float, ...]] = None,
        trials: int = 40,
        p: float = 0.8,
    ) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.options = sampler_options or (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
        self.trials = trials
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target, dataset = inputs

        if random.random() > self.p:
            return image, target, dataset

        width, height = _get_image_size(image)

        if 'boxes' not in target or target['boxes'].numel() == 0:
            return image, target, dataset

        boxes_tensor, fmt, canvas = _clone_boxes(target['boxes'])
        boxes_xyxy = boxes_tensor
        if fmt is not None and fmt != tv_tensors.BoundingBoxFormat.XYXY:
            boxes_xyxy = tv_tensors.convert_format(boxes_tensor, fmt, tv_tensors.BoundingBoxFormat.XYXY)

        params = self._sample_crop(boxes_xyxy, width, height)
        if params is None:
            return image, target, dataset

        left, top, new_w, new_h, keep_mask = params

        # Crop image
        if isinstance(image, torch.Tensor):
            cropped_image = image[:, top:top + new_h, left:left + new_w]
        else:
            cropped_image = image.crop((left, top, left + new_w, top + new_h))

        # Adjust boxes
        kept_boxes = boxes_xyxy[keep_mask].clone()
        kept_boxes[:, 0::2] = kept_boxes[:, 0::2].clamp(min=left, max=left + new_w) - left
        kept_boxes[:, 1::2] = kept_boxes[:, 1::2].clamp(min=top, max=top + new_h) - top

        # Remove degenerate boxes
        widths = (kept_boxes[:, 2] - kept_boxes[:, 0]).clamp(min=0)
        heights = (kept_boxes[:, 3] - kept_boxes[:, 1]).clamp(min=0)
        valid = (widths > 1) & (heights > 1)
        kept_boxes = kept_boxes[valid]

        if kept_boxes.numel() == 0:
            return image, target, dataset

        if fmt is not None and fmt != tv_tensors.BoundingBoxFormat.XYXY:
            boxes_out = tv_tensors.convert_format(kept_boxes, tv_tensors.BoundingBoxFormat.XYXY, fmt)
        else:
            boxes_out = kept_boxes
        target['boxes'] = _restore_boxes(boxes_out, fmt, (new_h, new_w))

        # Adjust labels and other fields based on mask
        for key in ['labels', 'area', 'iscrowd']:
            if key in target and len(target[key]) == len(keep_mask):
                target[key] = target[key][keep_mask][valid]

        # Update keypoints
        if 'keypoints' in target and target['keypoints'].numel() > 0:
            keypoints = target['keypoints'][keep_mask].clone()
            keypoints = keypoints[valid]
            mask = _keypoints_visible_mask(keypoints)
            keypoints[..., 0][mask] -= left
            keypoints[..., 1][mask] -= top
            keypoints = _update_visibility(keypoints, new_w, new_h)
            target['keypoints'] = keypoints

        return cropped_image, target, dataset

    def _sample_crop(
        self,
        boxes_xyxy: torch.Tensor,
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int, int, int, torch.Tensor]]:
        if boxes_xyxy.numel() == 0:
            return None

        for min_jaccard in random.sample(self.options, len(self.options)):
            if min_jaccard >= 1.0:
                return None
            for _ in range(self.trials):
                r = torch.rand(2)
                new_w = int(width * (self.min_scale + (self.max_scale - self.min_scale) * r[0]))
                new_h = int(height * (self.min_scale + (self.max_scale - self.min_scale) * r[1]))
                if new_w <= 0 or new_h <= 0:
                    continue
                aspect = new_w / new_h
                if not (self.min_aspect_ratio <= aspect <= self.max_aspect_ratio):
                    continue
                left = int((width - new_w) * random.random())
                top = int((height - new_h) * random.random())
                right = left + new_w
                bottom = top + new_h

                cx = 0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2])
                cy = 0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3])
                keep_mask = (cx > left) & (cx < right) & (cy > top) & (cy < bottom)
                if not keep_mask.any():
                    continue

                ious = box_iou(
                    boxes_xyxy[keep_mask],
                    torch.tensor([[left, top, right, bottom]], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device),
                )
                if ious.max() < min_jaccard:
                    continue

                return left, top, new_w, new_h, keep_mask
        return None
