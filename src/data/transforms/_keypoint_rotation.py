"""
Custom rotation transform that keeps bounding boxes and keypoints aligned.
"""

import math
import random
from typing import Any, Tuple

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors
from torchvision.ops import box_convert

from ...core import register


@register()
class RandomRotationWithKeypoints(T.Transform):
    """Rotate image, bounding boxes, and keypoints around the image centre."""

    def __init__(self, degrees=15, fill=0, p=0.5):
        super().__init__()
        self.fill = fill
        self.p = p
        if isinstance(degrees, (int, float)):
            self.angle_range = (-degrees, degrees)
        else:
            self.angle_range = degrees

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]

        image, target, dataset = inputs

        if random.random() > self.p:
            if target is not None and 'rotation_angle' not in target:
                device = None
                if isinstance(target.get('boxes'), torch.Tensor):
                    device = target['boxes'].device
                elif isinstance(target.get('keypoints'), torch.Tensor):
                    device = target['keypoints'].device
                device = device or torch.device('cpu')
                target['rotation_angle'] = torch.tensor(0.0, device=device)
            return image, target, dataset

        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        if hasattr(image, "size"):
            img_width, img_height = image.size
        else:
            _, img_height, img_width = image.shape

        rotated_image = F.rotate(image, angle, fill=self.fill)

        if target is not None:
            target = self._transform_target(target, angle, img_width, img_height)

        return rotated_image, target, dataset

    def _transform_target(self, target: dict, angle_degrees: float, img_width: int, img_height: int) -> dict:
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        center_x = img_width / 2
        center_y = img_height / 2

        rotated_keypoints = None
        if "keypoints" in target:
            keypoints = target["keypoints"]
            if isinstance(keypoints, torch.Tensor) and keypoints.numel() > 0:
                rotated_keypoints = self._rotate_keypoints(
                    keypoints,
                    cos_a,
                    sin_a,
                    center_x,
                    center_y,
                    img_width,
                    img_height,
                )
                target["keypoints"] = rotated_keypoints

        if "boxes" in target:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                target["boxes"] = self._rotate_boxes(
                    boxes,
                    cos_a,
                    sin_a,
                    center_x,
                    center_y,
                    img_width,
                    img_height,
                    rotated_keypoints,
                )

        device = None
        if isinstance(target.get('boxes'), torch.Tensor):
            device = target['boxes'].device
        elif isinstance(target.get('keypoints'), torch.Tensor):
            device = target['keypoints'].device
        device = device or torch.device('cpu')
        target['rotation_angle'] = torch.tensor(angle_degrees, device=device, dtype=torch.float32)

        return target

    def _prepare_boxes(
        self,
        boxes: torch.Tensor,
        img_width: int,
        img_height: int,
    ) -> Tuple[torch.Tensor, str, Tuple[int, int]]:
        canvas_size = (img_height, img_width)
        if isinstance(boxes, tv_tensors.BoundingBoxes):
            fmt = boxes.format.value.lower()
            canvas_attr = getattr(boxes, "canvas_size", None) or getattr(boxes, "spatial_size", None)
            if canvas_attr is not None:
                canvas_size = tuple(canvas_attr)
            boxes_tensor = boxes.as_subclass(torch.Tensor).clone()
            return boxes_tensor, fmt, canvas_size

        boxes_tensor = boxes.clone().detach()
        return boxes_tensor, "xyxy", canvas_size

    def _boxes_to_corners(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
        return torch.stack(
            [
                torch.stack([x1, y1], dim=1),
                torch.stack([x2, y1], dim=1),
                torch.stack([x2, y2], dim=1),
                torch.stack([x1, y2], dim=1),
            ],
            dim=1,
        )

    def _rotate_boxes(
        self,
        boxes: torch.Tensor,
        cos_a: float,
        sin_a: float,
        center_x: float,
        center_y: float,
        img_width: int,
        img_height: int,
        keypoints: torch.Tensor | None = None,
    ) -> tv_tensors.BoundingBoxes:
        boxes_tensor, fmt, canvas_size = self._prepare_boxes(boxes, img_width, img_height)
        if boxes_tensor.numel() == 0:
            return tv_tensors.BoundingBoxes(
                boxes_tensor,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=canvas_size,
            )

        if fmt != "xyxy":
            boxes_xyxy = box_convert(boxes_tensor, in_fmt=fmt, out_fmt="xyxy")
        else:
            boxes_xyxy = boxes_tensor

        # Use image-coordinate rotation (y-axis points down), matching torchvision F.rotate
        rotation = torch.tensor(
            [[cos_a, sin_a], [-sin_a, cos_a]],
            device=boxes_xyxy.device,
            dtype=boxes_xyxy.dtype,
        )
        center = torch.tensor([center_x, center_y], device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)

        corners = self._boxes_to_corners(boxes_xyxy)
        rotated = (corners - center) @ rotation.T + center

        padding = torch.tensor(5.0, dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
        x_min = (rotated[..., 0].min(dim=1).values - padding).clamp_(0, img_width)
        y_min = (rotated[..., 1].min(dim=1).values - padding).clamp_(0, img_height)
        x_max = (rotated[..., 0].max(dim=1).values + padding).clamp_(0, img_width)
        y_max = (rotated[..., 1].max(dim=1).values + padding).clamp_(0, img_height)

        rotated_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

        if keypoints is not None and keypoints.numel() > 0:
            kp = keypoints
            if kp.shape[0] == rotated_boxes.shape[0]:
                vis = kp[..., 2] > 0
                if vis.any():
                    kp_x = kp[..., 0]
                    kp_y = kp[..., 1]
                    large = torch.full_like(kp_x, float('inf'))
                    neg_large = torch.full_like(kp_x, float('-inf'))

                    kp_x_min = torch.where(vis, kp_x, large).min(dim=1).values
                    kp_y_min = torch.where(vis, kp_y, large).min(dim=1).values
                    kp_x_max = torch.where(vis, kp_x, neg_large).max(dim=1).values
                    kp_y_max = torch.where(vis, kp_y, neg_large).max(dim=1).values

                    kp_has_vis = vis.any(dim=1)

                    rotated_boxes[:, 0] = torch.where(kp_has_vis, torch.minimum(rotated_boxes[:, 0], kp_x_min), rotated_boxes[:, 0])
                    rotated_boxes[:, 1] = torch.where(kp_has_vis, torch.minimum(rotated_boxes[:, 1], kp_y_min), rotated_boxes[:, 1])
                    rotated_boxes[:, 2] = torch.where(kp_has_vis, torch.maximum(rotated_boxes[:, 2], kp_x_max), rotated_boxes[:, 2])
                    rotated_boxes[:, 3] = torch.where(kp_has_vis, torch.maximum(rotated_boxes[:, 3], kp_y_max), rotated_boxes[:, 3])

        return tv_tensors.BoundingBoxes(
            rotated_boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=canvas_size,
        )

    def _rotate_keypoints(
        self,
        keypoints: torch.Tensor,
        cos_a: float,
        sin_a: float,
        center_x: float,
        center_y: float,
        img_width: int,
        img_height: int,
    ) -> torch.Tensor:
        if keypoints.dim() != 3 or keypoints.numel() == 0:
            return keypoints

        xy = keypoints[..., :2]
        rotation = torch.tensor(
            [[cos_a, sin_a], [-sin_a, cos_a]],
            dtype=xy.dtype,
            device=xy.device,
        )
        center = torch.tensor([center_x, center_y], dtype=xy.dtype, device=xy.device)

        rotated_xy = (xy - center) @ rotation.T + center

        rotated = keypoints.clone()
        rotated[..., 0:2] = rotated_xy

        if rotated.shape[-1] > 2:
            visibility = rotated[..., 2]
            out_of_bounds = (
                (rotated[..., 0] < 0)
                | (rotated[..., 0] >= img_width)
                | (rotated[..., 1] < 0)
                | (rotated[..., 1] >= img_height)
            )
            rotated[..., 2] = torch.where(out_of_bounds, torch.zeros_like(visibility), visibility)

        rotated[..., 0].clamp_(0, img_width)
        rotated[..., 1].clamp_(0, img_height)

        return rotated

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(degrees={self.angle_range}, fill={self.fill}, p={self.p})"
        )


RandomRotationKeypoints = RandomRotationWithKeypoints
