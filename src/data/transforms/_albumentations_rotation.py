"""
Albumentations-based rotation transform for RT-DETRv2 keypoint detection
ใช้ Albumentations library สำหรับ rotation augmentation ที่รองรับ keypoints
"""

import torch
import numpy as np
import albumentations as A
from typing import Any, Dict, Tuple
import torchvision.tv_tensors as tv_tensors
from PIL import Image

from ...core import register
import torch.nn as nn

@register()
class AlbumentationsRotation(nn.Module):
    """
    Albumentations-based rotation transform that properly handles keypoints
    รองรับ rotation ทั้ง bounding boxes และ keypoints อย่างถูกต้อง
    """
    
    def __init__(self, degrees=15, p=0.5, fill=0, interpolation=1, border_mode=0):
        super().__init__()
        """
        Args:
            degrees (int): Rotation range in degrees (±degrees)
            p (float): Probability of applying rotation
            fill (int): Fill value for empty pixels after rotation
            interpolation (int): OpenCV interpolation method (1=LINEAR, 0=NEAREST)
            border_mode (int): OpenCV border mode (0=CONSTANT, 1=REFLECT, 2=WRAP)
        """
        self.degrees = degrees
        self.p = p
        self.fill = fill
        self.interpolation = interpolation
        self.border_mode = border_mode
        
        # Create Albumentations rotation transform
        self.rotation_transform = A.Compose([
            A.Rotate(
                limit=degrees,
                p=1.0,  # Always apply when called (probability handled externally)
                interpolation=interpolation,
                border_mode=border_mode,
                crop_border=False  # Keep original image size
            )
        ], 
        # Configure bbox and keypoint handling
        bbox_params=A.BboxParams(
            format='pascal_voc',  # XYXY format
            min_area=0,
            min_visibility=0,
            label_fields=['bbox_labels']
        ),
        keypoint_params=A.KeypointParams(
            format='xy',
            label_fields=['keypoint_labels'],
            remove_invisible=False
        ))
        
    def forward(self, sample):
        """Forward method for TorchVision v2 compatibility"""
        if isinstance(sample, tuple) and len(sample) >= 2:
            image, target = sample[0], sample[1]
            dataset_instance = sample[2] if len(sample) > 2 else None
        else:
            # Handle single input case
            return sample
        
        rotated_image, rotated_target, _ = self._apply_rotation(image, target, dataset_instance)
        
        if dataset_instance is not None:
            return (rotated_image, rotated_target, dataset_instance)
        else:
            return (rotated_image, rotated_target)
    
    def __call__(self, image, target, dataset_instance=None):
        """Direct call method for backwards compatibility"""
        return self._apply_rotation(image, target, dataset_instance)
    
    def _apply_rotation(self, image, target, dataset_instance=None):
        """
        Apply rotation transform using Albumentations
        
        Args:
            image: PIL Image or torch.Tensor
            target: Dictionary with 'boxes', 'keypoints', 'labels', etc.
            dataset_instance: Dataset instance (for compatibility)
            
        Returns:
            Tuple of (rotated_image, rotated_target, dataset_instance)
        """
        
        # Apply rotation with probability
        if np.random.random() > self.p:
            return image, target, dataset_instance
        
        try:
            # Convert image to numpy array if needed
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:  # CHW format
                    image_np = image.permute(1, 2, 0).numpy()
                else:
                    image_np = image.numpy()
                
                # Ensure proper data type and range
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            elif isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = np.array(image)
            
            img_height, img_width = image_np.shape[:2]
            
            # Prepare bounding boxes for Albumentations
            boxes_list = []
            bbox_labels = []
            
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                
                # Handle TV tensors
                if hasattr(boxes, 'data'):
                    boxes = boxes.data
                
                for i, box in enumerate(boxes):
                    if hasattr(boxes, 'format'):
                        # Already in XYXY format from TV tensors
                        x1, y1, x2, y2 = box.tolist()
                    else:
                        # Assume CXCYWH normalized and convert to XYXY pixel coordinates
                        cx, cy, w, h = box.tolist()
                        x1 = (cx - w/2) * img_width
                        y1 = (cy - h/2) * img_height
                        x2 = (cx + w/2) * img_width
                        y2 = (cy + h/2) * img_height
                    
                    # Clamp to image boundaries
                    x1 = max(0, min(x1, img_width - 1))
                    y1 = max(0, min(y1, img_height - 1))
                    x2 = max(x1 + 1, min(x2, img_width))
                    y2 = max(y1 + 1, min(y2, img_height))
                    
                    boxes_list.append([x1, y1, x2, y2])
                    bbox_labels.append(target['labels'][i].item() if i < len(target['labels']) else 0)
            
            # Prepare keypoints for Albumentations
            keypoints_list = []
            keypoint_labels = []
            
            if 'keypoints' in target and target['keypoints'].numel() > 0:
                keypoints = target['keypoints']
                
                for face_idx, face_kpts in enumerate(keypoints):
                    for kp_idx, (x, y, vis) in enumerate(face_kpts):
                        if vis > 0:  # Only include visible keypoints
                            # Convert coordinates to pixel values if needed
                            if x <= 1.0 and y <= 1.0:  # Normalized coordinates
                                pixel_x = x.item() * img_width
                                pixel_y = y.item() * img_height
                            else:  # Already pixel coordinates
                                pixel_x = x.item()
                                pixel_y = y.item()
                            
                            # Clamp to image boundaries
                            pixel_x = max(0, min(pixel_x, img_width - 1))
                            pixel_y = max(0, min(pixel_y, img_height - 1))
                            
                            keypoints_list.append([pixel_x, pixel_y])
                            keypoint_labels.append(f"face_{face_idx}_kp_{kp_idx}")
            
            # Apply Albumentations rotation
            augmented = self.rotation_transform(
                image=image_np,
                bboxes=boxes_list,
                bbox_labels=bbox_labels,
                keypoints=keypoints_list,
                keypoint_labels=keypoint_labels
            )
            
            # Extract rotated results
            rotated_image = augmented['image']
            rotated_bboxes = augmented['bboxes']
            rotated_bbox_labels = augmented['bbox_labels']
            rotated_keypoints = augmented['keypoints']
            rotated_keypoint_labels = augmented['keypoint_labels']
            
            # Convert rotated image back to PIL
            rotated_image_pil = Image.fromarray(rotated_image)
            
            # Reconstruct target with rotated coordinates
            new_target = {}
            
            # Copy unchanged fields
            for key in target:
                if key not in ['boxes', 'keypoints', 'labels']:
                    new_target[key] = target[key]
            
            # Reconstruct rotated boxes
            if rotated_bboxes:
                # Convert back to TV tensor format
                boxes_tensor = torch.tensor(rotated_bboxes, dtype=torch.float32)
                
                # Determine original format and canvas size
                original_boxes = target['boxes']
                if hasattr(original_boxes, 'format') and hasattr(original_boxes, 'canvas_size'):
                    # Preserve original TV tensor format
                    new_target['boxes'] = tv_tensors.BoundingBoxes(
                        boxes_tensor,
                        format=original_boxes.format,
                        canvas_size=original_boxes.canvas_size
                    )
                else:
                    new_target['boxes'] = boxes_tensor
                    
                # Reconstruct labels
                new_target['labels'] = torch.tensor(rotated_bbox_labels, dtype=torch.int64)
            else:
                # No boxes after rotation (might be cropped out)
                new_target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                new_target['labels'] = torch.empty((0,), dtype=torch.int64)
            
            # Reconstruct rotated keypoints
            if 'keypoints' in target:
                original_keypoints = target['keypoints']
                new_keypoints = torch.zeros_like(original_keypoints)
                
                if rotated_keypoints and rotated_keypoint_labels:
                    # Map rotated keypoints back to original structure
                    for kp_coords, kp_label in zip(rotated_keypoints, rotated_keypoint_labels):
                        # Parse label: "face_{face_idx}_kp_{kp_idx}"
                        parts = kp_label.split('_')
                        face_idx = int(parts[1])
                        kp_idx = int(parts[3])
                        
                        if face_idx < new_keypoints.shape[0] and kp_idx < new_keypoints.shape[1]:
                            # Get original visibility
                            orig_vis = original_keypoints[face_idx, kp_idx, 2]
                            
                            # Albumentations returns pixel coordinates, check if we need normalization
                            pixel_x, pixel_y = kp_coords
                            
                            # Store as pixel coordinates (will be normalized later by NormalizeKeypoints)
                            new_keypoints[face_idx, kp_idx] = torch.tensor([
                                pixel_x, pixel_y, orig_vis
                            ])
                
                new_target['keypoints'] = new_keypoints
            
            return rotated_image_pil, new_target, dataset_instance
            
        except Exception as e:
            print(f"⚠️ Albumentations rotation failed: {e}")
            # Return original data if rotation fails
            return image, target, dataset_instance
    
    def __repr__(self):
        return f"AlbumentationsRotation(degrees={self.degrees}, p={self.p})"


@register()
class AlbumentationsRotationSimple(nn.Module):
    """
    Fixed version of Albumentations rotation with accurate GT transformation
    เวอร์ชัน fixed ที่หมุนทั้งรูปและ GT อย่างถูกต้อง
    
    FIXED: Now properly rotates both image and GT coordinates using full AlbumentationsRotation
    """
    
    def __init__(self, degrees=15, p=0.5):
        super().__init__()
        self.degrees = degrees 
        self.p = p
        
        self.transform = A.Rotate(
            limit=degrees,
            p=1.0,  # Always apply when called
            interpolation=1,  # Linear interpolation
            border_mode=0,    # Constant border
            crop_border=False
        )
    
    def forward(self, sample):
        """Forward method for TorchVision v2 compatibility"""
        if isinstance(sample, tuple) and len(sample) >= 2:
            image, target = sample[0], sample[1]  
            dataset_instance = sample[2] if len(sample) > 2 else None
        else:
            # Handle single input case - return unchanged
            return sample
        
        # Apply rotation with probability
        if np.random.random() > self.p:
            return sample
        
        # Use the full AlbumentationsRotation implementation for accurate GT
        try:
            # Create temporary full rotation instance
            full_rotation = AlbumentationsRotation(
                degrees=self.degrees, 
                p=1.0,  # Always apply when called
                fill=0
            )
            
            # Apply full rotation (both image and GT)
            rotated_image, rotated_target, _ = full_rotation._apply_rotation(image, target, dataset_instance)
            
            if dataset_instance is not None:
                return (rotated_image, rotated_target, dataset_instance)
            else:
                return (rotated_image, rotated_target)
                
        except Exception as e:
            print(f"⚠️ Full rotation failed: {e}")
            return sample
    
    def __call__(self, image, target, dataset_instance=None):
        """Direct call method for backwards compatibility"""
        return self._apply_full_rotation(image, target, dataset_instance)
    
    def _apply_full_rotation(self, image, target, dataset_instance=None):
        """Apply full rotation with accurate GT transformation"""
        
        if np.random.random() > self.p:
            return image, target, dataset_instance
        
        try:
            # Use the full AlbumentationsRotation for accurate GT handling
            full_rotation = AlbumentationsRotation(
                degrees=self.degrees, 
                p=1.0,  # Always apply when called
                fill=0
            )
            
            # Apply full rotation (both image and GT)
            return full_rotation._apply_rotation(image, target, dataset_instance)
            
        except Exception as e:
            print(f"⚠️ Full rotation failed: {e}")
            return image, target, dataset_instance
    
    def __repr__(self):
        return f"AlbumentationsRotationSimple(degrees={self.degrees}, p={self.p})"