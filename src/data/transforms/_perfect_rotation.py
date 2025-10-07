"""
Perfect rotation transform with mathematical precision
การหมุนที่แม่นยำด้วยคณิตศาสตร์
"""

import torch
import numpy as np
import math
from typing import Any, Dict, Tuple
import torchvision.tv_tensors as tv_tensors
from PIL import Image

from ...core import register
import torch.nn as nn

@register()
class PerfectRotation(nn.Module):
    """
    Perfect mathematical rotation for both image and GT coordinates
    การหมุนที่สมบูรณ์แบบด้วยคณิตศาสตร์สำหรับทั้งรูปและ GT
    """
    
    def __init__(self, degrees=15, p=0.5, fill=128):
        super().__init__()
        self.degrees = degrees
        self.p = p
        self.fill = fill
    
    def forward(self, sample):
        """Forward method for TorchVision v2 compatibility"""
        if isinstance(sample, tuple) and len(sample) >= 2:
            image, target = sample[0], sample[1]
            dataset_instance = sample[2] if len(sample) > 2 else None
        else:
            return sample
        
        # Apply rotation with probability
        if np.random.random() > self.p:
            return sample
        
        # Generate random angle
        angle = np.random.uniform(-self.degrees, self.degrees)
        
        try:
            rotated_image, rotated_target = self._apply_perfect_rotation(image, target, angle)
            
            if dataset_instance is not None:
                return (rotated_image, rotated_target, dataset_instance)
            else:
                return (rotated_image, rotated_target)
                
        except Exception as e:
            print(f"⚠️ Perfect rotation failed: {e}")
            return sample
    
    def __call__(self, *args, **kwargs):
        """Call method with flexible argument handling"""
        # Handle both single sample tuple and separate arguments
        if len(args) == 1 and isinstance(args[0], (tuple, list)) and len(args[0]) >= 2:
            # Called with sample tuple: transform(sample)
            return self.forward(args[0])
        elif len(args) >= 2:
            # Called with separate arguments: transform(image, target, dataset_instance)
            image, target = args[0], args[1]
            dataset_instance = args[2] if len(args) > 2 else kwargs.get('dataset_instance', None)
            
            # Apply rotation with probability
            if np.random.random() > self.p:
                return image, target, dataset_instance
            
            # Generate random angle
            angle = np.random.uniform(-self.degrees, self.degrees)
            
            try:
                rotated_image, rotated_target = self._apply_perfect_rotation(image, target, angle)
                return rotated_image, rotated_target, dataset_instance
            except Exception as e:
                print(f"⚠️ Perfect rotation failed: {e}")
                return image, target, dataset_instance
        else:
            # Fallback to forward method
            return self.forward(args[0] if args else None)
    
    def _apply_perfect_rotation(self, image, target, angle):
        """Apply perfect mathematical rotation"""
        
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image")
        
        orig_w, orig_h = image.size
        center_x = orig_w / 2
        center_y = orig_h / 2
        
        # Step 1: Rotate image
        rotated_image = image.rotate(angle, expand=False, fillcolor=(self.fill, self.fill, self.fill))
        
        # Step 2: Calculate rotation matrix
        # CRITICAL FIX: PIL rotates clockwise for positive angles
        # But coordinate transformation needs counter-clockwise
        # So we negate the angle for coordinate calculations
        angle_rad = math.radians(-angle)  # 🔧 FIX: Negate angle
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        def rotate_point(x, y):
            """Rotate a single point around center"""
            # Translate to center
            px = x - center_x
            py = y - center_y
            
            # Apply rotation
            new_px = px * cos_a - py * sin_a
            new_py = px * sin_a + py * cos_a
            
            # Translate back
            new_x = new_px + center_x
            new_y = new_py + center_y
            
            # Clamp to image boundaries
            new_x = max(0, min(new_x, orig_w))
            new_y = max(0, min(new_y, orig_h))
            
            return new_x, new_y
        
        # Step 3: Create new target
        new_target = {}
        
        # Copy unchanged fields
        for key in target:
            if key not in ['boxes', 'keypoints']:
                new_target[key] = target[key]
        
        # Step 4: Rotate bounding boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            rotated_boxes = []
            original_boxes = target['boxes']
            
            # Handle TV tensors
            if hasattr(original_boxes, 'data'):
                boxes_data = original_boxes.data
            else:
                boxes_data = original_boxes
            
            for box in boxes_data:
                if hasattr(original_boxes, 'format'):
                    # XYXY format from TV tensors
                    x1, y1, x2, y2 = box.tolist()
                else:
                    # Assume CXCYWH normalized format
                    cx, cy, w, h = box.tolist()
                    x1 = (cx - w/2) * orig_w
                    y1 = (cy - h/2) * orig_h
                    x2 = (cx + w/2) * orig_w
                    y2 = (cy + h/2) * orig_h
                
                # Rotate all 4 corners of bounding box
                corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                rotated_corners = [rotate_point(x, y) for x, y in corners]
                
                # Find new bounding box from rotated corners
                xs = [corner[0] for corner in rotated_corners]
                ys = [corner[1] for corner in rotated_corners]
                new_x1, new_x2 = min(xs), max(xs)
                new_y1, new_y2 = min(ys), max(ys)
                
                rotated_boxes.append([new_x1, new_y1, new_x2, new_y2])
            
            # Convert back to original format
            if rotated_boxes:
                boxes_tensor = torch.tensor(rotated_boxes, dtype=torch.float32)
                
                if hasattr(original_boxes, 'format') and hasattr(original_boxes, 'canvas_size'):
                    # Preserve TV tensor format
                    new_target['boxes'] = tv_tensors.BoundingBoxes(
                        boxes_tensor,
                        format=original_boxes.format,
                        canvas_size=original_boxes.canvas_size
                    )
                else:
                    new_target['boxes'] = boxes_tensor
            else:
                new_target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
        
        # Step 5: Rotate keypoints
        if 'keypoints' in target and target['keypoints'].numel() > 0:
            original_keypoints = target['keypoints']
            new_keypoints = torch.zeros_like(original_keypoints)
            
            for face_idx in range(original_keypoints.shape[0]):
                for kp_idx in range(original_keypoints.shape[1]):
                    kp_x, kp_y, visibility = original_keypoints[face_idx, kp_idx]
                    
                    if visibility > 0:
                        # Convert to absolute coordinates if needed
                        if kp_x <= 1.0 and kp_y <= 1.0:  # Normalized coordinates
                            abs_x = kp_x.item() * orig_w
                            abs_y = kp_y.item() * orig_h
                        else:  # Already absolute
                            abs_x = kp_x.item()
                            abs_y = kp_y.item()
                        
                        # Rotate keypoint
                        new_x, new_y = rotate_point(abs_x, abs_y)
                        
                        # Store back (keep same format as input)
                        if kp_x <= 1.0 and kp_y <= 1.0:  # Was normalized
                            new_keypoints[face_idx, kp_idx] = torch.tensor([
                                new_x / orig_w, new_y / orig_h, visibility
                            ])
                        else:  # Was absolute
                            new_keypoints[face_idx, kp_idx] = torch.tensor([
                                new_x, new_y, visibility
                            ])
                    else:
                        new_keypoints[face_idx, kp_idx] = original_keypoints[face_idx, kp_idx]
            
            new_target['keypoints'] = new_keypoints
        
        return rotated_image, new_target
    
    def __repr__(self):
        return f"PerfectRotation(degrees={self.degrees}, p={self.p})"