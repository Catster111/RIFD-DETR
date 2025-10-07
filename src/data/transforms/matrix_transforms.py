"""
Transform Matrix-based augmentations for unified keypoint and bbox handling
ระบบ transform matrix สำหรับจัดการ keypoint และ bbox อย่างสอดคล้องกัน
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Dict, Any, List
import torchvision.transforms.functional as F
import torchvision.tv_tensors as tv_tensors
import random
import math

class TransformMatrix:
    """Base class for matrix-based transformations"""
    
    def __init__(self):
        self.matrix = np.eye(3, dtype=np.float32)  # Identity matrix
        self.inverse_matrix = np.eye(3, dtype=np.float32)
    
    def __call__(self, image, target, dataset_ref=None):
        """Apply transformation to image, bbox, and keypoints"""
        
        # Get original image size
        if isinstance(image, Image.Image):
            orig_w, orig_h = image.size
        else:
            orig_h, orig_w = image.shape[-2:]
        
        # Apply transformation to image
        transformed_image = self.transform_image(image)
        
        # Get new image size
        if isinstance(transformed_image, Image.Image):
            new_w, new_h = transformed_image.size
        else:
            new_h, new_w = transformed_image.shape[-2:]
        
        # Apply transformation to target
        transformed_target = self.transform_target(target, orig_w, orig_h, new_w, new_h)
        
        return transformed_image, transformed_target, dataset_ref
    
    def transform_image(self, image):
        """Override in subclasses"""
        return image
    
    def transform_target(self, target, orig_w, orig_h, new_w, new_h):
        """Transform bbox and keypoints using the transformation matrix"""
        
        transformed_target = target.copy()
        
        # Transform bounding boxes
        if 'boxes' in target and target['boxes'].numel() > 0:
            transformed_target['boxes'] = self.transform_boxes(
                target['boxes'], orig_w, orig_h, new_w, new_h
            )
        
        # Transform keypoints
        if 'keypoints' in target and target['keypoints'].numel() > 0:
            transformed_target['keypoints'] = self.transform_keypoints(
                target['keypoints'], orig_w, orig_h, new_w, new_h
            )
        
        # Update size information
        transformed_target['orig_size'] = torch.tensor([orig_h, orig_w])
        transformed_target['size'] = torch.tensor([new_h, new_w])
        
        return transformed_target
    
    def transform_boxes(self, boxes, orig_w, orig_h, new_w, new_h):
        """Transform bounding boxes using matrix"""
        
        if boxes.numel() == 0:
            return boxes
        
        # Convert to numpy
        boxes_np = boxes.clone().float().numpy()
        
        # Check if boxes are in TV tensor format
        if hasattr(boxes, 'format'):
            if 'XYXY' in str(boxes.format):
                # XYXY format
                transformed_boxes = []
                for box in boxes_np:
                    x1, y1, x2, y2 = box
                    
                    # Transform corners
                    corners = np.array([
                        [x1, y1, 1],
                        [x2, y1, 1],
                        [x1, y2, 1],
                        [x2, y2, 1]
                    ]).T
                    
                    transformed_corners = self.matrix @ corners
                    
                    # Get new bounding box
                    x_coords = transformed_corners[0, :]
                    y_coords = transformed_corners[1, :]
                    
                    new_x1, new_x2 = x_coords.min(), x_coords.max()
                    new_y1, new_y2 = y_coords.min(), y_coords.max()
                    
                    # Clamp to image bounds
                    new_x1 = max(0, min(new_w, new_x1))
                    new_y1 = max(0, min(new_h, new_y1))
                    new_x2 = max(0, min(new_w, new_x2))
                    new_y2 = max(0, min(new_h, new_y2))
                    
                    transformed_boxes.append([new_x1, new_y1, new_x2, new_y2])
                
                # Convert back to tensor
                result = torch.tensor(transformed_boxes, dtype=boxes.dtype)
                
                # Recreate TV tensor with new canvas size
                return tv_tensors.BoundingBoxes(
                    result,
                    format=boxes.format,
                    canvas_size=(new_h, new_w)
                )
        
        # Fallback: assume CXCYWH format
        transformed_boxes = []
        for box in boxes_np:
            cx, cy, w, h = box
            
            # Convert to corners
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            
            # Transform corners
            corners = np.array([
                [x1, y1, 1],
                [x2, y1, 1],
                [x1, y2, 1],
                [x2, y2, 1]
            ]).T
            
            transformed_corners = self.matrix @ corners
            
            # Get new bounding box
            x_coords = transformed_corners[0, :]
            y_coords = transformed_corners[1, :]
            
            new_x1, new_x2 = x_coords.min(), x_coords.max()
            new_y1, new_y2 = y_coords.min(), y_coords.max()
            
            # Convert back to CXCYWH
            new_cx = (new_x1 + new_x2) / 2
            new_cy = (new_y1 + new_y2) / 2
            new_w = new_x2 - new_x1
            new_h = new_y2 - new_y1
            
            transformed_boxes.append([new_cx, new_cy, new_w, new_h])
        
        return torch.tensor(transformed_boxes, dtype=boxes.dtype)
    
    def transform_keypoints(self, keypoints, orig_w, orig_h, new_w, new_h):
        """Transform keypoints using matrix"""
        
        if keypoints.numel() == 0:
            return keypoints
        
        transformed_keypoints = keypoints.clone()
        
        for inst_idx in range(keypoints.shape[0]):
            for kp_idx in range(keypoints.shape[1]):
                x, y, v = keypoints[inst_idx, kp_idx]
                
                if v > 0:  # Only transform visible keypoints
                    # Transform point
                    point = np.array([float(x), float(y), 1])
                    transformed_point = self.matrix @ point
                    
                    new_x = transformed_point[0]
                    new_y = transformed_point[1]
                    
                    # Check if point is still within image bounds
                    if 0 <= new_x <= new_w and 0 <= new_y <= new_h:
                        transformed_keypoints[inst_idx, kp_idx, 0] = new_x
                        transformed_keypoints[inst_idx, kp_idx, 1] = new_y
                        # Keep visibility unchanged
                    else:
                        # Mark as invisible if outside bounds
                        transformed_keypoints[inst_idx, kp_idx, 2] = 0
        
        return transformed_keypoints

class MatrixResize(TransformMatrix):
    """Matrix-based resize transform"""
    
    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
    
    def transform_image(self, image):
        """Resize image"""
        if isinstance(image, Image.Image):
            return image.resize(self.size, Image.BILINEAR)
        else:
            return F.resize(image, self.size)
    
    def transform_target(self, target, orig_w, orig_h, new_w, new_h):
        """Set up scaling matrix and transform"""
        
        # Calculate scaling factors
        scale_x = new_w / orig_w
        scale_y = new_h / orig_h
        
        # Create scaling matrix
        self.matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return super().transform_target(target, orig_w, orig_h, new_w, new_h)

class MatrixHorizontalFlip(TransformMatrix):
    """Matrix-based horizontal flip transform"""
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.should_flip = False
    
    def __call__(self, image, target, dataset_ref=None):
        self.should_flip = random.random() < self.p
        if not self.should_flip:
            return image, target, dataset_ref
        
        return super().__call__(image, target, dataset_ref)
    
    def transform_image(self, image):
        """Flip image horizontally"""
        if self.should_flip:
            if isinstance(image, Image.Image):
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return F.hflip(image)
        return image
    
    def transform_target(self, target, orig_w, orig_h, new_w, new_h):
        """Set up flip matrix and transform"""
        
        if self.should_flip:
            # Create horizontal flip matrix
            self.matrix = np.array([
                [-1, 0, orig_w],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.matrix = np.eye(3, dtype=np.float32)
        
        return super().transform_target(target, orig_w, orig_h, new_w, new_h)

class MatrixRotation(TransformMatrix):
    """Matrix-based rotation transform"""
    
    def __init__(self, degrees, p=0.5):
        super().__init__()
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.p = p
        self.angle = 0
        self.should_rotate = False
    
    def __call__(self, image, target, dataset_ref=None):
        self.should_rotate = random.random() < self.p
        if self.should_rotate:
            self.angle = random.uniform(self.degrees[0], self.degrees[1])
        else:
            self.angle = 0
        
        return super().__call__(image, target, dataset_ref)
    
    def transform_image(self, image):
        """Rotate image"""
        if self.should_rotate:
            if isinstance(image, Image.Image):
                return image.rotate(self.angle, expand=True, fillcolor=0)
            else:
                return F.rotate(image, self.angle, expand=True, fill=0)
        return image
    
    def transform_target(self, target, orig_w, orig_h, new_w, new_h):
        """Set up rotation matrix and transform"""
        
        if self.should_rotate:
            # Convert angle to radians
            angle_rad = math.radians(self.angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            # Calculate center
            cx, cy = orig_w / 2, orig_h / 2
            
            # Rotation matrix around center
            self.matrix = np.array([
                [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy],
                [sin_a, cos_a, cy - sin_a * cx - cos_a * cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Adjust for image expansion
            if new_w != orig_w or new_h != orig_h:
                # Add translation to center in new image
                offset_x = (new_w - orig_w) / 2
                offset_y = (new_h - orig_h) / 2
                
                translation = np.array([
                    [1, 0, offset_x],
                    [0, 1, offset_y],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                self.matrix = translation @ self.matrix
        else:
            self.matrix = np.eye(3, dtype=np.float32)
        
        return super().transform_target(target, orig_w, orig_h, new_w, new_h)

class MatrixCrop(TransformMatrix):
    """Matrix-based crop transform"""
    
    def __init__(self, crop_box):
        super().__init__()
        self.crop_box = crop_box  # (x1, y1, x2, y2)
    
    def transform_image(self, image):
        """Crop image"""
        x1, y1, x2, y2 = self.crop_box
        if isinstance(image, Image.Image):
            return image.crop((x1, y1, x2, y2))
        else:
            return F.crop(image, y1, x1, y2 - y1, x2 - x1)
    
    def transform_target(self, target, orig_w, orig_h, new_w, new_h):
        """Set up crop matrix and transform"""
        
        x1, y1, x2, y2 = self.crop_box
        
        # Create translation matrix for crop
        self.matrix = np.array([
            [1, 0, -x1],
            [0, 1, -y1],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return super().transform_target(target, orig_w, orig_h, new_w, new_h)

# Compose multiple matrix transforms
class MatrixCompose:
    """Compose multiple matrix-based transforms"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target, dataset_ref=None):
        for transform in self.transforms:
            image, target, dataset_ref = transform(image, target, dataset_ref)
        return image, target, dataset_ref