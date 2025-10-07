"""
Simplified WiderFace dataset with direct RT-DETR format support and CorrectedAugmentation
WiderFace dataset ที่ปรับให้ใช้ RT-DETR format ตั้งแต่แรก พร้อม augmentation ที่ถูกต้อง
"""

import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T
from collections import defaultdict
import random
import math
import numpy as np

from ..core import register
from ._misc import convert_to_tv_tensor

# Import Albumentations augmentation
import albumentations as A

class AlbumentationsAugmentation:
    """Professional augmentation using Albumentations library - eliminates coordinate drift"""
    
    def __init__(self, flip_prob=0.5, color_prob=0.8, rotation_prob=0.7, max_rotation=15):
        self.flip_prob = flip_prob
        self.color_prob = color_prob
        self.rotation_prob = rotation_prob
        self.max_rotation = max_rotation
        
        # Define professional augmentation pipeline
        self.transform = A.Compose([
            # Geometric augmentations (affect coordinates)
            A.Rotate(
                limit=max_rotation,  # ±max_rotation degrees
                p=rotation_prob,
                interpolation=1,  # cv2.INTER_LINEAR
                border_mode=0,    # cv2.BORDER_CONSTANT
                crop_border=False  # Don't crop rotated image
            ),
            A.HorizontalFlip(p=flip_prob),
            
            # Color augmentations (don't affect coordinates)
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=1.0),
                A.ColorJitter(brightness=0.6, contrast=0.3, saturation=0.5, hue=0.1, p=1.0),
                A.ColorJitter(brightness=0.3, contrast=0.6, saturation=0.3, hue=0.15, p=1.0),
            ], p=color_prob),
        ], 
        # Define bbox parameters - use YOLO format (normalized cxcywh)
        bbox_params=A.BboxParams(
            format='yolo',  # YOLO format = normalized cxcywh
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        ),
        # Define keypoint parameters - normalized XY format
        keypoint_params=A.KeypointParams(
            format='xy',
            label_fields=['keypoint_labels']
        ))
    
    def __call__(self, image, target):
        """Apply professional augmentation with accurate coordinate handling"""
        
        # Convert tensor to numpy if needed
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
        else:
            # PIL to numpy
            image_np = np.array(image)
        
        # Extract data from target
        boxes = target['boxes'].clone()
        keypoints = target['keypoints'].clone()
        labels = target['labels'].clone()
        
        # Convert to lists for Albumentations with bounds checking
        boxes_list = []
        for box in boxes:
            cx, cy, w, h = box.tolist()
            # Convert from CXCYWH to YOLO format and clamp to [0,1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))  
            w = max(0.01, min(1.0, w))   # Minimum width 0.01
            h = max(0.01, min(1.0, h))   # Minimum height 0.01
            boxes_list.append([cx, cy, w, h])
        
        labels_list = labels.tolist()
        
        # Convert keypoints to Albumentations format (pixel coordinates)
        keypoints_list = []
        keypoint_labels = []
        
        img_height, img_width = image_np.shape[:2]
        
        for face_idx, face_kps in enumerate(keypoints):
            for kp_idx, (x, y, vis) in enumerate(face_kps):
                if vis > 0:  # Only include visible keypoints
                    # Convert from normalized [0,1] to pixel coordinates
                    pixel_x = x.item() * img_width
                    pixel_y = y.item() * img_height
                    
                    # CRITICAL FIX: Clamp coordinates to valid image bounds
                    pixel_x = max(0.0, min(pixel_x, img_width - 1.0))
                    pixel_y = max(0.0, min(pixel_y, img_height - 1.0))
                    
                    keypoints_list.append([pixel_x, pixel_y])
                    keypoint_labels.append(f"face_{face_idx}_kp_{kp_idx}")
        
        try:
            # Apply augmentation
            augmented = self.transform(
                image=image_np,
                bboxes=boxes_list,
                labels=labels_list,
                keypoints=keypoints_list,
                keypoint_labels=keypoint_labels
            )
            
            # Extract augmented data
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['labels']
            aug_keypoints = augmented['keypoints']
            aug_keypoint_labels = augmented['keypoint_labels']
            
        except Exception as e:
            print(f"Albumentations augmentation failed: {e}")
            # Fallback to original image/target with cleaned coordinates
            aug_image = image_np
            aug_bboxes = boxes_list
            aug_labels = labels_list
            
            # CRITICAL FIX: Clean keypoints before fallback
            cleaned_keypoints = []
            cleaned_labels = []
            for kp_coords, kp_label in zip(keypoints_list, keypoint_labels):
                # Ensure coordinates are within bounds and not NaN/inf
                x, y = kp_coords
                if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                    x = max(0.0, min(x, img_width - 1.0))
                    y = max(0.0, min(y, img_height - 1.0))
                    cleaned_keypoints.append([x, y])
                    cleaned_labels.append(kp_label)
            
            aug_keypoints = cleaned_keypoints
            aug_keypoint_labels = cleaned_labels
        
        # Convert back to tensors
        import torchvision.transforms.functional as TF
        
        # Image: numpy -> tensor
        aug_image_tensor = TF.to_tensor(aug_image)
        
        # Boxes: list -> tensor
        aug_boxes_tensor = torch.tensor(aug_bboxes, dtype=torch.float32) if aug_bboxes else torch.empty((0, 4))
        
        # Labels: list -> tensor  
        aug_labels_tensor = torch.tensor(aug_labels, dtype=torch.int64) if aug_labels else torch.empty((0,), dtype=torch.int64)
        
        # Keypoints: reconstruct from flat list
        aug_keypoints_tensor = torch.zeros_like(keypoints)
        
        if aug_keypoints and aug_keypoint_labels:
            # Map keypoints back to original structure
            for kp_coords, kp_label in zip(aug_keypoints, aug_keypoint_labels):
                # Parse label: "face_{face_idx}_kp_{kp_idx}"
                parts = kp_label.split('_')
                face_idx = int(parts[1])
                kp_idx = int(parts[3])
                
                if face_idx < aug_keypoints_tensor.shape[0] and kp_idx < aug_keypoints_tensor.shape[1]:
                    # Get original visibility
                    orig_vis = keypoints[face_idx, kp_idx, 2]
                    
                    # Albumentations returns keypoints in pixel coordinates
                    # Need to convert back to normalized [0,1] coordinates
                    img_height, img_width = image_np.shape[:2]
                    
                    x_norm = kp_coords[0] / img_width
                    y_norm = kp_coords[1] / img_height
                    
                    # CRITICAL FIX: Check for NaN/inf before normalization
                    if np.isnan(x_norm) or np.isnan(y_norm) or np.isinf(x_norm) or np.isinf(y_norm):
                        # Set to invisible if coordinates are invalid
                        x_norm, y_norm, orig_vis = 0.0, 0.0, 0.0
                    else:
                        # Ensure coordinates are in [0, 1] range
                        x_norm = max(0.0, min(1.0, x_norm))
                        y_norm = max(0.0, min(1.0, y_norm))
                    
                    aug_keypoints_tensor[face_idx, kp_idx] = torch.tensor([
                        x_norm, y_norm, orig_vis
                    ])
        
        # Create new target with augmented coordinates
        new_target = {}
        for key, value in target.items():
            if key == 'boxes':
                new_target[key] = aug_boxes_tensor
            elif key == 'keypoints':
                new_target[key] = aug_keypoints_tensor
            elif key == 'labels':
                new_target[key] = aug_labels_tensor
            else:
                # Copy other fields unchanged
                new_target[key] = value.clone() if hasattr(value, 'clone') else value
        
        return aug_image_tensor, new_target
    


@register()
class WiderFaceKeypointDatasetSimplified(Dataset):
    __inject__ = ['transforms', ]
    
    def __init__(self, ann_file, img_prefix, transforms=None, return_masks=False, **kwargs):
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        self.img_prefix = img_prefix
        self._transforms = transforms
        self.epoch = 0  # Initialize epoch for transform policies
        self.image_dict = {img['id']: img for img in self.coco['images']}
        
        # Group annotations by image_id for multi-instance images
        self.img_ann_dict = defaultdict(list)
        for ann in self.coco['annotations']:
            self.img_ann_dict[ann['image_id']].append(ann)
        
        self.image_ids = list(self.img_ann_dict.keys())
        
        # ALBUMENTATIONS: Professional augmentation with zero coordinate drift
        # Create Albumentations augmentation if no transforms provided
        if self._transforms is None:
            self._corrected_augmentation = AlbumentationsAugmentation(
                flip_prob=0.5,      # 50% horizontal flip
                color_prob=0.8,     # 80% color augmentation  
                rotation_prob=0.7,  # 70% rotation
                max_rotation=15     # ±15° rotation
            )
        else:
            self._corrected_augmentation = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_meta = self.image_dict[image_id]
        annotations = self.img_ann_dict[image_id]
        
        # Handle WiderFace path format: remove "WIDER_train/images/" prefix
        file_name = img_meta['file_name']
        if file_name.startswith('WIDER_train/images/'):
            file_name = file_name.replace('WIDER_train/images/', '')
        img_path = os.path.join(self.img_prefix, file_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return empty data if image can't be loaded
            return torch.zeros(3, 640, 640), {
                'boxes': torch.empty((0, 4)),
                'labels': torch.empty((0,), dtype=torch.int64),
                'keypoints': torch.empty((0, 5, 3)),
                'image_id': torch.tensor([image_id]),
                'idx': torch.tensor([idx])
            }
        
        orig_w, orig_h = image.size
        
        boxes = []
        labels = []
        keypoints_list = []
        areas = []
        
        for ann in annotations:
            # Convert bbox directly from COCO [x, y, w, h] to RT-DETR [cx, cy, w, h] normalized
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to center coordinates and normalize by image size
            cx = (x + w/2) / orig_w
            cy = (y + h/2) / orig_h 
            w_norm = w / orig_w
            h_norm = h / orig_h
            
            boxes.append([cx, cy, w_norm, h_norm])
            
            # Process keypoints if available
            if 'keypoints' in ann and ann['keypoints']:
                kpts = ann['keypoints']
                kpts_normalized = []
                
                # Process each keypoint (5 keypoints expected)
                for i in range(0, min(len(kpts), 15), 3):  # x, y, visibility for each keypoint
                    kp_x = kpts[i] / orig_w      # Normalize x
                    kp_y = kpts[i + 1] / orig_h  # Normalize y
                    kp_vis = kpts[i + 2] if i + 2 < len(kpts) else 2  # Visibility
                    kpts_normalized.append([kp_x, kp_y, kp_vis])
                
                # Ensure we have exactly 5 keypoints (pad with zeros if needed)
                while len(kpts_normalized) < 5:
                    kpts_normalized.append([0.0, 0.0, 0])  # Invisible keypoint
                
                keypoints_list.append(torch.tensor(kpts_normalized[:5], dtype=torch.float32))
            else:
                # No keypoints provided, create default invisible keypoints
                keypoints_list.append(torch.zeros(5, 3, dtype=torch.float32))
            
            labels.append(0)  # Face class = 0 (for training consistency)
            areas.append(ann['area'] / (orig_w * orig_h))  # Normalize area too
        
        # Create tensors in RT-DETR format
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        
        target = {
            'boxes': boxes_tensor,  # Already in CXCYWH normalized format
            'labels': torch.tensor(labels, dtype=torch.int64),
            'keypoints': torch.stack(keypoints_list) if keypoints_list else torch.empty((0, 5, 3)),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'size': torch.tensor([orig_w, orig_h]),  # W, H format for consistency
            'idx': torch.tensor([idx])
        }

        # Apply transforms
        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self)
        elif self._corrected_augmentation is not None:
            # Use corrected augmentation
            image, target = self._corrected_augmentation(image, target)
        else:
            # Apply basic resize to 640x640 if no transforms
            basic_transform = T.Compose([
                T.Resize((640, 640)),
                T.ToTensor()
            ])
            image = basic_transform(image)

        return image, target
    
    def load_item(self, idx):
        """Load item for evaluation (required by COCO evaluator) - returns raw image and target"""
        image_id = self.image_ids[idx]
        img_meta = self.image_dict[image_id]
        annotations = self.img_ann_dict[image_id]
        
        # Handle WiderFace path format: remove "WIDER_train/images/" prefix
        file_name = img_meta['file_name']
        if file_name.startswith('WIDER_train/images/'):
            file_name = file_name.replace('WIDER_train/images/', '')
        img_path = os.path.join(self.img_prefix, file_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create dummy image for evaluation
            image = Image.new('RGB', (640, 640), (0, 0, 0))
        
        orig_w, orig_h = image.size
        
        boxes = []
        labels = []
        keypoints_list = []
        areas = []
        
        for ann in annotations:
            # Convert bbox directly from COCO [x, y, w, h] to RT-DETR [cx, cy, w, h] normalized
            bbox = ann['bbox']
            x, y, w, h = bbox
            
            # Convert to center coordinates and normalize by image size
            cx = (x + w/2) / orig_w
            cy = (y + h/2) / orig_h 
            w_norm = w / orig_w
            h_norm = h / orig_h
            
            boxes.append([cx, cy, w_norm, h_norm])
            
            # Process keypoints if available
            if 'keypoints' in ann and ann['keypoints']:
                kpts = ann['keypoints']
                kpts_normalized = []
                
                # Process each keypoint (5 keypoints expected)
                for i in range(0, min(len(kpts), 15), 3):  # x, y, visibility for each keypoint
                    kp_x = kpts[i] / orig_w      # Normalize x
                    kp_y = kpts[i + 1] / orig_h  # Normalize y
                    kp_vis = kpts[i + 2] if i + 2 < len(kpts) else 2  # Visibility
                    kpts_normalized.append([kp_x, kp_y, kp_vis])
                
                # Ensure we have exactly 5 keypoints (pad with zeros if needed)
                while len(kpts_normalized) < 5:
                    kpts_normalized.append([0.0, 0.0, 0])  # Invisible keypoint
                
                keypoints_list.append(torch.tensor(kpts_normalized[:5], dtype=torch.float32))
            else:
                # No keypoints provided, create default invisible keypoints
                keypoints_list.append(torch.zeros(5, 3, dtype=torch.float32))
            
            labels.append(0)  # Face class = 0 (for training consistency)
            areas.append(ann['area'] / (orig_w * orig_h))  # Normalize area too
        
        # Create tensors in RT-DETR format
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        
        target = {
            'boxes': boxes_tensor,  # Already in CXCYWH normalized format
            'labels': torch.tensor(labels, dtype=torch.int64),
            'keypoints': torch.stack(keypoints_list) if keypoints_list else torch.empty((0, 5, 3)),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'size': torch.tensor([orig_w, orig_h]),  # W, H format for consistency
            'idx': torch.tensor([idx])
        }

        return image, target
    
    def set_epoch(self, epoch):
        """Set epoch for distributed training and transform policies"""
        self.epoch = epoch
