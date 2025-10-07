#!/usr/bin/env python3

# Padding-Aware Mixed Augmentation for Model Retraining
# Combines distorted and padded preprocessing to teach model both conditions

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from typing import Dict, List, Tuple, Any
import random

from src.core import register

@register()
class PaddingAwareMixedAugmentation:
    """
    Mixed augmentation pipeline that randomly applies either:
    1. Distorted preprocessing (current training method)
    2. Padded preprocessing (aspect ratio preservation)
    
    This teaches the model to handle both conditions naturally.
    """
    
    def __init__(self, 
                 padding_ratio: float = 0.3,
                 target_size: Tuple[int, int] = (640, 640),
                 bbox_format: str = 'coco',
                 keypoint_format: str = 'xy',
                 color_prob: float = 0.8,
                 geometric_prob: float = 0.7,
                 flip_prob: float = 0.5):
        """
        Args:
            padding_ratio: Probability of using padded preprocessing (0.0-1.0)
            target_size: Target image size (width, height)
            bbox_format: Bounding box format for albumentations
            keypoint_format: Keypoint format for albumentations
            color_prob: Probability of applying color augmentations
            geometric_prob: Probability of applying geometric augmentations
            flip_prob: Probability of horizontal flip
        """
        self.padding_ratio = padding_ratio
        self.target_size = target_size
        self.bbox_format = bbox_format
        self.keypoint_format = keypoint_format
        
        # Color augmentations (applied to both distorted and padded)
        self.color_aug = A.Compose([
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1, 
                p=color_prob
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=color_prob * 0.8
            ),
        ], bbox_params=A.BboxParams(format=bbox_format, label_fields=['class_labels']),
           keypoint_params=A.KeypointParams(format=keypoint_format, label_fields=['keypoint_labels']))
        
        # Geometric augmentations (applied before preprocessing choice)
        self.geometric_aug = A.Compose([
            A.HorizontalFlip(p=flip_prob),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=15, 
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=geometric_prob
            ),
        ], bbox_params=A.BboxParams(format=bbox_format, label_fields=['class_labels']),
           keypoint_params=A.KeypointParams(format=keypoint_format, label_fields=['keypoint_labels']))
        
        # Final normalization
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def __call__(self, image: np.ndarray, targets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply mixed augmentation pipeline
        
        Args:
            image: Input image (H, W, C) numpy array
            targets: Dictionary containing 'boxes', 'labels', 'keypoints' etc.
            
        Returns:
            Dictionary with augmented image and targets
        """
        
        # Prepare data for albumentations
        bboxes = targets.get('boxes', [])
        keypoints = targets.get('keypoints', [])
        labels = targets.get('labels', [])
        
        # Convert keypoints format if needed
        keypoint_data = []
        keypoint_labels = []
        if keypoints is not None and len(keypoints) > 0:
            for i, face_keypoints in enumerate(keypoints):
                if hasattr(face_keypoints, 'shape'):  # Tensor
                    face_keypoints = face_keypoints.numpy()
                
                for k, kp in enumerate(face_keypoints):
                    if len(kp) >= 2:
                        x, y = float(kp[0]), float(kp[1])
                        visibility = float(kp[2]) if len(kp) > 2 else 1.0
                        if visibility > 0.3:  # Only include visible keypoints
                            keypoint_data.append([x, y])
                            keypoint_labels.append(f"face_{i}_kp_{k}")
        
        class_labels = [f"face_{i}" for i in range(len(bboxes))]
        
        # Apply color augmentations first
        try:
            augmented = self.color_aug(
                image=image,
                bboxes=bboxes,
                keypoints=keypoint_data,
                class_labels=class_labels,
                keypoint_labels=keypoint_labels
            )
        except Exception as e:
            # Fallback: skip color augmentation if it fails
            augmented = {
                'image': image,
                'bboxes': bboxes,
                'keypoints': keypoint_data,
                'class_labels': class_labels,
                'keypoint_labels': keypoint_labels
            }
        
        # Apply geometric augmentations
        try:
            augmented = self.geometric_aug(
                image=augmented['image'],
                bboxes=augmented['bboxes'],
                keypoints=augmented['keypoints'],
                class_labels=augmented['class_labels'],
                keypoint_labels=augmented['keypoint_labels']
            )
        except Exception as e:
            # Fallback: skip geometric augmentation if it fails
            pass
        
        # Randomly choose preprocessing method
        use_padding = np.random.random() < self.padding_ratio
        
        if use_padding:
            # PADDING PREPROCESSING (aspect ratio preservation)
            processed = self._apply_padding_preprocessing(
                augmented['image'], 
                augmented['bboxes'], 
                augmented['keypoints']
            )
        else:
            # DISTORTION PREPROCESSING (current method)
            processed = self._apply_distortion_preprocessing(
                augmented['image'], 
                augmented['bboxes'], 
                augmented['keypoints']
            )
        
        # Apply final normalization
        normalized = self.normalize(image=processed['image'])
        processed['image'] = normalized['image']
        
        # Convert back to expected format
        result = {
            'image': processed['image'],
            'boxes': torch.tensor(processed['bboxes'], dtype=torch.float32) if processed['bboxes'] else torch.empty((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.empty((0,), dtype=torch.long),
            'preprocessing_method': processed['preprocessing_method'],
            'effective_area_ratio': processed['effective_area_ratio'],
            'scale_factor': processed['scale_factor']
        }
        
        # Handle keypoints if present
        if processed['keypoints'] and len(processed['keypoints']) > 0:
            # Reconstruct keypoints per face
            reconstructed_keypoints = []
            for i in range(len(bboxes)):
                face_keypoints = []
                for j, kp_data in enumerate(processed['keypoints']):
                    if j // 5 == i:  # 5 keypoints per face
                        face_keypoints.append([kp_data[0], kp_data[1], 1.0])  # x, y, visibility
                
                if len(face_keypoints) == 5:
                    reconstructed_keypoints.append(face_keypoints)
                else:
                    # Pad with dummy keypoints if needed
                    while len(face_keypoints) < 5:
                        face_keypoints.append([0.0, 0.0, 0.0])
                    reconstructed_keypoints.append(face_keypoints[:5])
            
            result['keypoints'] = torch.tensor(reconstructed_keypoints, dtype=torch.float32)
        
        return result
    
    def _apply_padding_preprocessing(self, image: np.ndarray, bboxes: List, keypoints: List) -> Dict[str, Any]:
        """Apply aspect ratio preservation with padding"""
        
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Calculate scaling factor (maintain aspect ratio)
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize maintaining aspect ratio
        if new_w > 0 and new_h > 0:
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_image = image
            new_w, new_h = w, h
            scale = 1.0
        
        # Create padded canvas with black background
        padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding offsets (center the image)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Paste resized image onto padded canvas
        padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image
        
        # Transform bounding boxes
        transformed_bboxes = []
        for bbox in bboxes:
            if len(bbox) >= 4:
                x, y, x2, y2 = bbox[:4]
                # Scale and offset coordinates
                new_x = x * scale + pad_x
                new_y = y * scale + pad_y
                new_x2 = x2 * scale + pad_x
                new_y2 = y2 * scale + pad_y
                
                # Ensure bbox is within image bounds
                new_x = max(0, min(new_x, target_w))
                new_y = max(0, min(new_y, target_h))
                new_x2 = max(0, min(new_x2, target_w))
                new_y2 = max(0, min(new_y2, target_h))
                
                if new_x2 > new_x and new_y2 > new_y:  # Valid bbox
                    transformed_bboxes.append([new_x, new_y, new_x2, new_y2])
        
        # Transform keypoints
        transformed_keypoints = []
        for keypoint in keypoints:
            if len(keypoint) >= 2:
                x, y = keypoint[:2]
                new_x = x * scale + pad_x
                new_y = y * scale + pad_y
                
                # Ensure keypoint is within image bounds
                new_x = max(0, min(new_x, target_w))
                new_y = max(0, min(new_y, target_h))
                
                transformed_keypoints.append([new_x, new_y])
        
        return {
            'image': padded_image,
            'bboxes': transformed_bboxes,
            'keypoints': transformed_keypoints,
            'preprocessing_method': 'padded',
            'effective_area_ratio': (new_w * new_h) / (target_w * target_h),
            'scale_factor': scale,
            'padding': (pad_x, pad_y)
        }
    
    def _apply_distortion_preprocessing(self, image: np.ndarray, bboxes: List, keypoints: List) -> Dict[str, Any]:
        """Apply current distortion preprocessing (destroys aspect ratio)"""
        
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        # Simple resize that distorts aspect ratio
        resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate scaling factors
        scale_x = target_w / w
        scale_y = target_h / h
        
        # Transform bounding boxes
        transformed_bboxes = []
        for bbox in bboxes:
            if len(bbox) >= 4:
                x, y, x2, y2 = bbox[:4]
                new_x = x * scale_x
                new_y = y * scale_y
                new_x2 = x2 * scale_x
                new_y2 = y2 * scale_y
                
                # Ensure bbox is within image bounds
                new_x = max(0, min(new_x, target_w))
                new_y = max(0, min(new_y, target_h))
                new_x2 = max(0, min(new_x2, target_w))
                new_y2 = max(0, min(new_y2, target_h))
                
                if new_x2 > new_x and new_y2 > new_y:  # Valid bbox
                    transformed_bboxes.append([new_x, new_y, new_x2, new_y2])
        
        # Transform keypoints
        transformed_keypoints = []
        for keypoint in keypoints:
            if len(keypoint) >= 2:
                x, y = keypoint[:2]
                new_x = x * scale_x
                new_y = y * scale_y
                
                # Ensure keypoint is within image bounds
                new_x = max(0, min(new_x, target_w))
                new_y = max(0, min(new_y, target_h))
                
                transformed_keypoints.append([new_x, new_y])
        
        return {
            'image': resized_image,
            'bboxes': transformed_bboxes,
            'keypoints': transformed_keypoints,
            'preprocessing_method': 'distorted',
            'effective_area_ratio': 1.0,
            'scale_factor': min(scale_x, scale_y),
            'padding': (0, 0)
        }
    
    def set_padding_ratio(self, new_ratio: float):
        """Update padding ratio for curriculum learning"""
        self.padding_ratio = max(0.0, min(1.0, new_ratio))
        print(f"📊 Updated padding ratio: {self.padding_ratio:.2f}")

@register()
class CurriculumPaddingAugmentation(PaddingAwareMixedAugmentation):
    """
    Curriculum learning version that automatically adjusts padding ratio during training
    """
    
    def __init__(self, *args, **kwargs):
        self.initial_ratio = kwargs.get('padding_ratio', 0.1)
        self.target_ratio = kwargs.get('target_ratio', 0.8)
        self.curriculum_epochs = kwargs.get('curriculum_epochs', 30)
        self.current_epoch = 0
        
        super().__init__(*args, **kwargs)
    
    def update_epoch(self, epoch: int):
        """Update current epoch and adjust padding ratio accordingly"""
        self.current_epoch = epoch
        
        if epoch < self.curriculum_epochs:
            # Linear progression from initial to target ratio
            progress = epoch / self.curriculum_epochs
            new_ratio = self.initial_ratio + (self.target_ratio - self.initial_ratio) * progress
            self.set_padding_ratio(new_ratio)
        else:
            # Maintain target ratio after curriculum period
            self.set_padding_ratio(self.target_ratio)