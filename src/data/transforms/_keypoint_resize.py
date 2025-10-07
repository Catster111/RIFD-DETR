"""
Keypoint-aware Resize transform for RT-DETRv2
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.transforms import InterpolationMode
from torchvision import tv_tensors
from typing import Any, Dict
from ...core import register

@register()
class KeypointResize(T.Transform):
    """Resize transform that properly handles keypoints"""
    
    def __init__(self, size, max_size=None, interpolation=InterpolationMode.BILINEAR, antialias=None):
        super().__init__()
        self.size = size if isinstance(size, (list, tuple)) else [size, size]
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias
        
        # Create standard TorchVision Resize for image
        self.image_resize = T.Resize(
            size=size, 
            max_size=max_size, 
            interpolation=interpolation,
            antialias=antialias
        )
    
    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        
        image, target, dataset = inputs
        
        # Get original image dimensions
        if hasattr(image, 'shape'):
            # Tensor [C, H, W]
            if image.dim() == 3:
                _, orig_height, orig_width = image.shape
            else:
                orig_height, orig_width = image.shape
        else:
            # PIL Image
            orig_width, orig_height = image.size
        
        # Resize the image using standard TorchVision
        resized_image = self.image_resize(image)
        
        # Get new image dimensions
        if hasattr(resized_image, 'shape'):
            # Tensor [C, H, W]
            if resized_image.dim() == 3:
                _, new_height, new_width = resized_image.shape
            else:
                new_height, new_width = resized_image.shape
        else:
            # PIL Image
            new_width, new_height = resized_image.size
        
        # Calculate scaling factors
        width_scale = new_width / orig_width
        height_scale = new_height / orig_height
        
        # Scale target annotations
        new_target = {}
        for key, value in target.items():
            if key == 'boxes' and value.numel() > 0:
                scaled_boxes = value.clone()
                scaled_boxes[:, 0] *= width_scale
                scaled_boxes[:, 1] *= height_scale
                scaled_boxes[:, 2] *= width_scale
                scaled_boxes[:, 3] *= height_scale

                if isinstance(value, tv_tensors.BoundingBoxes):
                    new_target[key] = tv_tensors.BoundingBoxes(
                        scaled_boxes,
                        format=value.format,
                        canvas_size=(new_height, new_width)
                    )
                else:
                    new_target[key] = scaled_boxes
                
            elif key == 'keypoints' and value.numel() > 0:
                # Scale keypoints [x, y, visibility]
                scaled_keypoints = value.clone()
                scaled_keypoints[:, :, 0] *= width_scale  # x coordinates
                scaled_keypoints[:, :, 1] *= height_scale # y coordinates
                # visibility unchanged
                new_target[key] = scaled_keypoints
                
            elif key == 'size':
                # Update current size information
                new_target[key] = torch.tensor([new_height, new_width], dtype=value.dtype)
                
            elif key == 'orig_size':
                # CRITICAL FIX: Keep original image size unchanged for PostProcessor scaling
                # Do NOT update orig_size to resized dimensions
                new_target[key] = value
                
            else:
                # Keep other fields unchanged
                new_target[key] = value
        
        return resized_image, new_target, dataset

# Also provide backward compatibility
@register(name='Resize')
class Resize(KeypointResize):
    """Backward compatible Resize that handles keypoints"""
    pass
