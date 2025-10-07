"""
Working Rotation Transform compatible with TorchVision TV tensors
การหมุนที่ทำงานได้กับ TV tensors
"""

import torch
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from ...core import register

@register()
class WorkingRotation(T.RandomRotation):
    """
    Working rotation transform that properly handles TorchVision TV tensors
    Uses TorchVision v2 RandomRotation for automatic coordinate transformation
    """
    
    def __init__(self, degrees=15, p=1.0):
        # Convert degrees to range format expected by TorchVision
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        
        super().__init__(
            degrees=degrees,
            interpolation=T.InterpolationMode.BILINEAR,
            expand=False,  # Don't expand image size
            center=None,   # Rotate around center
            fill=0         # Fill with black
        )
        self.p = p
    
    def forward(self, *inputs):
        """Forward method compatible with RT-DETR transform pipeline"""
        
        # Handle different input formats
        if len(inputs) == 1 and isinstance(inputs[0], tuple):
            # Called as transform(sample) where sample = (image, target, dataset)
            sample = inputs[0]
            if len(sample) == 3:
                image, target, dataset = sample
            else:
                image, target = sample[:2]
                dataset = None
        elif len(inputs) >= 2:
            # Called as transform(image, target, dataset)
            image, target = inputs[:2]
            dataset = inputs[2] if len(inputs) > 2 else None
        else:
            raise ValueError("Invalid input format for WorkingRotation")
        
        # Apply rotation with probability
        if torch.rand(1) > self.p:
            if dataset is not None:
                return image, target, dataset
            else:
                return image, target
        
        try:
            # Prepare TV tensors for TorchVision v2 transform
            if not isinstance(target['boxes'], tv_tensors.BoundingBoxes):
                # Convert regular tensor to TV tensor if needed
                boxes = tv_tensors.BoundingBoxes(
                    target['boxes'],
                    format='XYXY',
                    canvas_size=image.size[::-1]  # (H, W) format
                )
                target = dict(target)  # Create a copy
                target['boxes'] = boxes
            
            # Apply TorchVision v2 rotation (handles all coordinate transformations automatically)
            rotated_inputs = super().forward(image, target)
            
            if isinstance(rotated_inputs, tuple) and len(rotated_inputs) == 2:
                rotated_image, rotated_target = rotated_inputs
            else:
                # Sometimes TorchVision returns different format
                rotated_image = rotated_inputs
                rotated_target = target
            
            # Return in original format
            if dataset is not None:
                return rotated_image, rotated_target, dataset
            else:
                return rotated_image, rotated_target
            
        except Exception as e:
            print(f"⚠️ Rotation failed, using original: {e}")
            # Fallback to original if rotation fails
            if dataset is not None:
                return image, target, dataset
            else:
                return image, target
    
    def __call__(self, *inputs, **kwargs):
        """Call method - delegates to forward"""
        return self.forward(*inputs)
    
    def __repr__(self):
        return f"WorkingRotation(degrees={self.degrees}, p={self.p})"