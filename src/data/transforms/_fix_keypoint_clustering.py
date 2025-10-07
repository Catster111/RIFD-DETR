"""
Fix for keypoint clustering in WiderFace dataset
แก้ปัญหา keypoint clustering ในข้อมูล WiderFace
"""

import torch
import random
import numpy as np
from ...core import register

@register()
class FilterLowDiversityKeypoints:
    """Filter out training samples with low keypoint diversity to prevent model collapse"""
    
    def __init__(self, min_x_std=0.03, min_y_std=0.03, min_visible_kp=3, skip_probability=1.0):
        self.min_x_std = min_x_std
        self.min_y_std = min_y_std  
        self.min_visible_kp = min_visible_kp
        self.skip_probability = skip_probability  # Probability to skip low-diversity samples
    
    def forward(self, *inputs):
        """Filter samples based on keypoint diversity after normalization"""
        
        if len(inputs) == 1:
            inputs = inputs[0]
        
        image, target, dataset = inputs
        
        if 'keypoints' in target and target['keypoints'].numel() > 0:
            kp = target['keypoints']
            
            if len(kp.shape) == 3:  # [num_instances, num_keypoints, 3]
                for inst_idx in range(kp.shape[0]):
                    instance_kp = kp[inst_idx]
                    
                    # Get visible keypoints
                    visible_mask = instance_kp[:, 2] > 0
                    visible_kp = instance_kp[visible_mask]
                    
                    if len(visible_kp) >= self.min_visible_kp:
                        coords = visible_kp[:, :2].cpu().numpy()  # [N, 2]
                        
                        x_std = np.std(coords[:, 0])
                        y_std = np.std(coords[:, 1])
                        
                        # Check if diversity is too low
                        if x_std < self.min_x_std or y_std < self.min_y_std:
                            # Decide whether to skip this sample
                            if random.random() < self.skip_probability:
                                # Skip this sample by returning the next sample indices
                                # (This is handled by the dataset loader)
                                return None
        
        return image, target, dataset
    
    def __call__(self, *inputs):
        return self.forward(*inputs)

@register()
class EnhanceKeypointDiversity:
    """Add controlled noise to keypoints to prevent clustering and improve generalization"""
    
    def __init__(self, noise_std=0.025, p=0.8, min_diversity_threshold=0.03, force_min_std=0.035):
        self.noise_std = noise_std
        self.p = p
        self.min_diversity_threshold = min_diversity_threshold
        self.force_min_std = force_min_std  # Force minimum standard deviation
    
    def forward(self, *inputs):
        """Add small noise to keypoints when diversity is low"""
        
        if len(inputs) == 1:
            inputs = inputs[0]
        
        image, target, dataset = inputs
        
        if random.random() > self.p:
            return image, target, dataset
        
        if 'keypoints' in target and target['keypoints'].numel() > 0:
            kp = target['keypoints'].clone()
            
            if len(kp.shape) == 3:  # [num_instances, num_keypoints, 3]
                for inst_idx in range(kp.shape[0]):
                    visible_mask = kp[inst_idx, :, 2] > 0
                    visible_coords = kp[inst_idx, visible_mask, :2]
                    
                    if len(visible_coords) >= 3:  # Need sufficient keypoints
                        # Check current diversity
                        x_std = torch.std(visible_coords[:, 0]).item()
                        y_std = torch.std(visible_coords[:, 1]).item()
                        
                        # Only add noise if diversity is low
                        if x_std < self.min_diversity_threshold or y_std < self.min_diversity_threshold:
                            # Calculate how much noise is needed to reach minimum standard deviation
                            target_x_std = max(x_std, self.force_min_std)
                            target_y_std = max(y_std, self.force_min_std)
                            
                            # Calculate center of visible keypoints
                            center_x = visible_coords[:, 0].mean()
                            center_y = visible_coords[:, 1].mean()
                            
                            # Enhanced noise generation - spread keypoints from center
                            for kp_idx in range(kp.shape[1]):
                                if visible_mask[kp_idx]:
                                    # Calculate distance from center
                                    dx = kp[inst_idx, kp_idx, 0] - center_x
                                    dy = kp[inst_idx, kp_idx, 1] - center_y
                                    
                                    # Add directional noise (away from center) + random noise
                                    directional_noise_x = dx.sign() * self.noise_std * 0.5  # Push away from center
                                    directional_noise_y = dy.sign() * self.noise_std * 0.5
                                    
                                    random_noise_x = torch.randn(1) * self.noise_std
                                    random_noise_y = torch.randn(1) * self.noise_std
                                    
                                    # Total noise
                                    total_noise_x = directional_noise_x + random_noise_x
                                    total_noise_y = directional_noise_y + random_noise_y
                                    
                                    # Apply noise
                                    kp[inst_idx, kp_idx, 0] = kp[inst_idx, kp_idx, 0] + total_noise_x
                                    kp[inst_idx, kp_idx, 1] = kp[inst_idx, kp_idx, 1] + total_noise_y
                            
                            # Clamp to valid range [0, 1] (assuming normalized coordinates)
                            kp[inst_idx, :, 0] = torch.clamp(kp[inst_idx, :, 0], 0, 1)
                            kp[inst_idx, :, 1] = torch.clamp(kp[inst_idx, :, 1], 0, 1)
            
            target['keypoints'] = kp
        
        return image, target, dataset
    
    def __call__(self, *inputs):
        return self.forward(*inputs)

@register() 
class AdaptiveKeypointNormalization:
    """Adaptive normalization that ensures minimum keypoint spread"""
    
    def __init__(self, min_spread=0.05, spread_boost_factor=2.0):
        self.min_spread = min_spread
        self.spread_boost_factor = spread_boost_factor
    
    def forward(self, *inputs):
        """Ensure keypoints have minimum spread after normalization"""
        
        if len(inputs) == 1:
            inputs = inputs[0]
        
        image, target, dataset = inputs
        
        if 'keypoints' in target and target['keypoints'].numel() > 0:
            kp = target['keypoints'].clone()
            
            if len(kp.shape) == 3:  # [num_instances, num_keypoints, 3]
                for inst_idx in range(kp.shape[0]):
                    visible_mask = kp[inst_idx, :, 2] > 0
                    visible_coords = kp[inst_idx, visible_mask, :2]
                    
                    if len(visible_coords) >= 3:
                        # Calculate current spread
                        x_range = visible_coords[:, 0].max() - visible_coords[:, 0].min()
                        y_range = visible_coords[:, 1].max() - visible_coords[:, 1].min()
                        
                        # If spread is too small, enhance it
                        if x_range < self.min_spread or y_range < self.min_spread:
                            # Find center
                            center_x = visible_coords[:, 0].mean()
                            center_y = visible_coords[:, 1].mean()
                            
                            # Scale coordinates from center
                            scale_factor = max(
                                self.min_spread / x_range if x_range > 0 else self.spread_boost_factor,
                                self.min_spread / y_range if y_range > 0 else self.spread_boost_factor
                            )
                            scale_factor = min(scale_factor, self.spread_boost_factor)  # Limit scaling
                            
                            # Apply scaling
                            scaled_coords = visible_coords.clone()
                            scaled_coords[:, 0] = center_x + (scaled_coords[:, 0] - center_x) * scale_factor
                            scaled_coords[:, 1] = center_y + (scaled_coords[:, 1] - center_y) * scale_factor
                            
                            # Clamp to [0, 1]
                            scaled_coords = torch.clamp(scaled_coords, 0, 1)
                            
                            # Update original tensor
                            kp[inst_idx, visible_mask, :2] = scaled_coords
            
            target['keypoints'] = kp
        
        return image, target, dataset
    
    def __call__(self, *inputs):
        return self.forward(*inputs)

# For debugging and monitoring
class KeypointDiversityMonitor:
    """Monitor keypoint diversity during training"""
    
    def __init__(self):
        self.diversity_stats = []
    
    def __call__(self, image, target, dataset=None):
        """Monitor and log keypoint diversity"""
        
        if 'keypoints' in target and target['keypoints'].numel() > 0:
            kp = target['keypoints']
            
            if len(kp.shape) == 3:
                for inst_idx in range(kp.shape[0]):
                    visible_mask = kp[inst_idx, :, 2] > 0
                    visible_coords = kp[inst_idx, visible_mask, :2]
                    
                    if len(visible_coords) >= 2:
                        x_std = torch.std(visible_coords[:, 0]).item()
                        y_std = torch.std(visible_coords[:, 1]).item()
                        
                        self.diversity_stats.append({
                            'x_std': x_std,
                            'y_std': y_std,
                            'num_visible': len(visible_coords)
                        })
        
        return image, target, dataset
    
    def get_stats(self):
        """Get diversity statistics"""
        if not self.diversity_stats:
            return None
        
        x_stds = [s['x_std'] for s in self.diversity_stats]
        y_stds = [s['y_std'] for s in self.diversity_stats]
        
        return {
            'mean_x_std': np.mean(x_stds),
            'mean_y_std': np.mean(y_stds),
            'min_x_std': np.min(x_stds),
            'min_y_std': np.min(y_stds),
            'samples': len(self.diversity_stats),
            'low_diversity_count': sum(1 for s in self.diversity_stats 
                                     if s['x_std'] < 0.03 or s['y_std'] < 0.03)
        }