"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn.functional as F 
import torch.distributed
import torchvision
from torch import Tensor 

from ...core import register

from typing import Dict 


__all__ = ['DetNMSPostProcessor', ]


@register()
class DetNMSPostProcessor(torch.nn.Module):
    def __init__(self, \
                iou_threshold=0.7, 
                score_threshold=0.01, 
                keep_topk=300, 
                box_fmt='cxcywh',
                logit_fmt='sigmoid') -> None:
        super().__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.keep_topk = keep_topk
        self.box_fmt = box_fmt.lower()
        self.logit_fmt = logit_fmt.lower()
        self.logit_func = getattr(F, self.logit_fmt, None)
        self.deploy_mode = False 
    
    def forward(self, outputs: Dict[str, Tensor], orig_target_sizes: Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        pred_boxes = torchvision.ops.box_convert(boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        pred_boxes *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        values, pred_labels = torch.max(logits, dim=-1)
        
        if self.logit_func:
            pred_scores = self.logit_func(values)
        else:
            pred_scores = values

        # Handle keypoints if present
        pred_keypoints = None
        if 'pred_keypoint_heatmaps' in outputs and 'pred_keypoint_offsets' in outputs:
            # Convert heatmaps + offsets to keypoint coordinates
            heatmaps = outputs['pred_keypoint_heatmaps']  # [B, N, K, H, W]
            offsets = outputs['pred_keypoint_offsets']    # [B, N, K, 2]
            pred_keypoints = self._extract_keypoints_from_heatmaps(heatmaps, offsets)

        # TODO for onnx export
        if self.deploy_mode:
            blobs = {
                'pred_labels': pred_labels, 
                'pred_boxes': pred_boxes,
                'pred_scores': pred_scores
            }
            if pred_keypoints is not None:
                blobs['pred_keypoints'] = pred_keypoints
            return blobs

        results = []
        for i in range(logits.shape[0]):
            score_keep = pred_scores[i] > self.score_threshold
            pred_box = pred_boxes[i][score_keep]
            pred_label = pred_labels[i][score_keep]
            pred_score = pred_scores[i][score_keep]

            # Process keypoints if available
            pred_keypoint_batch = None
            if pred_keypoints is not None:
                pred_keypoint_batch = pred_keypoints[i][score_keep]  # [N_keep, K, 3]

            keep = torchvision.ops.batched_nms(pred_box, pred_score, pred_label, self.iou_threshold)            
            keep = keep[:self.keep_topk]

            blob = {
                'labels': pred_label[keep],
                'boxes': pred_box[keep],
                'scores': pred_score[keep],
            }

            # Transform keypoints to absolute coordinates
            if pred_keypoint_batch is not None:
                filtered_keypoints = pred_keypoint_batch[keep]  # [N_final, K, 3]
                
                # Transform bbox-normalized keypoints to absolute coordinates
                final_boxes = pred_box[keep]  # [N_final, 4] in xyxy format
                transformed_keypoints = []
                
                for box_idx, box in enumerate(final_boxes):
                    if box_idx < len(filtered_keypoints):
                        x1, y1, x2, y2 = box
                        box_w = x2 - x1
                        box_h = y2 - y1
                        
                        kpts = filtered_keypoints[box_idx]  # [K, 3]
                        abs_kpts = torch.zeros_like(kpts)
                        
                        for k in range(kpts.shape[0]):
                            x_norm, y_norm, conf = kpts[k]
                            # Convert bbox-normalized to absolute coordinates
                            abs_x = x1 + x_norm * box_w
                            abs_y = y1 + y_norm * box_h
                            abs_kpts[k] = torch.tensor([abs_x, abs_y, conf])
                        
                        transformed_keypoints.append(abs_kpts)
                
                if transformed_keypoints:
                    blob['keypoints'] = torch.stack(transformed_keypoints)

            results.append(blob)
            
        return results

    def _extract_keypoints_from_heatmaps(self, heatmaps, offsets):
        """Extract keypoint coordinates from heatmaps and offsets"""
        B, N, K, H, W = heatmaps.shape
        device = heatmaps.device
        
        # Initialize keypoints tensor [B, N, K, 3] (x, y, confidence)
        keypoints = torch.zeros(B, N, K, 3, device=device)
        
        for b in range(B):
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[b, n, k]  # [H, W]
                    offset = offsets[b, n, k]    # [2]
                    
                    # Find max response in heatmap
                    max_val = torch.max(heatmap)
                    max_idx = torch.argmax(heatmap.flatten())
                    max_y = max_idx // W
                    max_x = max_idx % W
                    
                    # Convert peak position to normalized coordinates [0, 1]
                    norm_x = max_x.float() / (W - 1)
                    norm_y = max_y.float() / (H - 1)
                    
                    # Apply offset (already in normalized bbox coordinates from keypoint head)
                    norm_x = norm_x + offset[0]
                    norm_y = norm_y + offset[1]
                    
                    # Clamp to [0, 1] - these are bbox-normalized coordinates
                    norm_x = torch.clamp(norm_x, 0, 1)
                    norm_y = torch.clamp(norm_y, 0, 1)
                    
                    # Store as [x, y, confidence]
                    keypoints[b, n, k] = torch.tensor([norm_x, norm_y, max_val], device=device)
        
        return keypoints

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
