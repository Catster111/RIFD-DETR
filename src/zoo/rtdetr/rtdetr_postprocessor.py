"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np

import torchvision

from ...core import register
from .rtdetr_keypoint_head import (
    RTDETRKeypointHead,
    POLAR_RADIUS_SCALE,
    heatmap_expectation_xy,
)


__all__ = ['RTDETRPostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class RTDETRPostProcessor(nn.Module):
    __share__ = [
        'num_classes', 
        'use_focal_loss', 
        'num_top_queries', 
        'remap_mscoco_category',
        'remap_widerface_category'
    ]
    
    def __init__(
        self, 
        num_classes=80, 
        use_focal_loss=True, 
        num_top_queries=300, 
        remap_mscoco_category=False,
        remap_widerface_category=False,
        use_keypoints=False,
        keypoint_heatmap_domain='cartesian',
        enable_nms: bool = False,
        nms_iou_threshold: float = 0.6,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category 
        self.remap_widerface_category = remap_widerface_category
        self.use_keypoints = use_keypoints
        self.deploy_mode = False 
        self.keypoint_heatmap_domain = keypoint_heatmap_domain
        if self.use_keypoints and self.keypoint_heatmap_domain not in {'cartesian', 'polar'}:
            raise ValueError(f"Unsupported heatmap domain: {self.keypoint_heatmap_domain}")
        # Optional NMS to remove duplicate detections from multiple queries
        self.enable_nms = bool(enable_nms)
        self.nms_iou_threshold = float(nms_iou_threshold)

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        # Convert to XYWH for COCO evaluation compatibility
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xywh')
        # FIXED: Use correct W,H,W,H scaling instead of H,W,H,W
        # Extract H,W from orig_target_sizes (which is in H,W format)
        img_h, img_w = orig_target_sizes.unbind(1)
        # Create scaling factor: [W, H, W, H] for [x, y, w, h]
        scale_wh = torch.stack([img_w, img_h, img_w, img_h], dim=1).unsqueeze(1)
        bbox_pred *= scale_wh
        
        # Clamp bbox coordinates to valid range (fix negative coordinates)
        bbox_pred[..., 0] = torch.clamp(bbox_pred[..., 0], min=0)  # x >= 0
        bbox_pred[..., 1] = torch.clamp(bbox_pred[..., 1], min=0)  # y >= 0  
        bbox_pred[..., 2] = torch.clamp(bbox_pred[..., 2], min=1)  # w >= 1
        bbox_pred[..., 3] = torch.clamp(bbox_pred[..., 3], min=1)  # h >= 1

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            # CRITICAL FIX: Prevent topk out of range error
            k = min(self.num_top_queries, scores.flatten(1).shape[1])
            scores, index = torch.topk(scores.flatten(1), k, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
            
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))
        
        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)
        
        # WiderFace category remapping: label 0 -> category_id 1  
        elif self.remap_widerface_category:
            # Convert model predictions (0) back to COCO category_id (1) for evaluation
            labels = labels + 1

        # Process keypoints if available
        keypoints_list = None
        if self.use_keypoints and 'pred_keypoint_heatmaps' in outputs and 'pred_keypoint_offsets' in outputs:
            pred_heatmaps = outputs['pred_keypoint_heatmaps'] 
            pred_offsets = outputs['pred_keypoint_offsets']
            
            # Convert boxes to XYXY format for keypoint decoding
            # Use 'boxes' (selected top-k) instead of 'bbox_pred' (all predictions)
            boxes_xyxy = torchvision.ops.box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')
            
            # Get top k indices for keypoint extraction
            keypoints_list = []
            for b_idx in range(pred_heatmaps.shape[0]):
                batch_keypoints = []
                batch_indices = index[b_idx] if self.use_focal_loss else torch.arange(boxes.shape[1], device=boxes.device)
                
                for i, query_idx in enumerate(batch_indices):
                    if i >= len(boxes[b_idx]):
                        break
                        
                    # Extract keypoints for this query
                    query_heatmaps = pred_heatmaps[b_idx, query_idx]  # [5, H, W]
                    query_offsets = pred_offsets[b_idx, query_idx]    # [5, 2]
                    query_bbox = boxes_xyxy[b_idx, i]               # [4] XYXY
                    
                    # Decode keypoints using RTDETRKeypointHead method
                    # NOTE: query_bbox is already scaled to original image size, no need for orig_target_sizes
                    query_keypoints = self._decode_keypoints_single(
                        query_heatmaps, query_offsets, query_bbox, None
                    )
                    batch_keypoints.append(query_keypoints)
                
                keypoints_list.append(batch_keypoints)
        
        # Optionally apply per-image NMS to suppress duplicate boxes
        if self.enable_nms:
            # Keep per-image lists after NMS (variable number per image)
            boxes_xyxy_all = torchvision.ops.box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')
            labels_list, boxes_list, scores_list = [], [], []
            new_keypoints_list = [] if keypoints_list is not None else None
            for b in range(boxes.shape[0]):
                b_boxes_xyxy = boxes_xyxy_all[b]
                b_scores = scores[b]
                keep = torchvision.ops.nms(b_boxes_xyxy, b_scores, self.nms_iou_threshold)
                labels_list.append(labels[b][keep])
                boxes_list.append(boxes[b][keep])  # keep still in xywh
                scores_list.append(b_scores[keep])
                if keypoints_list is not None:
                    kept_kpts = [keypoints_list[b][int(i)] for i in keep.tolist()]
                    new_keypoints_list.append(kept_kpts)
            labels, boxes, scores = labels_list, boxes_list, scores_list
            if keypoints_list is not None:
                keypoints_list = new_keypoints_list

        results = []
        # Support both batched tensors (no-NMS path) and per-image lists (NMS path)
        if isinstance(labels, list):
            iterable = zip(range(len(labels)), labels, boxes, scores)
        else:
            iterable = zip(range(labels.shape[0]), labels, boxes, scores)

        for i, lab, box, sco in iterable:
            result = dict(labels=lab, boxes=box, scores=sco)
            
            # Add keypoints if available
            if keypoints_list is not None and i < len(keypoints_list):
                result['keypoints'] = keypoints_list[i]
            
            results.append(result)
        
        return results
    
    def _decode_keypoints_single(self, heatmaps, offsets, bbox, orig_target_size=None):
        """
        Decode keypoints for a single query
        
        Args:
            heatmaps: [num_keypoints, H, W] heatmap predictions
            offsets: [num_keypoints, 2] offset predictions 
            bbox: [4] bounding box in XYXY format (already scaled to original image size)
            orig_target_size: DEPRECATED - bbox should already be scaled
            
        Returns:
            keypoints: [num_keypoints, 3] (x, y, confidence)
        """
        num_keypoints, H, W = heatmaps.shape
        device = heatmaps.device
        
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        bbox_w = bbox_x2 - bbox_x1
        bbox_h = bbox_y2 - bbox_y1
        
        keypoints = []
        
        for k in range(num_keypoints):
            heatmap = heatmaps[k]  # [H, W]
            offset = offsets[k]    # [2] (dx, dy)

            base_x, base_y = heatmap_expectation_xy(
                heatmap, self.keypoint_heatmap_domain, POLAR_RADIUS_SCALE
            )
            final_x_norm = torch.clamp(base_x + offset[0], 0, 1)
            final_y_norm = torch.clamp(base_y + offset[1], 0, 1)
            max_conf = heatmap.max()
            
            # Transform to absolute coordinates within the bounding box
            # Keypoints should be relative to the bounding box, not full image
            final_x = bbox_x1 + final_x_norm * bbox_w
            final_y = bbox_y1 + final_y_norm * bbox_h
            
            keypoints.append([final_x.item(), final_y.item(), max_conf.item()])
        
        return torch.tensor(keypoints, device=device)  # [5, 3]

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 
