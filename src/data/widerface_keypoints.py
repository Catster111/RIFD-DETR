import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T
from collections import defaultdict
import torchvision.tv_tensors as tv_tensors

from ..core import register
from ._misc import convert_to_tv_tensor

@register()
class WiderFaceKeypointDataset(Dataset):
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
        image = Image.open(img_path).convert('RGB')

        # Collect all bounding boxes and keypoints for this image
        boxes = []
        keypoints_list = []
        labels = []
        areas = []
        
        for ann in annotations:
            # Convert bbox from COCO format [x, y, w, h] to [x1, y1, x2, y2]
            bbox = ann['bbox']
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            
            # Keep keypoints in absolute image coordinates (will be normalized by NormalizeKeypoints transform)
            kpts_raw = torch.tensor(ann['keypoints'], dtype=torch.float32).view(-1, 3)  # [num_kpts, 3]
            keypoints_list.append(kpts_raw)
            
            labels.append(0)  # Face class = 0 (for training consistency)
            areas.append(ann['area'])
            
        # Create TorchVision-compatible tensors
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        boxes_tv = tv_tensors.BoundingBoxes(
            boxes_tensor, 
            format="XYXY",
            canvas_size=(img_meta['height'], img_meta['width'])
        )
        
        target = {
            'boxes': boxes_tv,  # Use TV tensor format
            'labels': torch.tensor(labels, dtype=torch.int64),
            'keypoints': torch.stack(keypoints_list) if keypoints_list else torch.empty((0, 5, 3)),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'orig_size': torch.tensor([img_meta['height'], img_meta['width']]),
            'size': torch.tensor([img_meta['width'], img_meta['height']])  # W, H format for TV
        }

        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self)
        else:
            # Apply basic transforms if no transform is provided
            orig_h, orig_w = image.height, image.width
            basic_transform = T.Compose([
                T.Resize((640, 640)),
                T.ToTensor()
            ])
            image = basic_transform(image)
            
            # Normalize boxes to [0, 1] range and convert to cxcywh format
            if len(target['boxes']) > 0:
                boxes = target['boxes'].clone().float()
                # Convert from x1y1x2y2 to cxcywh and normalize using ORIGINAL image size
                cx = (boxes[:, 0] + boxes[:, 2]) / 2 / orig_w
                cy = (boxes[:, 1] + boxes[:, 3]) / 2 / orig_h
                bw = (boxes[:, 2] - boxes[:, 0]) / orig_w
                bh = (boxes[:, 3] - boxes[:, 1]) / orig_h
                
                # Clamp to [0, 1] to ensure valid values
                cx = torch.clamp(cx, 0, 1)
                cy = torch.clamp(cy, 0, 1) 
                bw = torch.clamp(bw, 0, 1)
                bh = torch.clamp(bh, 0, 1)
                
                target['boxes'] = torch.stack([cx, cy, bw, bh], dim=1)
            
            # KEYPOINTS ARE ALREADY NORMALIZED BY BBOX in __getitem__
            # No additional normalization needed here
            
            # Add idx and convert boxes to proper format like COCO dataset
            target['idx'] = torch.tensor([idx])
            if 'boxes' in target and len(target['boxes']) > 0:
                target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=(640, 640))

        return image, target

    def set_epoch(self, epoch):
        """Set epoch for data augmentation policies"""
        self.epoch = epoch
        if hasattr(self._transforms, 'set_epoch'):
            self._transforms.set_epoch(epoch)

    def load_item(self, idx):
        """Load item without transforms for COCO evaluation compatibility"""
        image_id = self.image_ids[idx]
        img_meta = self.image_dict[image_id]
        annotations = self.img_ann_dict[image_id]
        
        # Handle WiderFace path format: remove "WIDER_train/images/" prefix
        file_name = img_meta['file_name']
        if file_name.startswith('WIDER_train/images/'):
            file_name = file_name.replace('WIDER_train/images/', '')
        img_path = os.path.join(self.img_prefix, file_name)
        image = Image.open(img_path).convert('RGB')

        # Collect all bounding boxes and keypoints for this image
        boxes = []
        keypoints_list = []
        labels = []
        areas = []
        
        for ann in annotations:
            # Convert bbox from COCO format [x, y, w, h] to [x1, y1, x2, y2]
            bbox = ann['bbox']
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            
            # Keep keypoints in absolute image coordinates (will be normalized by NormalizeKeypoints transform)
            kpts_raw = torch.tensor(ann['keypoints'], dtype=torch.float32).view(-1, 3)  # [num_kpts, 3]
            keypoints_list.append(kpts_raw)
            
            labels.append(0)  # Face class = 0 (for training consistency)
            areas.append(ann['area'])
        
        # Create TorchVision-compatible tensors for load_item too
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        boxes_tv = tv_tensors.BoundingBoxes(
            boxes_tensor, 
            format="XYXY",
            canvas_size=(img_meta['height'], img_meta['width'])
        )
            
        target = {
            'boxes': boxes_tv,  # Use TV tensor format
            'labels': torch.tensor(labels, dtype=torch.int64),
            'keypoints': torch.stack(keypoints_list) if keypoints_list else torch.empty((0, 5, 3)),
            'area': torch.tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'orig_size': torch.tensor([img_meta['height'], img_meta['width']]),
            'size': torch.tensor([img_meta['width'], img_meta['height']]),  # W, H format for TV
            'idx': torch.tensor([idx])  # Add idx like COCO dataset
        }

        return image, target
