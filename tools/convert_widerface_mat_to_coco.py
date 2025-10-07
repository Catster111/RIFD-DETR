#!/usr/bin/env python3

"""Convert WiderFace official .mat annotations to COCO JSON.

This builds a COCO-style file for the validation split using the official
`wider_face_val.mat` (or similarly structured) file that contains event/file
lists and face bounding boxes. It also injects dummy 5-keypoint entries so it
remains compatible with the WiderFaceKeypointDatasetWorking during evaluation
with bbox-only metrics.

Example:
  python tools/convert_widerface_mat_to_coco.py \
    --mat dataset/splits/wider_face_val.mat \
    --images-root dataset/WIDER_val/images \
    --output dataset/annotations/widerface_val_coco.json

Notes:
- Requires SciPy to read .mat files: `pip install scipy`.
- COCO `file_name` is stored as `WIDER_val/images/<event>/<file>.jpg`.
  Adjust `--prefix` if you want a different path recorded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def _to_py_str(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode('utf-8')
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return _to_py_str(x.item())
        raise TypeError('Expected scalar array for string cell')
    return str(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('Convert WiderFace .mat to COCO JSON')
    p.add_argument('--mat', required=True, type=str, help='Path to wider_face_val.mat')
    p.add_argument('--images-root', required=True, type=str, help='Root dir to images (…/WIDER_val/images)')
    p.add_argument('--output', required=True, type=str, help='Output COCO JSON path')
    p.add_argument('--prefix', type=str, default='WIDER_val/images/', help='Prefix to store in file_name')
    p.add_argument('--print-keys', action='store_true', help='Print top-level .mat keys and chosen field names')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from scipy.io import loadmat  # type: ignore
    except Exception:
        raise SystemExit('scipy is required. Please `pip install scipy`.')

    data = loadmat(args.mat)

    # Optional: print top-level keys for quick debugging
    if args.print_keys:
        tl_keys = [k for k in data.keys() if not k.startswith('__')]
        print('Top-level .mat keys:', tl_keys)

    def pick(keys):
        for k in keys:
            if k in data:
                return data[k]
        return None

    event_keys = ['event_list', 'event', 'Event']
    file_keys  = ['file_list', 'file', 'File', 'List']
    bbox_keys  = ['face_bbx_list', 'bbox_list', 'bboxes', 'face_bbx']
    blur_keys  = ['blur_label_list', 'blur_list', 'blur']
    occ_keys   = ['occlusion_label_list', 'occlusion_list', 'occlusion']
    pose_keys  = ['pose_label_list', 'pose_list', 'pose']
    inv_keys   = ['invalid_label_list', 'invalid_list', 'invalid']

    event_arr = pick(event_keys)
    file_arr = pick(file_keys)
    # bbox list can appear under different names
    bbox_arr = pick(bbox_keys)
    blur_arr = pick(blur_keys)
    occ_arr = pick(occ_keys)
    pose_arr = pick(pose_keys)
    invalid_arr = pick(inv_keys)

    # Log chosen fields and warn for missing
    def chosen(src_keys, val, name):
        found = next((k for k in src_keys if k in data), None)
        print(f"{name}: {'FOUND ' + found if found else 'MISSING'}")

    if args.print_keys:
        chosen(event_keys, event_arr, 'event_list')
        chosen(file_keys,  file_arr,  'file_list')
        chosen(bbox_keys,  bbox_arr,  'face_bbx_list')
        chosen(blur_keys,  blur_arr,  'blur_label_list')
        chosen(occ_keys,   occ_arr,   'occlusion_label_list')
        chosen(pose_keys,  pose_arr,  'pose_label_list')
        chosen(inv_keys,   invalid_arr,'invalid_label_list')
    if event_arr is None or file_arr is None or bbox_arr is None:
        raise SystemExit('Missing keys in .mat (need event_list, file_list, and face_bbx_list).')

    if args.print_keys and (blur_arr is None or occ_arr is None or pose_arr is None or invalid_arr is None):
        print('⚠️  One or more difficulty attribute arrays are missing. These will default to 0.')

    ev_flat = np.array(event_arr, dtype=object).reshape(-1)
    fi_flat = np.array(file_arr, dtype=object).reshape(-1)
    bb_flat = np.array(bbox_arr, dtype=object).reshape(-1)
    bl_flat = np.array(blur_arr, dtype=object).reshape(-1) if blur_arr is not None else None
    oc_flat = np.array(occ_arr, dtype=object).reshape(-1) if occ_arr is not None else None
    po_flat = np.array(pose_arr, dtype=object).reshape(-1) if pose_arr is not None else None
    in_flat = np.array(invalid_arr, dtype=object).reshape(-1) if invalid_arr is not None else None

    if not (len(ev_flat) == len(fi_flat) == len(bb_flat)):
        raise SystemExit('Mismatched lengths between events/files/bboxes in .mat')

    images: List[Dict] = []
    annotations: List[Dict] = []
    categories = [{"id": 1, "name": "face"}]

    image_id = 1
    ann_id = 1

    images_root = Path(args.images_root)
    prefix = args.prefix.rstrip('/') + '/'

    for idx, (ev_cell, files_cell, bboxes_cell) in enumerate(zip(ev_flat, fi_flat, bb_flat)):
        event_name = _to_py_str(ev_cell)
        files = np.array(files_cell, dtype=object).reshape(-1)
        bboxes_list = np.array(bboxes_cell, dtype=object).reshape(-1)
        bl_list = np.array(bl_flat[idx], dtype=object).reshape(-1) if bl_flat is not None else None
        oc_list = np.array(oc_flat[idx], dtype=object).reshape(-1) if oc_flat is not None else None
        po_list = np.array(po_flat[idx], dtype=object).reshape(-1) if po_flat is not None else None
        in_list = np.array(in_flat[idx], dtype=object).reshape(-1) if in_flat is not None else None
        if len(files) != len(bboxes_list):
            raise SystemExit(f'Mismatch inside event {event_name}: {len(files)} files vs {len(bboxes_list)} bbox sets')

        for j, (fcell, bcell) in enumerate(zip(files, bboxes_list)):
            fname = _to_py_str(fcell)
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fname = f'{fname}.jpg'
            rel_path = f'{prefix}{event_name}/{fname}'.replace('\\', '/')
            img_path = images_root / event_name / fname
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
            except Exception as e:
                raise SystemExit(f'Cannot open image: {img_path} ({e})')

            images.append({
                'id': image_id,
                'file_name': rel_path,
                'width': width,
                'height': height,
            })

            # bcell can be empty or Nx4; ensure 2D
            b_arr = np.array(bcell)
            if b_arr.size == 0:
                b_arr = b_arr.reshape(0, 4)
            else:
                b_arr = b_arr.reshape(-1, 4)

            # per-image attribute arrays (same length as bboxes), may be missing
            bl = np.array(bl_list[j]).reshape(-1) if bl_list is not None else np.zeros((b_arr.shape[0],), dtype=np.int32)
            oc = np.array(oc_list[j]).reshape(-1) if oc_list is not None else np.zeros((b_arr.shape[0],), dtype=np.int32)
            po = np.array(po_list[j]).reshape(-1) if po_list is not None else np.zeros((b_arr.shape[0],), dtype=np.int32)
            inv = np.array(in_list[j]).reshape(-1) if in_list is not None else np.zeros((b_arr.shape[0],), dtype=np.int32)

            for k, bb in enumerate(b_arr):
                x, y, w, h = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
                annotations.append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': [x, y, w, h],
                    'area': float(w * h),
                    'iscrowd': 0,
                    'segmentation': [],
                    # dummy 5 keypoints (x,y,v) all zero to satisfy dataset class
                    'keypoints': [0]*15,
                    'num_keypoints': 0,
                    'wf_blur': int(bl[k]) if k < len(bl) else 0,
                    'wf_occlusion': int(oc[k]) if k < len(oc) else 0,
                    'wf_pose': int(po[k]) if k < len(po) else 0,
                    'wf_invalid': int(inv[k]) if k < len(inv) else 0,
                })
                ann_id += 1

            image_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
        'licenses': [],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        json.dump(coco, f)
    print(f'Wrote COCO JSON: {out_path} (images={len(images)}, anns={len(annotations)})')


if __name__ == '__main__':
    main()
