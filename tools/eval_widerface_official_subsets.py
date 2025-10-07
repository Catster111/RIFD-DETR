#!/usr/bin/env python3

"""Evaluate on WiderFace validation official subsets (easy/medium/hard).

This script follows the split protocol used in common WiderFace repos by
reading three text files that list image paths for each subset. Each line
should contain a relative path like:
    WIDER_val/images/0--Parade/0_Parade_marchingband_1_849.jpg

The script maps those to the COCO-converted dataset used in this repo and
evaluates AP for each subset via COCOeval by restricting imgIds.

Example:
  python tools/eval_widerface_official_subsets.py \
      -c configs/rtdetr/rtdetr_v2_face.yaml \
      --checkpoint output/rtdetr_r50vd_widerface_keypoints_v2_corrected_augmentation/last.pth \
      --easy-list dataset/splits/widerface_val_easy.txt \
      --medium-list dataset/splits/widerface_val_medium.txt \
      --hard-list dataset/splits/widerface_val_hard.txt

You can override model config like train.py using -u, e.g.:
  -u RTDETR.backbone=PResNet PResNet.depth=101 HybridEncoder.in_channels=[512,1024,2048]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig
from src.core.yaml_utils import parse_cli as parse_update
from src.data.dataset.coco_utils import get_coco_api_from_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser('WiderFace official subset evaluation')
    p.add_argument('-c', '--config', required=True, type=str)
    p.add_argument('--checkpoint', required=True, type=str)
    p.add_argument('-u', '--update', nargs='+', default=None)
    # Either provide text lists OR .mat lists (official WiderFace)
    p.add_argument('--easy-list', type=str, help='Text file with image paths (one per line)')
    p.add_argument('--medium-list', type=str, help='Text file with image paths (one per line)')
    p.add_argument('--hard-list', type=str, help='Text file with image paths (one per line)')
    p.add_argument('--easy-mat', type=str, help='Official WiderFace easy .mat file')
    p.add_argument('--medium-mat', type=str, help='Official WiderFace medium .mat file')
    p.add_argument('--hard-mat', type=str, help='Official WiderFace hard .mat file')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--mat-prefix', type=str, default='WIDER_val/images/',
                   help='Path prefix to prepend to event/file from .mat (default: WIDER_val/images/)')
    return p.parse_args()


def load_model(cfg_path: str, ckpt_path: str, update_args):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    update = parse_update(update_args) if update_args else {}
    cfg = YAMLConfig(cfg_path, **update)
    model = cfg.model
    post = cfg.postprocessor

    state = torch.load(ckpt_path, map_location='cpu')
    def _try_load(module, sd):
        try:
            module.load_state_dict(sd)
        except Exception as e:
            print(f"⚠️  Strict load failed: {e}. Retrying with strict=False…")
            module.load_state_dict(sd, strict=False)
    _try_load(model, state.get('model', state))
    if 'postprocessor' in state and post is not None:
        _try_load(post, state['postprocessor'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    if post is not None:
        post.to(device).eval()
    return model, post, cfg


def read_list(path: str) -> List[str]:
    with open(path, 'r') as f:
        items = [ln.strip() for ln in f.readlines() if ln.strip()]
    # Normalize slashes and ensure we compare with dataset file_name format
    items = [p.replace('\\', '/').lstrip('./') for p in items]
    return items


def _to_py_str(x) -> str:
    try:
        # scipy.io.loadmat returns numpy arrays of dtype=object/bytes
        import numpy as np
        if isinstance(x, np.ndarray):
            if x.dtype.kind in {'U', 'S'} and x.size == 1:
                return str(x.item())
            # nested cell -> take first element if 1x1
            if x.size == 1:
                return _to_py_str(x.item())
            raise TypeError
        if isinstance(x, (bytes, bytearray)):
            return x.decode('utf-8')
        return str(x)
    except Exception:
        return str(x)


def read_widerface_mat(mat_path: str, split_prefix: str = 'WIDER_val/images/') -> List[str]:
    """Parse WiderFace official subset .mat into a list of image relative paths.

    Supports common structures: {event_list, file_list} as nested cells/arrays.
    """
    items: List[str] = []
    try:
        from scipy.io import loadmat  # type: ignore
        import numpy as np
        data = loadmat(mat_path)

        def pick(keys):
            for k in keys:
                if k in data:
                    return data[k]
            return None

        event_arr = pick(['event_list', 'event', 'Event'])
        file_arr = pick(['file_list', 'file', 'File', 'List'])
        if event_arr is None or file_arr is None:
            raise KeyError('event_list/file_list not found in mat')

        # Flatten to 1-D lists while preserving pairing
        ev_flat = np.array(event_arr, dtype=object).reshape(-1)
        fi_flat = np.array(file_arr, dtype=object).reshape(-1)
        if ev_flat.shape[0] != fi_flat.shape[0]:
            # Some mats may be (1,E) vs (E,1); reshape(-1) handles, but assert anyway
            raise ValueError(f'mismatched event/file list lengths: {ev_flat.shape[0]} vs {fi_flat.shape[0]}')

        for ev_cell, files_cell in zip(ev_flat, fi_flat):
            ev_name = _to_py_str(ev_cell)
            files_cell = np.array(files_cell, dtype=object).reshape(-1)
            for fcell in files_cell:
                fn = _to_py_str(fcell)
                if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fn = f'{fn}.jpg'
                items.append(f'{split_prefix}{ev_name}/{fn}')

        return [p.replace('\\', '/').lstrip('./') for p in items]
    except ImportError:
        raise SystemExit('scipy is required to read .mat lists. Please `pip install scipy`.')


def build_imgid_mapping_from_coco(coco_gt) -> Dict[str, int]:
    # Map file_name to image_id
    mapping: Dict[str, int] = {}
    for img_id, meta in coco_gt.imgs.items():
        fn = meta.get('file_name', '')
        fn = fn.replace('\\', '/').lstrip('./')
        if not fn:
            # Some converted COCO objects omit file_name; skip to allow dataset fallback
            continue
        mapping[fn] = img_id
        # Also map without the leading 'WIDER_{train,val}/images/' for convenience
        if fn.startswith('WIDER_train/images/'):
            mapping[fn[len('WIDER_train/images/'):]] = img_id
        if fn.startswith('WIDER_val/images/'):
            mapping[fn[len('WIDER_val/images/'):]] = img_id
    return mapping


def build_imgid_mapping_from_dataset(dataset) -> Dict[str, int]:
    """Fallback mapping using dataset.image_dict (COCO JSON entries)."""
    mapping: Dict[str, int] = {}
    image_dict = getattr(dataset, 'image_dict', None)
    if not isinstance(image_dict, dict):
        return mapping
    for img_id, meta in image_dict.items():
        fn = str(meta.get('file_name', '')).replace('\\', '/').lstrip('./')
        if not fn:
            continue
        mapping[fn] = img_id
        if fn.startswith('WIDER_train/images/'):
            mapping[fn[len('WIDER_train/images/'):]] = img_id
        if fn.startswith('WIDER_val/images/'):
            mapping[fn[len('WIDER_val/images/'):]] = img_id
    return mapping


def ids_from_list(files: List[str], name2id: Dict[str, int]) -> List[int]:
    miss = 0
    ids = []
    for p in files:
        p2 = p
        if p2 not in name2id and p2.startswith('WIDER_val/images/'):
            p2 = p2[len('WIDER_val/images/') :]
        if p2 not in name2id and p2.startswith('WIDER_train/images/'):
            p2 = p2[len('WIDER_train/images/') :]
        if p2 not in name2id and p2.startswith('dataset/'):
            p2 = p2[len('dataset/') :]
        img_id = name2id.get(p2)
        if img_id is None:
            miss += 1
        else:
            ids.append(img_id)
    if miss:
        print(f'⚠️  {miss} paths from list not found in COCO annotations (check that lists match your dataset).')
    return ids


@torch.no_grad()
def run_inference(model, post, data_loader):
    device = next(model.parameters()).device
    all_results = []
    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)
        orig_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(device)
        results = post(outputs, orig_sizes)
        for tgt, out in zip(targets, results):
            image_id = int(tgt['image_id'])
            boxes = out['boxes']  # xywh in pixels
            scores = out['scores']
            labels = out['labels']
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu()
                scores = scores.detach().cpu()
                labels = labels.detach().cpu()
            for b, s, c in zip(boxes, scores, labels):
                all_results.append({
                    'image_id': image_id,
                    'category_id': int(c),
                    'bbox': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    'score': float(s),
                })
    return all_results


def evaluate_subset(coco_gt, results: List[Dict], img_ids: List[int], label: str):
    # Prefer faster_coco_eval if available
    try:
        from faster_coco_eval.core.coco_eval import COCOeval
    except Exception:
        from pycocotools.cocoeval import COCOeval  # type: ignore

    coco_dt = coco_gt.loadRes(results) if len(results) else None
    e = COCOeval(coco_gt, coco_dt, iouType='bbox')
    e.params.imgIds = img_ids
    e.params.useCats = 1
    e.evaluate(); e.accumulate();
    print(f'===== {label} subset =====')
    e.summarize()
    return e.stats[0], e.stats[1]


def main():
    args = parse_args()
    model, post, cfg = load_model(args.config, args.checkpoint, args.update)

    # Build (optionally override) val loader
    val_loader = cfg.val_dataloader
    if args.batch_size is not None and args.batch_size > 0:
        from src.core.workspace import create
        g = cfg.global_cfg
        g['val_dataloader']['total_batch_size'] = args.batch_size
        val_loader = create('val_dataloader', g, batch_size=args.batch_size)

    coco_gt = get_coco_api_from_dataset(val_loader.dataset)
    name2id = build_imgid_mapping_from_coco(coco_gt)
    if not name2id:
        # convert_to_coco_api may not preserve file_name; fallback to dataset mapping
        name2id = build_imgid_mapping_from_dataset(val_loader.dataset)

    # Build file lists from either .txt or .mat inputs
    def get_files(txt_path: str | None, mat_path: str | None, label: str) -> List[str]:
        if txt_path:
            return read_list(txt_path)
        if mat_path:
            return read_widerface_mat(mat_path, split_prefix=args.mat_prefix)
        raise SystemExit(f'Missing {label} list: provide either --{label}-list or --{label}-mat')

    easy_files = get_files(args.easy_list, args.easy_mat, 'easy')
    med_files = get_files(args.medium_list, args.medium_mat, 'medium')
    hard_files = get_files(args.hard_list, args.hard_mat, 'hard')

    easy_ids = ids_from_list(easy_files, name2id)
    med_ids = ids_from_list(med_files, name2id)
    hard_ids = ids_from_list(hard_files, name2id)

    print(f'Images: easy={len(easy_ids)}, medium={len(med_ids)}, hard={len(hard_ids)}')

    # Inference once over the (full) loader; we will filter per-subset via imgIds
    print('Running inference on validation set…')
    results = run_inference(model, post, val_loader)
    print(f'Collected {len(results)} detections.')

    easy_ap, easy_ap50 = evaluate_subset(coco_gt, results, easy_ids, 'easy')
    med_ap, med_ap50 = evaluate_subset(coco_gt, results, med_ids, 'medium')
    hard_ap, hard_ap50 = evaluate_subset(coco_gt, results, hard_ids, 'hard')

    print('Summary (AP, AP50):')
    print(f'  Easy  : {easy_ap*100:.2f}, {easy_ap50*100:.2f}')
    print(f'  Medium: {med_ap*100:.2f}, {med_ap50*100:.2f}')
    print(f'  Hard  : {hard_ap*100:.2f}, {hard_ap50*100:.2f}')


if __name__ == '__main__':
    main()
