#!/usr/bin/env python3

"""Visualize rotated FDDB samples with GT overlays."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Visualize FDDB rotated datasets')
    parser.add_argument('--rot-root', required=True,
                        help='Root directory produced by generate_fddb_rotated_sets.py')
    parser.add_argument('--orientations', nargs='+', default=['right', 'left', 'up', 'down'],
                        help='Orientations to visualize (default: all)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of images per orientation to visualize')
    parser.add_argument('--output-dir', default='fddb_rot_visuals', help='Directory to store visualizations')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling images')
    return parser.parse_args()


def read_label_file(path: Path) -> Dict[str, List[List[float]]]:
    mapping: Dict[str, List[List[float]]] = {}
    current: str | None = None
    boxes: List[List[float]] = []
    with path.open('r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if current is not None:
                    mapping[current] = boxes
                current = line[1:].strip()
                boxes = []
            else:
                parts = line.split()
                if len(parts) < 4:
                    continue
                x1, y1, x2, y2 = map(float, parts[:4])
                boxes.append([x1, y1, x2, y2])
    if current is not None:
        mapping[current] = boxes
    return mapping


def draw_boxes(image_path: Path, boxes: List[List[float]], save_path: Path) -> None:
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rot_root = Path(args.rot_root)
    output_dir = Path(args.output_dir)
    orientations = [o.lower() for o in args.orientations]

    for orientation in orientations:
        orient_root = rot_root / orientation
        label_path = orient_root / 'label.txt'
        images_root = orient_root / 'images'
        if not label_path.is_file():
            print(f'⚠️  Missing label file for {orientation}: {label_path}')
            continue
        mapping = read_label_file(label_path)
        if not mapping:
            print(f'⚠️  No labeled images for orientation {orientation}')
            continue

        items = list(mapping.items())
        random.shuffle(items)
        subset = items[: min(args.num_samples, len(items))]

        print(f'Orientation {orientation}: visualizing {len(subset)} images')
        for idx, (rel_path, boxes) in enumerate(subset, 1):
            img_path = images_root / rel_path
            if not img_path.is_file():
                print(f'   ⚠️  Missing image: {img_path}')
                continue
            save_path = output_dir / orientation / f'{idx:02d}_{Path(rel_path).name}'
            draw_boxes(img_path, boxes, save_path)
            print(f'   saved {save_path}')


if __name__ == '__main__':
    main()

