#!/usr/bin/env python3

"""Generate rotated FDDB splits (right/left/up/down) with updated labels."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageFile

# Allow loading slightly truncated JPEGs in AFLW without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate rotated FDDB datasets')
    parser.add_argument('--image-root', required=True, help='Root directory containing FDDB images')
    parser.add_argument('--label-file', required=True, help='FDDB rectangle label file (x1 y1 x2 y2)')
    parser.add_argument('--output-root', required=True, help='Output root directory')
    parser.add_argument('--orientations', nargs='+', default=['right', 'left', 'up', 'down'],
                        help='Subset of {right,left,up,down} to generate')
    return parser.parse_args()


def read_fddb_labels(path: str) -> List[Tuple[str, List[List[float]]]]:
    entries: List[Tuple[str, List[List[float]]]] = []
    current: Optional[str] = None
    boxes: List[List[float]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if current is not None:
                    entries.append((current, boxes))
                current = line[1:].strip()
                boxes = []
            else:
                xs = line.split()
                if len(xs) < 4:
                    continue
                x1, y1, x2, y2 = map(float, xs[:4])
                boxes.append([x1, y1, x2, y2])
    if current is not None:
        entries.append((current, boxes))
    return entries


def rotate_points(points: Iterable[Tuple[float, float]], width: float, height: float, angle_deg: int) -> List[Tuple[float, float]]:
    angle_rad = math.radians(angle_deg % 360)
    sin_a = math.sin(angle_rad)
    cos_a = math.cos(angle_rad)
    cx, cy = width / 2.0, height / 2.0
    if angle_deg % 180 == 0:
        new_cx, new_cy = cx, cy
    else:
        new_cx, new_cy = height / 2.0, width / 2.0

    rotated: List[Tuple[float, float]] = []
    for x, y in points:
        x0 = x - cx
        y0 = y - cy
        xr = x0 * cos_a + y0 * sin_a
        yr = -x0 * sin_a + y0 * cos_a
        rotated.append((xr + new_cx, yr + new_cy))
    return rotated


def rotate_box(box: List[float], width: float, height: float, angle_deg: int) -> List[float]:
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    rotated = rotate_points(corners, width, height, angle_deg)
    xs = [p[0] for p in rotated]
    ys = [p[1] for p in rotated]
    return [min(xs), min(ys), max(xs), max(ys)]


def clamp_box(box: List[float], width: float, height: float) -> Optional[List[float]]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(width, x1))
    y1 = max(0.0, min(height, y1))
    x2 = max(0.0, min(width, x2))
    y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def rotate_image_and_boxes(image: Image.Image,
                           boxes: List[List[float]],
                           orientation: str) -> Tuple[Image.Image, List[List[float]]]:
    orientation = orientation.lower()
    orientation_map = {
        'up': 0,
        'down': 180,
        'left': 270,   # rotate 270° (−90°) to make faces tilt left in the new image
        'right': 90,   # rotate +90° to make faces tilt right
    }

    if orientation not in orientation_map:
        raise ValueError(f'Unsupported orientation: {orientation}')

    angle = orientation_map[orientation]
    if angle == 0:
        rotated = image
    elif angle == 180:
        rotated = image.transpose(Image.ROTATE_180)
    elif angle == 90:
        rotated = image.transpose(Image.ROTATE_90)
    else:  # 270
        rotated = image.transpose(Image.ROTATE_270)

    width, height = image.size
    new_width, new_height = rotated.size
    rotated_boxes: List[List[float]] = []
    for box in boxes:
        rot = rotate_box(box, width, height, angle)
        clamped = clamp_box(rot, new_width, new_height)
        if clamped is not None:
            rotated_boxes.append(clamped)
    return rotated, rotated_boxes


def open_image_safe(path: Path) -> Image.Image | None:
    """Open an image robustly; skip if corrupted/truncated beyond recovery."""
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert('RGB')
    except Exception as e:
        print(f"⚠️  Skipping corrupted image: {path} ({e})")
        return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_label(path: Path, labels: Dict[str, List[List[float]]]) -> None:
    with path.open('w') as f:
        for rel_path, boxes in labels.items():
            f.write(f'# {rel_path}\n')
            for box in boxes:
                x1, y1, x2, y2 = box
                f.write(f'{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n')


def main() -> None:
    args = parse_args()
    entries = read_fddb_labels(args.label_file)
    print(f'Loaded {len(entries)} entries from {args.label_file}')

    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    orientations = [o.lower() for o in args.orientations]
    valid = {'right', 'left', 'up', 'down'}
    for orientation in orientations:
        if orientation not in valid:
            raise ValueError(f'Invalid orientation: {orientation} (allowed: {valid})')

    for orientation in orientations:
        print(f"\n=== Generating {orientation.upper()} ===")
        orient_root = output_root / orientation
        orient_images = orient_root / 'images'
        ensure_dir(orient_images)

        label_map: Dict[str, List[List[float]]] = {}

        for rel_path, boxes in entries:
            src = image_root / rel_path
            if not src.is_file():
                print(f'⚠️  Missing image: {src}')
                continue

            image = open_image_safe(src)
            if image is None:
                continue
            rotated_image, rotated_boxes = rotate_image_and_boxes(image, boxes, orientation)

            dest = orient_images / rel_path
            ensure_dir(dest.parent)
            rotated_image.save(dest)

            if rotated_boxes:
                label_map[rel_path] = rotated_boxes

        label_path = orient_root / 'label.txt'
        write_label(label_path, label_map)
        print(f'Saved {len(label_map)} labelled images to {label_path}')


if __name__ == '__main__':
    main()
