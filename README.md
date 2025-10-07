# RIFD-DETR
The DETR based for Rotation invariance Face Detection utilize Polar Transformation
<img width="559" height="776" alt="image" src="https://github.com/user-attachments/assets/c3ac8d65-08fc-4a8f-9c29-80cba6bd72cd" />
# RT-DETRv2 Face + 5 Keypoints — Training & Inference

This guide explains how to train and run inference for face detection with 5 facial keypoints using the customized RT‑DETRv2 in `rtdetrv2_pytorch` (config: `configs/rtdetr/rtdetr_v2_face.yaml`).

## 1) Installation & Environment
- Recommended: Python 3.10+ and PyTorch 2.0+
- Install dependencies (run inside `RIFD-DETR/rtdetrv2_pytorch`):

```bash
cd RIFD-DETR/rtdetrv2_pytorch
pip install -r requirements.txt
```

Note: If using a GPU, install a PyTorch/CUDA build compatible with your system.

## 2) Dataset Structure (COCO + Keypoints)
- COCO-style JSON with 5 keypoints per face: `dataset/annotations/widerface_keypoints_coco.json`
- Images folder: `dataset/images/`
- The dataset class (`WiderFaceKeypointDatasetWorking`) will strip the `WIDER_train/images/` prefix (if present) from `file_name` and look for images under `img_prefix` (`dataset/images/`).
- Keypoint order: `left_eye, right_eye, nose, mouth_left, mouth_right` (each point is `(x, y, v)` like COCO).

Related tools:
- Convert WiderFace `.mat` → COCO (for val/bbox-only): `tools/convert_widerface_mat_to_coco.py` (requires `scipy`).
- For training keypoints, you must have real 5-point annotations in the JSON (the converter inserts dummy keypoints only for bbox evaluation, not suitable for keypoint training).

## 3) Training the Model
Main config: `configs/rtdetr/rtdetr_v2_face.yaml` (polar heatmaps, 5 keypoints, `num_queries=75`, NMS enabled in the postprocessor, etc.)

Single-GPU example:
```bash
python tools/train.py -c configs/rtdetr/rtdetr_v2_face.yaml --use-amp --seed 0
```

Resume from a checkpoint:
```bash
python tools/train.py -c configs/rtdetr/rtdetr_v2_face.yaml \
  --use-amp --seed 0 \
  -r output/bestModel/last.pth
```

Evaluate only (test-only):
```bash
python tools/train.py -c configs/rtdetr/rtdetr_v2_face.yaml \
  -r output/bestModel/last.pth --test-only
```

Multi-GPU (example with 2 GPUs):
```bash
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr/rtdetr_v2_face.yaml --use-amp --seed 0
```

Override config fields with `-u key=value`, for example enabling/tuning NMS:
```bash
python tools/train.py -c configs/rtdetr/rtdetr_v2_face.yaml --use-amp -u \
  RTDETRPostProcessor.enable_nms=true RTDETRPostProcessor.nms_iou_threshold=0.6
```

## 4) Inference (Boxes + Keypoints)
Recommended script: `inference_visualize_all_faces.py`

```bash
python inference_visualize_all_faces.py \
  --image test_images/test01.jpg \
  --config configs/rtdetr/rtdetr_v2_face.yaml \
  --checkpoint output/bestModel/last.pth \
  --score-threshold 0.3 \
  --output test_images/test01_rtdetr.png
```

Alternative (minimal): `working_face_inference.py`
```bash
python working_face_inference.py path/to/image.jpg -o out.png --conf 0.5
```

Notes:
- The default config sets `enable_nms=true` in the postprocessor.
- If nothing is detected, try lowering `--score-threshold`.

## 5) Visualizing Keypoint Heatmaps (Polar/Cartesian)
Script: `tools/visualize_polar_heatmaps.py`

```bash
python tools/visualize_polar_heatmaps.py \
  --config configs/rtdetr/rtdetr_v2_face.yaml \
  --checkpoint output/bestModel/last.pth \
  --image test_images/test01.jpg \
  --output debug_heatmaps/example.png \
  --topk 1 --score-thr 0.2 --query-thr 0.5
```

Latest visualization combines the 5 keypoint heatmaps into a single overlayed subfigure with a legend for readability.

## 6) Checklist / FAQ
- “Image file not found”: Ensure `file_name` entries in `dataset/annotations/*.json` match files under `dataset/images/`. The loader removes the `WIDER_train/images/` prefix automatically.
- “CUDA error / GPU not visible”: Install a PyTorch build that matches your CUDA/runtime.
- “No keypoints at inference”: Use a config with `use_keypoints: true` and a checkpoint trained with keypoints.
- “No detections”: Lower `--score-threshold` or keep NMS enabled to reduce duplicates.

## 7) Key Scripts / Files
- Main config: `configs/rtdetr/rtdetr_v2_face.yaml`
- Train / Evaluate: `tools/train.py`
- Inference with visualization (boxes + keypoints): `inference_visualize_all_faces.py`, `working_face_inference.py`
- Heatmap visualization: `tools/visualize_polar_heatmaps.py`
- Dataset loader used in training: `src/data/widerface_keypoints_working.py`

If you need additional helper scripts or examples (e.g., batch inference on a folder), feel free to ask.


