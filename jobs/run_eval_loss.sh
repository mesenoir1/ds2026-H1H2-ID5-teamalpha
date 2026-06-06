#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: evaluate ROI-loss model ==="
date
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

echo "=== Files ==="
ls -la
ls -la code
ls -la data/splits
ls -la outputs/densenet_roi_loss || true

echo "=== CUDA check ==="
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "=== Install dependencies ==="
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir opencv-python-headless

echo "=== OpenCV check ==="
python - <<'PY'
import cv2
print("cv2 imported successfully")
print(cv2.__version__)
PY

echo "=== Start ROI-loss evaluation on validation split ==="
python code/evaluate_roi_loss.py \
  --checkpoint outputs/m2_densenet_roi_loss_gaussian_noise_continued_noinv_blur/best_model.pt \
  --csv data/splits/val.csv \
  --model-name m2_densenet_roi_loss_gaussian_noise_continued_noinv_blur \
  --output-dir data/eval \
  --batch-size 8 \
  --num-workers 0

echo "=== Start ROI-loss evaluation on test split ==="
python code/evaluate_roi_loss.py \
  --checkpoint outputs/m2_densenet_roi_loss_gaussian_noise_continued_noinv_blur/best_model.pt \
  --csv data/splits/test.csv \
  --model-name m2_densenet_roi_loss_gaussian_noise_continued_noinv_blur \
  --output-dir data/eval \
  --batch-size 8 \
  --num-workers 0

echo "=== Evaluation outputs ==="
ls -la data/eval | grep roi_loss || true

echo "=== Job finished: evaluate m2_densenet_roi_loss_gaussian_noise_continued_noinv_blur==="
date