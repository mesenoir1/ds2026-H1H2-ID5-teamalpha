#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: Grad-CAM validation for m2_noinv_blur ==="
date
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

echo "=== Files ==="
ls -la
ls -la code
ls -la data/splits
ls -la outputs/m2_noinv_blur_darken || true

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

echo "=== Start Grad-CAM generation: validation split, predicted class, denseblock4 ==="
python code/generate_gradcam.py \
  --csv data/splits/val.csv \
  --checkpoint outputs/m2_noinv_blur_loss_borders/best_model.pt \
  --output-dir outputs/gradcam/m2_noinv_blur_loss_borders/val_denseblock4_predicted \
  --target predicted

echo "=== Grad-CAM outputs ==="
ls -la outputs/gradcam/m2_noinv_blur_loss_borders/val_denseblock4_predicted || true
ls -la outputs/gradcam/m2_noinv_blur_loss_borders/val_denseblock4_predicted/heatmaps || true

echo "=== Job finished: Grad-CAM validation for m2_noinv_blur_loss_borders ==="
date