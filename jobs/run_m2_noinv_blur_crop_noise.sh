#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: M2 no-inversion blur + crop + Gaussian noise ==="
date
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

echo "=== Files ==="
ls -la
ls -la code
ls -la data/splits

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

echo "=== Start M2 training: no inversion + blur + crop + Gaussian noise ==="
python code/densenet_train_m2_noinv_blur_crop_noise_weighted_ce.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --output-dir outputs/m2_noinv_blur_crop_noise \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42 \
  --suppression-prob 0.5 \
  --blur-kernel 21 \
  --crop-scale-min 0.90 \
  --crop-ratio-min 0.95 \
  --crop-ratio-max 1.05 \
  --noise-std 0.03 \
  --noise-prob 0.5

echo "=== Training outputs ==="
ls -la outputs/m2_noinv_blur_crop_noise || true

echo "=== Job finished: M2 no-inversion blur + crop + Gaussian noise ==="
date