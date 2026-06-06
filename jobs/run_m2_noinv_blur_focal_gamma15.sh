#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: M2 no-inversion ROI blur + weighted focal loss ==="
date
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

echo "=== Files ==="
ls -la
ls -la code
ls -la data/splits
ls -la outputs || true

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

echo "=== Start training: M2 no-inversion ROI blur + weighted focal loss ==="
python code/densenet_train_m2_noinv_blur_focal.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --output-dir outputs/m2_noinv_blur_focal_gamma15 \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42 \
  --suppression-prob 0.5 \
  --blur-kernel 21 \
  --gamma 1.5

echo "=== Training outputs ==="
ls -la outputs/m2_noinv_blur_focal_gamma15 || true

echo "=== Config ==="
cat outputs/m2_noinv_blur_focal_gamma15/config.json || true

echo "=== Training history tail ==="
tail -5 outputs/m2_noinv_blur_focal_gamma15/training_history.csv || true

echo "=== Job finished: M2 no-inversion ROI blur + weighted focal loss ==="
date