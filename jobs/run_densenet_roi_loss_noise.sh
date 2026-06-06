#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: DenseNet ROI guided attention loss + Gaussian noise ==="
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

echo "=== Start DenseNet ROI guided attention loss + Gaussian noise training ==="
python code/densenet_train_roi_loss_noise.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --output-dir outputs/densenet_roi_loss_gaussian_noise \
  --epochs 30 \
  --batch-size 4 \
  --lr 1e-4 \
  --num-workers 0 \
  --seed 42 \
  --lambda-border 0.0 \
  --lambda-roi 0.1 \
  --noise-std 0.05 \
  --noise-prob 0.5

echo "=== Training outputs ==="
ls -la outputs/densenet_roi_loss_gaussian_noise || true

echo "=== Config ==="
cat outputs/densenet_roi_loss_gaussian_noise/config.json || true

echo "=== Training history tail ==="
tail -5 outputs/densenet_roi_loss_gaussian_noise/training_history.csv || true

echo "=== Job finished: DenseNet ROI guided attention loss + Gaussian noise ==="
date