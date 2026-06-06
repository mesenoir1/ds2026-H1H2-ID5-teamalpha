#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: Continue DenseNet M2 Blur No-Inv ==="
date
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

echo "=== CUDA check ==="
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "=== Install dependencies ==="
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir opencv-python-headless

echo "=== Start Continuation Training ==="
python code/densenet_continue_m2_blur_noinv.py \
  --resume-weights outputs/densenet_roi_loss_gaussian_noise/best_model.pt \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --output-dir outputs/m2_densenet_roi_loss_gaussian_noise_continued_noinv_blur \
  --epochs 15 \
  --batch-size 8 \
  --lr 5e-5 \
  --suppression-prob 0.75 \
  --blur-kernel 21 \
  --seed 42

echo "=== Job finished ==="
date