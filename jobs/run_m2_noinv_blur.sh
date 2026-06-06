#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: M2 no-inversion blur ==="
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
pip install --no-cache-dir -r requirements.txt

echo "=== Start M2 training: no inversion + blur background suppression ==="
python code/densenet_train_m2_noinv_blur_border_loss_weighted_ce.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --output-dir outputs/m2_noinv_blur31 \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42 \
  --suppression-prob 0.50 \
  --blur-kernel 31

echo "=== Job finished: m2_noinv_blur31 ==="
date