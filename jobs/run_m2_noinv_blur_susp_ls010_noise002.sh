#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: M2 no-inv blur + suspicious label smoothing + Gaussian noise ==="
date
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

echo "=== Files ==="
ls -la
ls -la code
ls -la data/splits
ls -la outputs || true

echo "=== Check suspicious-case input ==="
SUSPICIOUS_CSV="outputs/xai_region_analysis_dynamic/densenet_weighted_ce/train_denseblock4_predicted/suspicious_cases.csv"

if [ ! -f "$SUSPICIOUS_CSV" ]; then
  echo "ERROR: suspicious CSV not found: $SUSPICIOUS_CSV"
  echo "Available suspicious_cases.csv files:"
  find outputs/xai_region_analysis_dynamic -maxdepth 4 -type f -name "suspicious_cases.csv" || true
  exit 1
fi

echo "Using suspicious CSV: $SUSPICIOUS_CSV"
head -5 "$SUSPICIOUS_CSV" || true
wc -l "$SUSPICIOUS_CSV" || true

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

echo "=== Start training: M2 no-inv blur + suspicious LS 0.10 + Gaussian noise 0.02 ==="
python code/densenet_train_m2_noinv_blur_susp_ls_noise.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --suspicious-csv "$SUSPICIOUS_CSV" \
  --output-dir outputs/m2_noinv_blur_susp_ls010_noise002 \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42 \
  --suppression-prob 0.5 \
  --blur-kernel 21 \
  --suspicious-label-smoothing 0.10 \
  --noise-std 0.02 \
  --noise-prob 0.5

echo "=== Training outputs ==="
ls -la outputs/m2_noinv_blur_susp_ls010_noise002 || true

echo "=== Config ==="
cat outputs/m2_noinv_blur_susp_ls010_noise002/config.json || true

echo "=== Training history tail ==="
tail -5 outputs/m2_noinv_blur_susp_ls010_noise002/training_history.csv || true

echo "=== Job finished: M2 no-inv blur + suspicious label smoothing + Gaussian noise ==="
date