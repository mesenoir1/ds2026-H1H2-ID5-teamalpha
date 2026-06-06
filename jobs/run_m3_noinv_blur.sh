#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: M3 suspicious-case targeted no-inversion blur ==="
date
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

echo "=== Files ==="
ls -la
ls -la code
ls -la data/splits

echo "=== Check suspicious-case input ==="
SUSPICIOUS_CSV="outputs/xai_region_analysis_dynamic/densenet_weighted_ce/train_denseblock4_predicted/suspicious_cases.csv"

if [ ! -f "$SUSPICIOUS_CSV" ]; then
  echo "ERROR: suspicious CSV not found: $SUSPICIOUS_CSV"
  echo "Available xai_region_analysis_dynamic folders:"
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

echo "=== Start M3 training: suspicious cases only + no-inversion ROI blur ==="
python code/densenet_train_m3_suspicious_noinv_blur_weighted_ce.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --suspicious-csv "$SUSPICIOUS_CSV" \
  --output-dir outputs/m3_suspicious1_noinv_blur \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42 \
  --suspicious-suppression-prob 1.0 \
  --blur-kernel 21

echo "=== Training outputs ==="
ls -la outputs/m3_suspicious1_noinv_blur || true

echo "=== Config ==="
cat outputs/m3_suspicious1_noinv_blur/config.json || true

echo "=== Training history tail ==="
tail -5 outputs/m3_suspicious1_noinv_blur/training_history.csv || true

echo "=== Job finished: M3 suspicious-case1 targeted no-inversion blur ==="
date