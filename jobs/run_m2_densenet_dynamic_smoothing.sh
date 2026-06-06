#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: DenseNet Dynamic Label Smoothing ==="
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

echo "=== Start Training: Dynamic Label Smoothing ==="
# ACHTUNG: Passe hier ggf. die Pfade zu deinen CSV-Dateien an, die die Suspicious-Scores enthalten!
python code/densenet_train_dynamic_smoothing.py \
  --train-csv outputs/xai_region_analysis_dynamic_fixed_roi/densenet_weighted_ce/train_denseblock4_predicted/all_cases_dynamic_roi_scores.csv \
  --val-csv outputs/xai_region_analysis_dynamic_fixed_roi/densenet_weighted_ce/val_denseblock4_predicted/all_cases_dynamic_roi_scores.csv \
  --output-dir outputs/densenet_dynamic_smoothing \
  --max-alpha 0.5 \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --seed 42

echo "=== Job finished: DenseNet Dynamic Label Smoothing ==="
date
