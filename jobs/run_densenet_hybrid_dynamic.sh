#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: Hybrid Model (Blur + Aug + Dynamic Loss) ==="
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

echo "=== Start Training ==="
#--resume-weights outputs/pfad/best_model.pt and --lr 5e-5 for continuation
python code/densenet_train_hybrid_dynamic.py \
  --train-csv outputs/xai_region_analysis_dynamic_fixed_roi/densenet_weighted_ce/train_denseblock4_predicted/all_cases_dynamic_roi_scores.csv \
  --val-csv outputs/xai_region_analysis_dynamic_fixed_roi/densenet_weighted_ce/val_denseblock4_predicted/all_cases_dynamic_roi_scores.csv \
  --output-dir outputs/densenet_hybrid_dynamic \
  --max-alpha 0.5 \
  --lambda-border-base 0.1 \
  --lambda-roi-base 0.1 \
  --noise-std 0.05 \
  --suppression-prob 0.5 \
  --blur-kernel 21 \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --seed 42

echo "=== Job finished ==="
date