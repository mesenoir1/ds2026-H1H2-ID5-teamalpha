#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha_resNet"

cd "$PROJECT_DIR"

echo "=== ResNet job started ==="
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

echo "=== Start ResNet101 training ==="
python -u code/model_weighted.py

echo "=== ResNet job finished ==="
date