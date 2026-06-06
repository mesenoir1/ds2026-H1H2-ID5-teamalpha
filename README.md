# XAI-Guided Training for Knee Osteoarthritis Grading

## Project Title
XAI-Guided Training for Knee Osteoarthritis Grading

RQ1: Can saliency maps identify knee X-ray cases where a CNN predicts the correct KL grade for the wrong reason?

RQ2: Can these cases then be used to improve both classification performance and saliency faithfulness?

## Project Overview

This project investigates whether predicted-class Grad-CAM can identify correctly classified knee X-ray cases where a CNN predicts the correct Kellgren–Lawrence grade using non-diagnostic image regions. The workflow trains a DenseNet201 baseline, generates Grad-CAM explanations, quantifies saliency overlap with diagnostic proxy regions, and flags suspicious correct predictions for later XAI-guided training interventions.

## Group Members
Jan Gindorf (7037280)

Sirisha Sandadi (7072365)

Elina Abdrashitova (7069012)

Zumrud Hasanova (7071133)

Vladyslava Semenova (7062334)


## Dataset
SilpaCS/kneeosteoarthritis

https://huggingface.co/datasets/SilpaCS/kneeosteoarthritis


## Research question
Can saliency maps identify knee X-ray cases where a CNN predicts the correct KL grade for the wrong reason, and can those cases be used to improve both classification performance and saliency faithfulness?

## Repository Structure

```text
.
├── code/
│   ├── dataset_download.py
│   ├── dataset.py
│   ├── split.py
│   ├── densenet_dataset.py
│   ├── densenet_train_ce.py
│   ├── densenet_train_weighted_ce.py
│   ├── evaluate_models.py
│   ├── generate_gradcam.py
│   └── post_gradcam_region_analysis_final.py
│
├── data/
│   ├── kneeosteoarthritis/
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── eval/
│
├── outputs/
│   ├── densenet_ce/
│   ├── densenet_weighted_ce/
│   ├── gradcam/
│   └── xai_region_analysis/
│
├── report/
│   └── figures/
│
├── runlogs/
├── JOURNAL.md
├── README.md
└── requirements.txt

```

## How to Run the Project

### 1. Install packages

```bash
pip install -r requirements.txt
```

### 2. Dataset

```bash
python code/dataset_download.py
```

This script downloads the `SilpaCS/kneeosteoarthritis` dataset from Hugging Face and extracts it into:

```text
data/kneeosteoarthritis/
```

### 3. Inspection

```bash
python code/dataset.py
```

This script checks the dataset structure, counts the number of images per KL grade, samples image sizes, and generates basic figures.


```text
report/figures/
```

Expected outputs include:

```text
class_distribution.png
sample_images_by_class.png
```

### 4. Splits

```bash
python code/split.py
```

This script creates stratified train/validation/test splits using a 70/15/15 ratio. 

Generated CSV files are saved in:

```text
data/splits/
```

Expected output files:

```text
train.csv
val.csv
test.csv
```

### 5. Train DenseNet201 with weighted cross-entropy baseline

The official baseline model is **B1: DenseNet201 with weighted cross-entropy**. It was trained on the SIC HPC cluster using HTCondor and Docker. The training script is:

```text
code/densenet_train_weighted_ce.py
```

For the full cluster workflow, including Docker image, dependency installation, GPU request, and HTCondor submit-file template, see:
```text
Cluster configurations and workflow
```
Expected outputs:

```text
outputs/densenet_weighted_ce/best_model.pt
outputs/densenet_weighted_ce/training_history.csv
outputs/densenet_weighted_ce/config.json
```

### 6. Evaluate model
```bash
python code/evaluate_models.py
```

Expected outputs:

```text
data/eval/
report/figures/
```

### 7. Generate Grad-CAM Explanations
For validation splt:
```bash
python code/generate_gradcam.py \
  --csv data/splits/val.csv \
  --checkpoint outputs/densenet_weighted_ce/best_model.pt \
  --output-dir outputs/gradcam/densenet_weighted_ce/val_denseblock4_predicted \
  --split val \
  --batch-size 1 \
  --num-workers 4 \
  --target predicted
```

For training split:
```bash
python code/generate_gradcam.py \
  --csv data/splits/train.csv \
  --checkpoint outputs/densenet_weighted_ce/best_model.pt \
  --output-dir outputs/gradcam/densenet_weighted_ce/train_denseblock4_predicted \
  --split train \
  --batch-size 1 \
  --num-workers 4 \
  --target predicted
```

Expected outputs:
```text
outputs/gradcam/densenet_weighted_ce/
├── train_denseblock4_predicted/
│   ├── heatmaps/
│   ├── originals/
│   ├── overlays/
│   ├── gradcam_config.json
│   └── gradcam_predictions.csv
└── val_denseblock4_predicted/
    ├── heatmaps/
    ├── originals/
    ├── overlays/
    ├── gradcam_config.json
    └── gradcam_predictions.csv
```

### 8. Suspicious cases detection

For validation split:

```bash
python code/suspicious_cases_dynamic.py \
  --gradcam-csv outputs/gradcam/densenet_weighted_ce/val_denseblock4_predicted/gradcam_predictions.csv \
  --inversion-qc-csv outputs/data_quality/inversion_qc.csv \
  --output-dir outputs/xai_region_analysis_dynamic_fixed_roi/densenet_weighted_ce/val_denseblock4_predicted \
  --roi-threshold 0.40 \
  --border-threshold 0.20 \
  --border-frac 0.08 \
  --num-debug-figures 25
```


For train split:

```bash
python code/suspicious_cases_dynamic.py \
  --gradcam-csv outputs/gradcam/densenet_weighted_ce/train_denseblock4_predicted/gradcam_predictions.csv \
  --inversion-qc-csv outputs/data_quality/inversion_qc.csv \
  --output-dir outputs/xai_region_analysis_dynamic_fixed_roi/densenet_weighted_ce/train_denseblock4_predicted \
  --roi-threshold 0.40 \
  --border-threshold 0.20 \
  --border-frac 0.08 \
  --num-debug-figures 25
```

Or can be produces on the cluster:
Adjust for wich split the Grad_CAM is produced, the model, and output directory in 
```text
jobs/run_gradcam.sh
```
Submitting the job:
```bash
condor_submit jobs/submit_gradcam.sub
```

Expected output:
```text
outputs/xai_region_analysis_dynamic/densenet_weighted_ce/
├── train_denseblock4_predicted/
└── val_denseblock4_predicted/
```

Each analysis folder contains:

```text
all_cases_dynamic_roi_scores.csv
suspicious_cases.csv
suspicious_summary.csv
suspicion_reason_summary.csv
classwise_suspicious_summary.csv
debug_figures/
├── suspicious/
├── low_roi/
├── high_border/
├── non_suspicious_correct/
└── likely_inverted/
```
### 9. Intervention training
All intervention-model training scripts are stored in:

```text
code/models_try/
```
These scripts were trained on the SIC HPC cluster.

## Cluster configurations and workflow

These scripts were trained on the SIC HPC cluster using HTCondor with Docker. The standard Docker image was:

```bash
pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

Most training jobs used the following resource configuration:

```text
request_GPUs   = 1
request_CPUs   = 4
request_memory = 16G
```
Experiments are designed for a Linux-based GPU cluster.

GPU jobs are used for:

```bash
# 1. SSH into the cluster
ssh <username>@conduit.hpc.uni-saarland.de

# 2. Go to the project directory
cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

# 3. Prepare a run script for the selected model
# Example:
# jobs/run_m2_noinv_blur.sh

# 4. Prepare a matching HTCondor submit file
# Example:
# jobs/submit_m2_noinv_blur.sub

# 5. Submit the job
condor_submit jobs/submit_m2_noinv_blur.sub

# 6. Monitor the job
condor_q

# 7. Inspect logs after completion
ls -la runlogs/
```
A typical run script:
```bash
#!/bin/bash
set -euo pipefail

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started ==="
date
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

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

echo "=== Start training ==="
python code/models_try/<training_script>.py \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --output-dir outputs/<model_name> \
  --epochs <epochs> \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42

echo "=== Job finished ==="
date
```

A typical HTCondor submit file:
```bash
universe                = docker
docker_image            = pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

executable              = jobs/run_<model_name>.sh

output                  = runlogs/<model_name>.$(ClusterId).$(ProcId).out
error                   = runlogs/<model_name>.$(ClusterId).$(ProcId).err
log                     = runlogs/<model_name>.$(ClusterId).log

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT

request_GPUs            = 1
request_CPUs            = 4
request_memory          = 16G

requirements            = UidDomain == "cs.uni-saarland.de"
+WantGPUHomeMounted     = true
+WantScratchMounted     = true

queue 1
```
## Reproducibility Notes

- Dataset splits are saved as CSV files under `data/splits/`.
- Model checkpoints and training metadata are saved under `outputs/`.
- Grad-CAM runs save `gradcam_config.json`.
- Region-analysis runs save `post_gradcam_config.json`.
- Suspicious-case thresholds are saved in `suspicious_summary.csv`.
- Test-set explanations should only be used for final evaluation and should not guide training interventions.