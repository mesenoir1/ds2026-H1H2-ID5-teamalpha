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
│   └── suspicious_cases_dynamic.py
│
├── data/
|   ├── analysis
│   ├── kneeosteoarthritis/
│   ├── splits/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └──eval/
├── jobs/
│
├── outputs/
|   ├── data_quality/
│   ├── densenet_ce/
│   ├── densenet_weighted_ce/
│   ├── gradcam/
|   ├── models_try/
│   └── xai_region_analysis_dynamic_fixed_roi/
│
├── report/
│   └── figures/
|       └──models_try/
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
A detailed note on AI-assisted development of intervention-training scripts is documented in `JOURNAL.md`.
All intervention-model training scripts are stored in:

```text
code/models_try/
```
These scripts were trained on the SIC HPC cluster.

### 10. Model evaluation

After training, all models are evaluated on the held-out test split using the evaluation script.

Most standard DenseNet201 models are evaluated with:

```bash
python code/evaluate_models.py
```
Before running the script, update the model list inside code/evaluate_models.py:
```bash
MODELS_TO_EVALUATE = [
    "densenet_weighted_ce",
    "m2_noinv_blur",
    "m2_inv_blur",
    "m2_noinv_blur_darken",
    "m2_noinv_darken",
    "m2_noinv_blur_suppression75",
    "m2_noinv_blur_crop_noise",
    "m3_suspicious05_noinv_blur",
]
```

Also can be run in a cluster.
```bash
condor_submit jobs/submit_eval.sub
```
or

```bash
condor_submit jobs/submit_eval_loss.sub
```

Expected output:
```text
data/eval/eval_<model_name>.csv
data/eval/summary_<model_name>.csv
data/eval/predictions_<model_name>.csv
data/eval/model_comparison.csv
report/figures/cm__<model_name>.png
```
For ROI-loss / explanation-loss models, use the separate evaluator because these models may not use the same checkpoint structure or forward pass as the plain DenseNet201 classifier:
```bash
python code/evaluate_roi_loss.py
```
The final comparison table in the report and journal is based on the generated summary_<model_name>.csv files.

## Cluster configurations and workflow

The final XAI-guided models were trained on the SIC HPC cluster using HTCondor
with Docker. The cluster run script calls the unified master training script
instead of separate intervention-specific scripts. The standard Docker image
was:

```bash
pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

Most training jobs used the following resource configuration:

```text
request_GPUs   = 1
request_CPUs   = 4
request_memory = 16G
```
Experiments are designed for a Linux-based GPU cluster. The example below shows
how to train the final selected model configuration:

```text
Weighted XAI loss
lambda_roi = 0.9
alpha_max = 0.0
without Dynamic Label Smoothing
seeds 39-45
```

### Example run script

Create a file such as:

```text
jobs/run_final_weighted_xai.sh
```

with the following content:

```bash
#!/bin/bash
set -euo pipefail

SEED="${1:-39}"
SUBMISSION_ID="${2:-local}"
MODEL_NAME="weighted_xai_roi09_a0_seed${SEED}_${SUBMISSION_ID}"
OUTPUT_DIR="outputs/${MODEL_NAME}"

TRAIN_CSV="outputs/xai_region_analysis_dynamic_fixed_roi/densenet_weighted_ce/train_denseblock4_predicted/all_cases_dynamic_roi_scores.csv"
VAL_CSV="outputs/xai_region_analysis_dynamic_fixed_roi/densenet_weighted_ce/val_denseblock4_predicted/all_cases_dynamic_roi_scores.csv"

cd /home/dsbwl26_team005/ds2026-H1H2-ID5-teamalpha

echo "=== Job started: ${MODEL_NAME} ==="
date
echo "PWD: $(pwd)"
echo "Host: $(hostname)"

export PYTHONNOUSERSITE=1
VENV_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}/venv_${MODEL_NAME}"
export PYTHONUSERBASE="${_CONDOR_SCRATCH_DIR:-/tmp}/python_user_${MODEL_NAME}"
export PIP_CACHE_DIR="${_CONDOR_SCRATCH_DIR:-/tmp}/pip_cache_${MODEL_NAME}"

echo "=== Create clean Python environment ==="
python -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

echo "=== CUDA check ==="
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "=== Install dependencies ==="
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir --ignore-installed opencv-python-headless

echo "=== Start final XAI-guided master-script training ==="
python code/densenet_train_master.py \
  --train-csv "${TRAIN_CSV}" \
  --val-csv "${VAL_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --attention-mode weighted \
  --max-alpha 0.0 \
  --lambda-border-base 0.0 \
  --lambda-roi-base 0.9 \
  --roi-loss-threshold 0.40 \
  --border-loss-threshold 0.20 \
  --flip-prob 0.5 \
  --invert-prob 0.5 \
  --noise-std 0.0 \
  --suppression-prob 0.5 \
  --blur-kernel 31 \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed "${SEED}"

echo "=== Job finished ==="
date
```

This writes the checkpoint and metadata to:

```text
outputs/weighted_xai_roi09_a0_seed<seed>_<cluster-id>/
```

### Example HTCondor submit file

Create a file such as:

```text
jobs/submit_final_weighted_xai_seeds.sub
```

with the following content:

```bash
universe                = docker
docker_image            = pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

executable              = jobs/run_final_weighted_xai.sh
arguments               = $(SEED) $(ClusterId)

output                  = runlogs/final_weighted_xai_seed$(SEED).$(ClusterId).$(ProcId).out
error                   = runlogs/final_weighted_xai_seed$(SEED).$(ClusterId).$(ProcId).err
log                     = runlogs/final_weighted_xai_seeds.$(ClusterId).log

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT

request_GPUs            = 1
request_CPUs            = 4
request_memory          = 16G

requirements            = UidDomain == "cs.uni-saarland.de"
+WantGPUHomeMounted     = true
+WantScratchMounted     = true

queue SEED from (
39
40
41
42
43
44
45
)
```

### Submit and monitor

```bash
# 1. SSH into the cluster

# 2. Go to the project directory

# 3. Submit the final multi-seed training job
condor_submit jobs/submit_final_weighted_xai_seeds.sub

# 4. Monitor the job
condor_q

# 5. Inspect logs after completion
ls -la runlogs/
```
## Reproducibility Notes

- Dataset splits are saved as CSV files under `data/splits/`.
- Model checkpoints and training metadata are saved under `outputs/`.
- Grad-CAM runs save `gradcam_config.json`.
- Suspicious-case thresholds are saved in `suspicious_summary.csv`.
- Test-set explanations should only be used for final evaluation and should not guide training interventions.

## Prototype Requirements

The repository includes a local Streamlit prototype for interactive inspection of
the baseline and final XAI-guided knee OA models.

### Required Prototype Files

The prototype-specific files that must be included are:

```text
app.py
prototype_utils.py
requirements.txt
```

The prototype reuses the existing implementation in `code/`. The following
modules are required for preprocessing, model construction, Grad-CAM, dynamic
ROI masks, and proxy saliency metrics:

```text
code/densenet_dataset.py
code/evaluate_faithfulness.py
code/generate_gradcam.py
code/roi_dynamic.py
code/suspicious_cases_dynamic.py
```

### Required Model Folders

The prototype expects the model checkpoints in these folders:

```text
baseline_39_45/
final_models_39_45/
```

Each seed/model folder should contain:

```text
best_model.pt
config.json
training_history.csv
```

The app automatically discovers checkpoints matching:

```text
baseline_39_45/**/best_model.pt
final_models_39_45/**/best_model.pt
```

Because `best_model.pt` files are large, they should be tracked with Git LFS.
The repository should therefore also include:

```text
.gitattributes
```

### Running the Prototype

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the Streamlit app from the project root:

```powershell
streamlit run app.py
```