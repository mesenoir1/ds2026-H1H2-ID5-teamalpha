# XAI-Guided Training for Knee Osteoarthritis Grading

## Project Title
H1 + H2: XAI-Guided Training for Knee Osteoarthritis Grading

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
```bash
python code/densenet_train_weighted_ce.py
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
python code/post_gradcam_region_analysis_final.py \
  --split-csv data/splits/val.csv \
  --predictions-csv outputs/gradcam/densenet_weighted_ce/val_denseblock4_predicted/gradcam_predictions.csv \
  --heatmap-dir outputs/gradcam/densenet_weighted_ce/val_denseblock4_predicted/heatmaps \
  --output-dir outputs/xai_region_analysis/densenet_weighted_ce/val_denseblock4_predicted_final \
  --only-correct \
  --suspicion-quantile 0.20 \
  --border-quantile 0.80 \
  --min-low-methods 2 \
  --topk-fraction 0.10 \
  --num-debug-figures 100
```

For train split:

```bash
python code/post_gradcam_region_analysis_final.py \
  --split-csv data/splits/train.csv \
  --predictions-csv outputs/gradcam/densenet_weighted_ce/train_denseblock4_predicted/gradcam_predictions.csv \
  --heatmap-dir outputs/gradcam/densenet_weighted_ce/train_denseblock4_predicted/heatmaps \
  --output-dir outputs/xai_region_analysis/densenet_weighted_ce/train_denseblock4_predicted_final \
  --only-correct \
  --suspicion-quantile 0.20 \
  --border-quantile 0.80 \
  --min-low-methods 2 \
  --topk-fraction 0.10 \
  --num-debug-figures 100
```

Expected output:
```text
outputs/xai_region_analysis/densenet_weighted_ce/
├── train_denseblock4_predicted_final/
└── val_denseblock4_predicted_final/
```

Each analysis folder contains:

```text
region_scores.csv
region_method_summary.csv
all_cases_with_suspicion_flags.csv
suspicious_cases.csv
suspicious_summary.csv
suspicion_reason_summary.csv
post_gradcam_config.json
skipped_rows.csv
debug_figures/
```


## Cluster Usage

Experiments are designed for a Linux-based GPU cluster.

GPU jobs are used for:

```text
model training
Grad-CAM generation
```

## Reproducibility Notes

- Dataset splits are saved as CSV files under `data/splits/`.
- Model checkpoints and training metadata are saved under `outputs/`.
- Grad-CAM runs save `gradcam_config.json`.
- Region-analysis runs save `post_gradcam_config.json`.
- Suspicious-case thresholds are saved in `suspicious_summary.csv`.
- Test-set explanations should only be used for final evaluation and should not guide training interventions.