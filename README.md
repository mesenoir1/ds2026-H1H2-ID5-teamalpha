# XAI-Guided Training for Knee Osteoarthritis Grading

## Project Title
H1 + H2: XAI-Guided Training for Knee Osteoarthritis Grading

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




## How to Run the Project

### 1. Install packages

```bash
pip install torch torchvision scikit-learn matplotlib pillow huggingface_hub
```

### 2. Dataset

```bash
dataset_download.py
```

This script downloads the `SilpaCS/kneeosteoarthritis` dataset from Hugging Face and extracts it into:

```text
data/kneeosteoarthritis/
```

### 3. Inspection

```bash
dataset.py
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
split.py
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
