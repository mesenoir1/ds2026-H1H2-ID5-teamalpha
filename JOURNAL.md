## KW18 - 30 April 2026

### Decision: Initial project setup
We created the private repository for Topic H1+H2: XAI-Guided Training for Knee Osteoarthritis Grading. The project investigates whether saliency maps can identify knee X-ray cases where a CNN predicts the correct Kellgren–Lawrence grade for the wrong reason, and whether those cases can guide improved training.

### Decision: Dataset plan
We will use provided SilpaCS/kneeosteoarthritis dataset from Hugging Face. As a first step, we will inspect the dataset structure, class distribution across KL grades 0–4, image dimensions, and possible artifacts such as borders, cropping effects, or scanner marks.

## KW19 - 8 May 2026

### Decision: Dataset setup and inspection

We downloaded and extracted the dataset. We generated a class distribution plot and one sample image per class. Inspection is needed because the documentation states that the dataset has an imbalanced class distribution and pre-cropped images, that can affect model evaluation and saliency-map interpretation.

The initial audit showed that sampled images are consistently 224×224 pixels. The observed class distribution is imbalanced, with KL grade 0 being the largest class and KL grade 4 being the rarest. 

Next steps: We will use stratified splitting. Considering imbalance: metrics such as macro-F1, balanced accuracy, and per-class recall instead of just accuracy.


## KW19 - 10 May 2026

### Decision: Local Tests, Bug-Fixes and Data Splitter

We tested the current scrips locally and fixed a few minor issues. 
We added a data splitter for training. As mentioned previously the dataset is high imbalanced, so we added stratified splitting.
To keep track of the behavior we added a few print statements showing distribution after splitting.

### Decision: Checked GPU access to the cluster

We tested GPU availability.

### Decision: Refactoring Data Split Logic, extending Split and add csv File

Moved the data splitting logic from dataset.py into a separate split.py to ensure cluster compatibility.
Adjusted the split to Train (70%), Validation (15%), and Test (15%). The split is saved as dataset_split.csv in Figures Directory for reproducibility.

## KW20 - 11 May 2026

### Decision: Baseline model selection

We decided to train 4 baseline model variants for KL-grade classification with following work division:
- Sirisha and Vladyslava:
1. ResNet with standard cross-entropy loss
2. ResNet with weighted cross-entropy loss
- Jan, Zumrud, Elina:
3. DenseNet with standard cross-entropy loss
4. DenseNet with weighted cross-entropy loss.

This decision was based on the reference paper by Choi et al. (2025), which compared DenseNet201, ResNet101, and EfficientNetV2 for knee osteoarthritis KL-grade classification. The paper reported that DenseNet201 achieved the strongest overall performance, while ResNet101 served as a relevant comparison architecture. However, unlike the balanced dataset used in the reference paper, our SilpaCS/kneeosteoarthritis dataset has an imbalanced KL-grade distribution. Therefore, we decided to evaluate each architecture both with and without weighted cross-entropy loss to test whether class weighting improves minority-class performance.

After training, we will evaluate the four baseline variants.