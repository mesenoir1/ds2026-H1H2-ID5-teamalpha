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


## KW21 - 18 May 2026

### Decision: Baseline training

We trained 4 CNN baseline variants for KL-grade classification:
1. ResNet with standard cross-entropy loss
2. ResNet with weighted cross-entropy loss
3. DenseNet201 with standard cross-entropy loss
4. DenseNet201 with weighted cross-entropy loss

The image preprocessing pipeline converted grayscale knee X-ray images into 3-channel tensors because ImageNet-pretrained CNN backbones expect RGB-style input. Images were resized to 224×224 pixels. For training, we used random horizontal flipping as a simple data augmentation step. For validation and testing, we used deterministic preprocessing without augmentation. Images were normalized using ImageNet mean and standard deviation values because the CNN backbones were initialized with ImageNet-pretrained weights.

We used macro-F1, macro precision, macro recall, and accuracy for evaluation.

## KW21 - 19 May 2026


### Decision: DenseNet201 with weighted cross-entropy loss as a baseline model

For the DenseNet branch, each image is loaded as grayscale and converted to 3 channels before being passed into the model. This allows the use of ImageNet-pretrained DenseNet201 while preserving the grayscale medical image content.

The DenseNet201 classifier head was replaced with a linear layer producing five output classes, corresponding to KL grades 0–4. We trained two DenseNet variants: one with standard cross-entropy loss and one with weighted cross-entropy loss.

The weighted cross-entropy variant computes class weights from the training split using the inverse class frequency formula:

`class_weight = number_of_training_samples / (number_of_classes × class_count)`

This was done to reduce the dominance of majority classes during optimization.

For both DenseNet variants, the model checkpoint was selected using the best validation macro-F1 score rather than the final training epoch. This was necessary because later epochs showed signs of overfitting.


## KW21 - 20 May 2026

### Decision: Baseline model selection after evaluation

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| DenseNet201 + CE | 0.6877 | 0.6795 | 0.6994 | 0.6858 | 
| DenseNet201 + weighted CE | 0.6610 | 0.6907 | 0.6863 | 0.6870 |
| ResNet + CE | 0.6723 | 0.6775 | 0.6698 | 0.6716 |
| ResNet + weighted CE | 0.6489 | 0.6908 | 0.6554 | 0.6696 | 

We selected DenseNet201 with weighted cross-entropy as the primary baseline for the XAI analysis. Although DenseNet201 with standard cross-entropy achieved the highest accuracy, DenseNet201 with weighted cross-entropy achieved the highest macro-F1. Since the dataset is imbalanced, we prioritized macro-F1 over just accuracy. 

The selected DenseNet weighted CE model will be used as the baseline for H1.

## KW22 - 25 May 2026

### Decision: Grad-CAM

We implemented Grad-CAM for the selected baseline CNN. This step is required for H1 because we need to identify cases where the model predicts the correct KL grade but appears to rely on visually non-diagnostic regions. Grad-CAM was chosen because it is compatible with CNN backbones such as DenseNet and provides localized activation maps that can be compared against image regions expected to contain diagnostically relevant knee anatomy.

For each case, we will store the input image, true label, predicted label, prediction probability, Grad-CAM heatmap, and overlay visualization.

The next step is to define a prediction-relevant diagnostic region that can be used. This region will allow us to quantify how much model attention falls inside versus outside the expected knee joint area.

### Decision: Methods for defining prediction-relevant regions

We decided to test three possible approaches for identifying the prediction-important or diagnostically relevant image region:

1. **Segmentation**  
   We will test whether a segmentation-based approach can isolate the knee joint or relevant bone/joint-space structure. This is the most direct option if it produces stable masks, but it may require additional implementation effort and could fail if the X-ray images are too variable or low-contrast.

2. **Canny edge detection**  
   We will test whether edge detection can provide a lightweight anatomical proxy by detecting strong structural boundaries in the knee X-rays. This approach is computationally cheap, but it may also capture non-diagnostic edges such as borders, artifacts, or cropping boundaries.

3. **Downsampling / quantization**  
   We will test whether reducing image resolution or quantizing spatial regions can produce a robust coarse region-of-interest estimate. This may be useful if exact anatomical segmentation is unreliable, because the goal is not perfect medical segmentation but a reproducible proxy for comparing saliency concentration across regions.

We will compare these approaches based on mask stability, interpretability, failure rate, and usefulness. The selected method will be used to define the unfaithful-case selection rule for H1.

### Next step

The next step is to run qualitative and quantitative tests for the three region-definition methods. Based on the method that produces the most reliable and interpretable masks, we will define a fixed scoring rule and use it to identify the H1 subset.

### Decision: Experiments for defining prediction-relevant regions and reduce biases relying on device artefacts

The Grad-CAM maps revealed that our model bases its decisions on regions that should not contain any relevant information, such as the black background areas in the X-ray images.
Therefore we discussed (see above) several methods to improve the baseline.  
1. **Segmentation** 
   Here we decide to use Anatomical Cropping (ROI Extraction) Tiulpin et al. (2018) and Antony et al. (2017)
   This method simply crops the black background, s.t. the model is forced to look at the bones

2. **Gaussian Noise Augumentation**
   Instead of downsampling we chose a gaussian noise augumentation to cancel/average out unvisible device artefacts.
Gaussian noise injection is an established adversarial augmentation technique in medical imaging that prevents convolutional neural networks from memorizing scanner-specific noise profiles as shortcuts. By artificially introducing high-frequency noise during training, the model is forced to ignore these artifacts and instead learn robust, low-frequency biological features such as joint structure and bone morphology.

Overall the Grad-Cam heatmaps now give more reasonable and smaller areas.
The Background is becoming significantly less part of the prediction, however its not totally avoided.

## Next step
This experiments were only done in a sandbox, not yet optimized. The train seems to need more epochs till convergence than the previous baseline, but we took the same amount.
We aloso didnt do experiments with different hyperparameters on that augumentation

## KW22 - 28 May 2026

### Decision: Continue crop/noise experiments

After the previous sandbox experiments, we decided to further investigate the crop + Gaussian noise setup instead of immediately replacing the baseline. The current official baseline remains **DenseNet201 with weighted cross-entropy**, because it is the model used for the first Grad-CAM-based failure analysis. The crop + Gaussian noise model will be treated as an intervention candidate rather than as the new baseline.

---

### Decision: Candidate methods for defining prediction-relevant anatomical regions

We explored several heuristic methods for defining a diagnostically plausible region in the knee X-ray images. Because we do not have pixel-level anatomical annotations, these masks are not treated as ground truth segmentations. Instead, they are reproducible proxies for measuring whether Grad-CAM activation is broadly inside the expected knee/joint region or outside it.

The explored methods were:

1. **Central rectangle**  
   A fixed rectangular region in the center of the image. This was considered too simple because it includes too much irrelevant area and does not match the anatomical shape well.

2. **Central ellipse**  
   A fixed elliptical region centered around the expected knee anatomy. This is simple, stable, and anatomically more plausible than a rectangle.

3. **Central joint-band mask**  
   A narrower horizontal band inside the central knee region, intended to approximate the tibiofemoral joint-space area. This is clinically relevant because KL grading depends strongly on joint-space narrowing and nearby bone changes.

4. **Otsu central mask**  
   A threshold-based mask using Otsu binarization restricted to the central region. Initial inspection suggested that it was unstable and often selected structures that were not reliable diagnostic proxies.

5. **Inverted Otsu central mask**  
   An inverted threshold-based central mask. It produced very small masks with high enrichment but low absolute Grad-CAM coverage, so we decided not to use it in the final suspicious-case rule.

6. **Canny central mask**  
   An edge-based mask restricted to the center. This was rejected because Canny produces thin edge contours, while Grad-CAM is a coarse regional heatmap; therefore, overlap scores are not well matched.

7. **Quantized central mask**  
   A coarse image-adaptive mask based on downsampling/quantization and central restriction. This method is more adaptive than a fixed ellipse while being more stable than raw thresholding.

8. **Border mask**  
   A non-diagnostic mask covering the outer image border. This is not used as an anatomical region; instead, it measures how much Grad-CAM activation falls into peripheral areas where shortcut cues or artifacts may appear.

## Next step

We decided to continue tests with three diagnostic-region proxies (central_ellipse, quantized_central, central_joint_band), while still exploring some other methods.

## KW22 - 31 May 2026

### Decision: Suspicious-case detection using diagnostic Grad-CAM region scores

We completed the first full suspicious-case detection run for the DenseNet201 + weighted cross-entropy baseline. The goal was to identify correctly classified knee X-ray cases where the model may have predicted the correct KL grade while relying less on diagnostically relevant image regions.

We used predicted-class Grad-CAM because the research question asks why the model made its actual prediction. The current Grad-CAM layer is DenseNet201 `denseblock4`.

A case was only considered for suspicious-case detection if it was classified correctly. A correctly classified case was flagged as suspicious if either of the following conditions was true:

1. At least 2 of the 3 selected diagnostic masks had low inside-mask Grad-CAM activation.
2. Border attention was high.

The selected diagnostic proxy masks were:

- `central_ellipse`
- `quantized_central`
- `central_joint_band`

The additional non-diagnostic attention metric was:

- `border_attention_score`

Configuration:

```text
model = DenseNet201 + weighted cross-entropy
Grad-CAM target = predicted class
Grad-CAM layer = DenseNet201 denseblock4
only_correct = true
suspicion_quantile = 0.20
border_quantile = 0.80
min_low_methods = 2
topk_fraction = 0.10
```

The suspicious-case detection was run on correctly classified training cases.

```text
Correct training cases evaluated: 4454
Suspicious training cases: 971
Suspicious fraction: 21.8%
Low-region-attention cases: 895
High-border-attention cases: 891
```
Threshhold:

```text
central_ellipse_inside_ratio <= 0.5496
quantized_central_inside_ratio <= 0.3892
central_joint_band_inside_ratio <= 0.2870
border_attention_score >= 0.1953
```

The same suspicious-case detection procedure was run on correctly classified validation cases.

```text
Correct validation cases evaluated: 851
Suspicious validation cases: 188
Suspicious fraction: 22.1%
Low-region-attention cases: 173
High-border-attention cases: 171
```
Threshold:

```text
central_ellipse_inside_ratio <= 0.5499
quantized_central_inside_ratio <= 0.3980
central_joint_band_inside_ratio <= 0.2882
border_attention_score >= 0.1939
```
### Decision: ROI Analysis and Data Cleanup

Since inverted images complicate Region of Interest (ROI) detection, we implemented a data cleanup strategy enabling on-the-fly inversion. To evaluate the model's visual focus, we accumulated Grad-CAM heatmaps across 100 balanced samples per class. 

![Accumulated Gradcam Baseline](report/figures/gradcam_accum_baseline.png)
*Figure 1: Accumulated Grad-CAM - Baseline Model*

![Accumulated Gradcam Baseline](report/figures/gradcam_accum_gauss_crop.png)
*Figure 2: Accumulated Grad-CAM - Gauss & Crop Model*

The accumulated heatmaps reveal recognizable differences between the models. To interpret these patterns, we cross-referenced the activations with the OARSI (Osteoarthritis Research Society International) guidelines for Kellgren-Lawrence (KL) grading, which define specific anatomical regions of interest:

* **KL 0:** General joint space (wide and even).
* **KL 1 & 2:** Joint margins (emerging osteophytes) and initial joint space narrowing.
* **KL 3 & 4:** Severe joint space narrowing, subchondral sclerosis, and structural bone deformation.

**Observations & Hypothesis:**
The heatmaps demonstrate that the joint space heavily influences the model's predictions, aligning well with clinical methodology. Notably, the model trained with Gaussian blur and cropping shows distinctly higher attention on marginal osteophytes for KL2 predictions. For KL3 and KL4, the strong activation around the outer margins is likely a reaction to advanced bone deformation. 

To definitively confirm whether the model focuses on these exact anatomical anomalies, a higher-resolution visualization technique (e.g., HiResCAM or Grad-CAM++) is required.

**Open Question for Further Strategy:**
Given that the diagnostic focus shifts depending on the severity of the disease, should we consider implementing **class-specific ROI masks** to explicitly define and evaluate where the model *should* be looking for each KL grade?


## KW23 - 1 June 2026

### Decision: Change of region detection method for Grad-CAM overlap analysis

We decided to replace the previous multi-mask diagnostic-region setup with a new region detection method, used together with border detection. The updated XAI analysis will therefore use two region-related components: the new method for estimating the relevant knee region and a separate border detection metric for identifying attention on non-diagnostic image borders.

This change was made because the previous setup relied on several proxy masks, which made the suspicious-case rule harder to explain and compare. The new method is intended to provide a cleaner and more reproducible definition of the relevant image region, while border detection remains important because the project brief explicitly focuses on cases where the model may rely on borders, scanner artifacts, or other non-diagnostic regions rather than clinically relevant knee anatomy.

The suspicious-case logic will be updated accordingly. A correctly classified image can be considered suspicious if the Grad-CAM activation is low inside the region detected by the new method, or if the border attention score is high.

This change affects H1 directly because it changes how we operationalise “diagnostically unfaithful” saliency. It may also affect H2 because the suspicious cases identified by this updated rule can later be used for XAI-guided intervention experiments. We will document the new method, its thresholds, and example visualizations before using it for final suspicious-case statistics.