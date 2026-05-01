# 🧠 Brain Tumor MRI Classification Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

A deep learning pipeline that classifies brain MRI scans into four diagnostic categories — **Glioma**, **Meningioma**, **Pituitary Tumour**, and **No Tumour** — using transfer learning with MobileNetV2. The project is designed to explore how pre-trained convolutional neural networks can be adapted for medical image analysis with limited training data.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Key Learnings](#key-learnings)
- [References](#references)

---

## Project Overview

Brain tumours are among the most life-threatening medical conditions. Early and accurate classification from MRI scans can significantly improve patient outcomes. Manual radiological analysis is time-consuming and subject to human error. This project investigates whether a lightweight deep learning model — fine-tuned on a relatively small dataset of ~7,000 images — can reliably distinguish between four brain tumour types.

**Research Questions Addressed:**
1. Can a MobileNetV2 model pretrained on natural images (ImageNet) transfer effectively to brain MRI classification?
2. Does two-phase fine-tuning (head-only training followed by partial base unfreezing) improve classification performance?
3. Which tumour types are most prone to misclassification, and why?

---

## Dataset

**Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle)

| Split    | Images | Classes |
|----------|--------|---------|
| Training | ~5,712 | 4       |
| Testing  | ~1,311 | 4       |

| Class       | Description |
|-------------|-------------|
| `glioma`    | Tumours arising from glial cells; most common malignant brain tumour |
| `meningioma`| Tumours of the meninges (brain lining); often benign but space-occupying |
| `pituitary` | Tumours of the pituitary gland; can disrupt hormone regulation |
| `notumor`   | Normal brain MRI scans (control group) |

**Data Augmentation Applied During Training:**
- Random horizontal flip
- Rotation (±20°)
- Width/height shift (±10%)
- Zoom (±20%)
- Nearest-neighbour fill for empty regions

---

## Model Architecture

```
INPUT IMAGE (224 × 224 × 3)
        │
        ▼
┌─────────────────────────────────────┐
│  MobileNetV2 Base (154 layers)      │
│  Pre-trained on ImageNet            │
│  Phase 1: ALL FROZEN                │
│  Phase 2: Last 100 layers UNFROZEN  │
└─────────────────────────────────────┘
        │
        ▼
GlobalAveragePooling2D  →  (1280,)
        │
        ▼
BatchNormalization
        │
        ▼
Dense(256, ReLU) + L2 Regularisation
        │
        ▼
Dropout(0.5)
        │
        ▼
Dense(4, Softmax)
        │
        ▼
OUTPUT: [P(glioma), P(meningioma), P(notumor), P(pituitary)]
```

**Why MobileNetV2?**
- Designed for efficiency: high accuracy with low computational cost
- Inverted residual blocks with linear bottlenecks — fewer parameters than VGG or ResNet
- Well-suited for deployment on resource-constrained hardware
- Strong baseline for medical imaging tasks in published literature

---

## Training Strategy

### Phase 1 — Classification Head Training
- **Epochs:** 10 (with early stopping, patience=5)
- **Learning Rate:** 0.001
- **Base model:** Fully frozen — acts as a fixed feature extractor
- **Goal:** Train the new Dense layers to learn tumour-specific patterns from MobileNetV2 features

### Phase 2 — Fine-tuning
- **Epochs:** 10 (with early stopping)
- **Learning Rate:** 0.0001 (10× smaller — prevents destroying pretrained weights)
- **Base model:** Last 100 layers unfrozen — adapts to MRI texture and contrast
- **Goal:** Allow the model to specialise for MRI data at a deep feature level

**Callbacks Used:**
| Callback | Purpose |
|---|---|
| `EarlyStopping` (patience=5) | Stops training if val_accuracy plateaus |
| `ModelCheckpoint` | Saves best model to disk |
| `ReduceLROnPlateau` (factor=0.5) | Halves learning rate when stuck |

---

## Results

> Results will be updated after training completes. Below are representative benchmarks from similar architectures in the literature.

| Metric | Value |
|--------|-------|
| Test Accuracy | ~97–98% (expected) |
| Glioma F1-score | ~0.97 |
| Meningioma F1-score | ~0.95 |
| Pituitary F1-score | ~0.99 |
| No Tumour F1-score | ~0.99 |

*Note: Actual results will be populated once training is complete. See `results/` folder for plots.*

**Generated Output Files:**
```
results/
├── brain_tumor_model.h5        ← Saved trained model weights
├── training_history.png        ← Accuracy & loss curves (Phase 1 + Phase 2)
├── confusion_matrix.png        ← Normalised confusion matrix
├── roc_curves.png              ← Per-class ROC curves with AUC scores
├── sample_predictions.png      ← Grid of test image predictions
└── class_distribution.png     ← Training set class balance chart
```

---

## Project Structure

```
brain-tumor-mri-classifier/
│
├── config.py           ← All hyperparameters and paths in one place
├── data_loader.py      ← Data pipeline: loading, augmentation, generators
├── model.py            ← MobileNetV2 transfer learning architecture
├── train.py            ← Full training loop (Phase 1 + Phase 2)
├── evaluate.py         ← Confusion matrix, ROC curves, classification report
├── predict.py          ← Single image inference with visualisation
│
├── data/               ← Dataset folder (download from Kaggle — NOT on GitHub)
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── results/            ← Auto-generated output figures and model weights
├── requirements.txt    ← Python package dependencies
└── .gitignore          ← Excludes data/ and model weights from GitHub
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ashwinsukumar/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
1. Go to: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Click **Download**
3. Extract the downloaded ZIP into a folder named `data/` in the project root
4. Ensure the folder structure matches what is shown above

### 4. Verify setup
```bash
python -c "import tensorflow as tf; print('TF version:', tf.__version__)"
```

---

## Usage

### Train the model
```bash
python train.py
```
This will:
1. Load and augment the training images
2. Run Phase 1 training (10 epochs, frozen base)
3. Run Phase 2 fine-tuning (10 epochs, last 100 layers unfrozen)
4. Save the best model to `results/brain_tumor_model.h5`
5. Save the training history plot

### Evaluate on the test set
```bash
python evaluate.py
```
Outputs: confusion matrix, ROC curves, classification report, sample predictions.

### Classify a single image
```bash
python predict.py --image data/Testing/glioma/Te-gl_0010.jpg
```

---

## Key Learnings

Through building this project, the following concepts were explored and understood:

- **Transfer Learning**: How features learned from natural image classification (ImageNet) generalise to medical imaging tasks, reducing training data requirements from millions to thousands of images.

- **Two-Phase Fine-Tuning**: The importance of training only the classification head first to stabilise weights before carefully unfreezing lower layers at a reduced learning rate.

- **Data Augmentation**: How artificial variation in training data (rotation, zoom, flip) improves model generalisation to real-world scan variability.

- **Class Imbalance Awareness**: Understanding how unequal class distribution can bias model performance, and how to detect this with distribution analysis.

- **Evaluation Metrics Beyond Accuracy**: Why precision, recall, F1-score, and AUC-ROC are more informative than accuracy alone for medical classification tasks.

---

## References

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). **MobileNetV2: Inverted Residuals and Linear Bottlenecks**. CVPR 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)

2. Nickparvar, M. (2021). **Brain Tumor MRI Dataset** [Data set]. Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

3. Chollet, F. (2021). *Deep Learning with Python*, 2nd Edition. Manning Publications.

4. Lee, B. et al. (2023). **Multiclass Teeth Segmentation using Teeth Attention Modules for Dental X-ray Images**. *IEEE Access*, 11. *(Related work on medical image segmentation)*

---

## Author

**Ashwin Sukumar**
B.Sc. Computer Science, SRM University (2021)
Senior Data Analyst | Aspiring Graduate Researcher

📧 ashwinachu179@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/ashwinsukumar) | [GitHub](https://github.com/ashwinsukumar)

---

*This project was developed as part of my preparation for graduate study in Computer Science / AI Convergence.*

---

**MIT License** — See [LICENSE](LICENSE) for details.
