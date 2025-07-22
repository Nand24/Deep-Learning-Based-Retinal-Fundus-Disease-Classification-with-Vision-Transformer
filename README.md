
# Retinal Fundus Classification with Vision Transformer (ViT)

This project explores the use of Vision Transformers (ViT) to classify retinal fundus images into 11 disease categories. It was developed during a research internship at IIT Roorkee, under the guidance of Prof. Dr. Millie Pant.

---

## Overview

Retinal diseases such as Glaucoma, Diabetic Retinopathy, and Macular Degeneration can lead to irreversible blindness if not diagnosed early. Manual diagnosis is resource-intensive and prone to subjectivity.

This project aims to:
- Develop an automated, interpretable, and accurate classification system using Vision Transformers.
- Implement the solution using accessible tools such as Google Colab for scalability and reproducibility.

---

## Dataset

- **Source:** Public Kaggle dataset
- **Total Images:** Approximately 20,000 labeled fundus images
- **Classes:**
  - Normal Fundus
  - Diabetic Retinopathy (Mild, Moderate, Severe, Proliferative)
  - Glaucoma
  - Cataract
  - Age-related Macular Degeneration (Dry & Wet)
  - Hypertensive Retinopathy
  - Pathological Myopia

---

## Model Architecture

- **Backbone:** vit_tiny_patch16_224 (via `timm`)
- **Pretrained on:** ImageNet
- **Optimizer:** AdamW
- **Scheduler:** CosineAnnealingLR
- **Loss Function:** Cross Entropy
- **Batch Size:** 32
- **Epochs:** 30
- **Mixed Precision:** Enabled (torch.cuda.amp)

---

## Features

- Trained and evaluated on Google Colab using GPUs
- Performance metrics and logs are stored in `log.csv`
- Final test accuracy: 72.17%
- Includes explainability using Grad-CAM and attention maps
- Generates visualizations including per-class metrics, confusion matrices, ROC curves

---

## Directory Structure

```
.
├── Code For Vision Transformer.py  # Full training and evaluation pipeline
├── /data
│   ├── train/
│   ├── val/
│   └── test/
├── /outputs
│   ├── vit_latest.pt               # Final model checkpoint
│   ├── log.csv                     # Training logs
│   ├── per_class_bar_plot.png     # Class-wise metrics
│   └── confusion_matrix.png       # Confusion matrix
```

---

## Setup & Installation

Install the following dependencies:

```bash
pip install torch torchvision timm torchmetrics matplotlib seaborn pandas scikit-learn opencv-python
```

---

## How to Run

1. Organize the dataset in the following structure: `train/`, `val/`, `test/`.
2. Upload the files to Google Colab or run locally with GPU support.
3. Execute the Python script or notebook line by line.
4. Monitor logs, training curves, and evaluation outputs saved in the output directory.

---

## Model Interpretability

The following tools were used to analyze and interpret model predictions:
- Grad-CAM for ViT
- t-SNE visualization for feature embeddings
- Class-wise bar plots and heatmaps for metrics

---

## Internship Details

- **Intern:** Pranav Sharma (B.Tech CSE, 2nd Year)  
- **Guide:** Prof. Dr. Millie Pant, HOD, AMS Dept., IIT Roorkee  
- **Internship Duration:** June 9, 2025 – July 11, 2025  
- **Home Institute:** Jaypee Institute of Information Technology, Noida

---

## Acknowledgements

- Prof. Dr. Millie Pant Ma'am for her expert guidance  
- Shubham Joshi Sir for mentorship and support  
- Google Colab for providing accessible computing resources

---
