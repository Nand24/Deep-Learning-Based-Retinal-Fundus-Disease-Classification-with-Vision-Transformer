# Retinal Fundus Classification Using Vision Transformer (ViT)

This project presents an in-depth investigation into the application of Vision Transformer (ViT) architectures for classifying retinal fundus images across 11 diagnostic categories. The work was undertaken as part of a research internship at the Indian Institute of Technology (IIT) Roorkee, under the supervision of Prof. Dr. Millie Pant.

---

## Research Context

Timely detection of retinal diseases such as Diabetic Retinopathy, Glaucoma, and Age-related Macular Degeneration (AMD) is essential for preventing irreversible visual impairment. Traditional diagnosis methods, which involve manual interpretation by specialists, are time-consuming and prone to inter-observer variability. In recent years, deep learning—particularly transformer-based models—has enabled significant advancements in developing automated, scalable, and interpretable diagnostic systems in the field of ophthalmology.

---

## Objectives

- To design and implement an end-to-end ViT-based deep learning pipeline for classifying retinal fundus images into multiple disease categories.
- To evaluate the model's performance using standard metrics: accuracy, precision, recall, F1-score, and AUC-ROC.
- To incorporate explainability techniques including Grad-CAM, t-SNE, and attention maps to enhance model transparency.
- To optimize training using mixed-precision and transfer learning within the computational limitations of Google Colab.
- To contribute a reproducible and adaptable framework for medical image classification tasks.

---

## Dataset Overview

- **Source**: Publicly available Kaggle dataset  
- **Link**: [Retinal Fundus Images](https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images)  
- **Total Samples**: Approximately 20,000 high-resolution labeled images  
- **Diagnostic Classes (11)**:
  - Normal Fundus
  - Diabetic Retinopathy (Mild, Moderate, Severe, Proliferative)
  - Glaucoma
  - Cataract
  - Hypertensive Retinopathy
  - Age-related Macular Degeneration (Dry and Wet)
  - Pathological Myopia

---

## Methodology

### Model Configuration

| Component         | Configuration                   |
|------------------|----------------------------------|
| Architecture      | ViT-Tiny (Patch size: 16×16)     |
| Input Resolution  | 224×224                          |
| Pretrained Weights| ImageNet-1K                      |
| Optimizer         | AdamW                            |
| Learning Rate Scheduler | CosineAnnealingLR         |
| Loss Function     | Cross Entropy                    |
| Epochs            | 30                               |
| Batch Size        | 32                               |
| Precision         | Mixed Precision (AMP)            |

### Training Protocol

- Dataset was split into 70% training, 10% validation, and 20% testing.
- Training was conducted on Google Colab using available GPU resources.
- Model checkpoints and training logs were recorded.
- Visualization outputs such as Grad-CAM overlays and evaluation plots were saved for analysis.

---

## Results

| Evaluation Metric      | Value     |
|------------------------|-----------|
| Test Accuracy          | 72.17%    |
| Number of Classes      | 11        |
| Best Class-wise F1     | ~0.82     |
| ROC & PR Curves        | Generated for all classes |
| Grad-CAM Visualization | Implemented for major classes |

The model achieved a balanced performance across most disease categories. Visualization techniques confirmed that the Vision Transformer architecture successfully learned localized retinal features indicative of disease presence.

---

## Evaluation & Visualization

- Confusion Matrix illustrating inter-class prediction behavior
- Per-class precision, recall, and F1-score bar plots
- Receiver Operating Characteristic (ROC) and Precision-Recall curves
- Grad-CAM heatmaps across two image grids for interpretability
- Optional attention head visualization from ViT for qualitative analysis
- t-SNE projections of feature embeddings for cluster validation

---

## Technology Stack

- Python, PyTorch, torchvision
- timm (ViT pre-trained model loading)
- scikit-learn, TorchMetrics
- OpenCV, matplotlib, seaborn
- Google Colab (free-tier GPU with AMP support)


---

## Internship Details

- **Intern Name**: Pranav Sharma  
- **Program**: B.Tech Computer Science & Engineering  
- **Home Institute**: Jaypee Institute of Information Technology, Noida  
- **Research Guide**: Prof. Dr. Millie Pant, Head of AMSC Department  
- **Host Institute**: Indian Institute of Technology (IIT) Roorkee  
- **Internship Period**: June 9, 2025 – July 11, 2025  

---

## Acknowledgements

The author expresses gratitude to:
- **Prof. Dr. Millie Pant Ma'am** for her insightful guidance and academic mentorship.
- **Shubham Joshi Sir** for technical assistance and feedback throughout the implementation phase.
- **Google Colab** for enabling this research with accessible GPU resources.
- The **Kaggle community** for providing the dataset essential to this study.

---

## Future Work

- Explore training with larger ViT architectures such as ViT-Base and ViT-Large.
- Implement advanced augmentation strategies (e.g., CutMix, AutoAugment).
- Investigate hybrid models combining CNN and transformer layers for spatial learning.
- Extend the current pipeline to multi-label classification of coexisting conditions.
- Deploy as a clinical decision-support application using frameworks like Flask or Streamlit.

---

## Citation

If this project contributes to your work, please consider citing or referencing this research internship conducted at IIT Roorkee under the guidance of Prof. Millie Pant in 2025.

