# Retinal-Fundus-ViT

A Vision Transformer (ViT) based pipeline for retinal fundus disease classification.

## Project Structure

- `config.py` - hyperparameters, dataset paths, and training configuration.
- `dataset.py` - data loading and augmentation utilities.
- `model.py` - ViT model creation and checkpoint helpers.
- `train.py` - training loop with mixed precision and checkpointing.
- `evaluate.py` - evaluation script with classification report and confusion matrix.
- `visualize.py` - plotting utilities for training metrics and confusion matrices.
- `gradcam.py` - attention-rollout style interpretability helper for ViT.
- `utils.py` - seed, checkpoint, and logging helpers.
- `requirements.txt` - Python dependencies.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare dataset folders:
   - `data/train`
   - `data/val`
   - `data/test` (optional)

3. Train the model:
   ```bash
   python train.py --epochs 30 --batch-size 32
   ```

4. Evaluate the model:
   ```bash
   python evaluate.py --checkpoint checkpoints/vit_epoch_30.pt
   ```

## Using Google Drive and Colab

If your dataset is on Google Drive and you want to run training in Colab:

1. Mount Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Copy or link your dataset into the notebook runtime:
   - If you want to use a small subset, copy files into `/content/data/train` and `/content/data/val`
   - Otherwise point the code to the Drive folders directly

3. If you prefer to keep the folder structure, import and run the script from Colab:
   ```python
   %cd /content/Retinal-Fundus-ViT
   !pip install -r requirements.txt
   !python train.py --epochs 30 --batch-size 32 --learning-rate 3e-4
   ```

4. Use the same `evaluate.py` command in Colab to test saved checkpoints.

## Colab Notebook Example

Use this sequence in a Colab notebook to mount Drive, install requirements, train, and evaluate.

```python
from google.colab import drive

drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/path/to/Deep-Learning-Based-Retinal-Fundus-Disease-Classification-with-Vision-Transformer
!pip install -r Retinal-Fundus-ViT/requirements.txt
```

```python
!python Retinal-Fundus-ViT/train.py --epochs 30 --batch-size 32 --learning-rate 3e-4
```

```python
!python Retinal-Fundus-ViT/evaluate.py --checkpoint Retinal-Fundus-ViT/checkpoints/vit_epoch_30.pt --test --batch-size 32
```

## Notes

- The default model is `vit_tiny_patch16_224` from `timm`.
- The dataset loader expects a folder structure compatible with `torchvision.datasets.ImageFolder`.
- The `gradcam.py` file provides a simple attention-rollout helper for ViT interpretability.
