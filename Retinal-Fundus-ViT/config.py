import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
CKPT_DIR = ROOT_DIR / "checkpoints"
LOG_FILE = CKPT_DIR / "training_log.csv"

MODEL_NAME = "vit_tiny_patch16_224"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
NUM_CLASSES = 11
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 4

LABELS = [
    "Normal",
    "Mild_DR",
    "Moderate_DR",
    "Severe_DR",
    "Proliferative_DR",
    "Glaucoma",
    "Cataract",
    "Hypertensive_Retinopathy",
    "AMD_Dry",
    "AMD_Wet",
    "Pathological_Myopia",
]
