from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def build_loader(directory: Path, batch_size: int = BATCH_SIZE, shuffle: bool = False):
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    dataset = datasets.ImageFolder(directory, transform=get_transforms(train=shuffle))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def get_train_loader(batch_size: int = BATCH_SIZE):
    return build_loader(TRAIN_DIR, batch_size=batch_size, shuffle=True)


def get_val_loader(batch_size: int = BATCH_SIZE):
    return build_loader(VAL_DIR, batch_size=batch_size, shuffle=False)


def get_test_loader(batch_size: int = BATCH_SIZE):
    return build_loader(TEST_DIR, batch_size=batch_size, shuffle=False)
