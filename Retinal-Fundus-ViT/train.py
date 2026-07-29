import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.classification import MulticlassAccuracy

from config import CKPT_DIR, DEVICE, EPOCHS, LEARNING_RATE, BATCH_SIZE
from dataset import get_train_loader, get_val_loader
from model import create_model
from utils import append_log, ensure_dir, save_checkpoint, set_seed


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            logits = model(inputs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, metric):
    model.eval()
    metric.reset()
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(inputs)
            metric.update(logits, labels)
    return metric.compute().item()


def main(args):
    set_seed()
    ensure_dir(CKPT_DIR)

    train_loader = get_train_loader(batch_size=args.batch_size)
    val_loader = get_val_loader(batch_size=args.batch_size)

    model = create_model(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    val_metric = MulticlassAccuracy(num_classes=model.num_classes).to(DEVICE)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_accuracy = validate(model, val_loader, val_metric)

        scheduler.step()
        save_checkpoint(model, optimizer, scheduler, epoch, CKPT_DIR / f"vit_epoch_{epoch}.pt")
        append_log([epoch, f"{train_loss:.4f}", f"{val_accuracy:.4f}"])

        print(f"Epoch {epoch}/{EPOCHS} - Train loss: {train_loss:.4f} - Val accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Retina Fundus ViT")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    main(args)
