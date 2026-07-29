import argparse
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import DEVICE, LABELS
from dataset import get_test_loader, get_val_loader
from model import create_model
from utils import load_checkpoint


def evaluate(model, loader, label_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
    matrix = confusion_matrix(all_labels, all_preds)
    return report, matrix


def main(args):
    model = create_model(pretrained=False).to(DEVICE)
    checkpoint = load_checkpoint(Path(args.checkpoint), model, device=DEVICE)
    loader = get_test_loader(batch_size=args.batch_size) if args.test else get_val_loader(batch_size=args.batch_size)

    report, matrix = evaluate(model, loader, LABELS)
    print(report)
    print("Confusion matrix:\n", matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Retina Fundus ViT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test", action="store_true", help="Evaluate on test set instead of validation set")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()
    main(args)
