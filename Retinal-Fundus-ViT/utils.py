import csv
import random
from pathlib import Path

import numpy as np
import torch

from config import CKPT_DIR, LOG_FILE, SEED


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(model, optimizer, scheduler, epoch: int, path: Path):
    ensure_dir(path.parent)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint


def append_log(row, log_file: Path = LOG_FILE):
    ensure_dir(log_file.parent)
    write_header = not log_file.exists()
    with log_file.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_accuracy"])
        writer.writerow(row)
