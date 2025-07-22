
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
from pathlib import Path
import random

src_root = Path("/content/drive/MyDrive/retina_data/retinal_fundus_images/train")
dst_root = Path("data/train")
dst_root.mkdir(parents=True, exist_ok=True)

for cls in os.listdir(src_root):
    src_class = src_root / cls
    dst_class = dst_root / cls
    dst_class.mkdir(parents=True, exist_ok=True)

    all_imgs = os.listdir(src_class)
    for img in random.sample(all_imgs, min(100, len(all_imgs))):
        shutil.copy(src_class / img, dst_class / img)

src_val = Path("/content/drive/MyDrive/retina_data/retinal_fundus_images/val")
dst_val = Path("data/val")
dst_val.mkdir(parents=True, exist_ok=True)

for cls in os.listdir(src_val):
    src_class = src_val / cls
    dst_class = dst_val / cls
    dst_class.mkdir(parents=True, exist_ok=True)

    all_imgs = os.listdir(src_class)
    for img in random.sample(all_imgs, min(40, len(all_imgs))):
        shutil.copy(src_class / img, dst_class / img)

import torch, timm, csv, time, pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

BATCH_SIZE = 64
IMAGE_SIZE = 64
EPOCHS = 30  # will take ~90 mins
CKPT_DIR = Path("/content/drive/MyDrive/retina_ckpt")
CKPT_DIR.mkdir(exist_ok=True, parents=True)
LOGFILE = CKPT_DIR / "log.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

train_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
val_tfms = train_tfms

train_ds = datasets.ImageFolder("data/train", transform=train_tfms)
val_ds = datasets.ImageFolder("data/val", transform=val_tfms)
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, BATCH_SIZE)

model = timm.create_model("vit_tiny_patch16_64", pretrained=True, num_classes=len(train_ds.classes)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()
acc_metric = MulticlassAccuracy(num_classes=len(train_ds.classes)).to(device)
scaler = torch.cuda.amp.GradScaler()

if not LOGFILE.exists():
    with open(LOGFILE, "w") as f: csv.writer(f).writerow(["epoch", "train_loss", "val_acc"])

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)

    scheduler.step()
    train_loss = total_loss / len(train_ds)

    model.eval(); acc_metric.reset()z
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            acc_metric.update(out, y)
    val_acc = acc_metric.compute().item()

    with open(LOGFILE, "a") as f:
        csv.writer(f).writerow([epoch+1, f"{train_loss:.4f}", f"{val_acc:.4f}"])
    print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f}")

!pip install -q timm torchmetrics

import torch, timm, csv, time, pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

BATCH_SIZE = 64
IMAGE_SIZE = 64
EPOCHS = 30  
CKPT_DIR = Path("/content/drive/MyDrive/retina_ckpt")
CKPT_DIR.mkdir(exist_ok=True, parents=True)
LOGFILE = CKPT_DIR / "log.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

train_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
val_tfms = train_tfms

train_ds = datasets.ImageFolder("data/train", transform=train_tfms)
val_ds = datasets.ImageFolder("data/val", transform=val_tfms)
train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, BATCH_SIZE)

model = timm.create_model("vit_tiny_patch16_64", pretrained=True, num_classes=len(train_ds.classes)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()
acc_metric = MulticlassAccuracy(num_classes=len(train_ds.classes)).to(device)
scaler = torch.cuda.amp.GradScaler()

if not LOGFILE.exists():
    with open(LOGFILE, "w") as f: csv.writer(f).writerow(["epoch", "train_loss", "val_acc"])

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)

    scheduler.step()
    train_loss = total_loss / len(train_ds)

    model.eval(); acc_metric.reset()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            acc_metric.update(out, y)
    val_acc = acc_metric.compute().item()

    with open(LOGFILE, "a") as f:
        csv.writer(f).writerow([epoch+1, f"{train_loss:.4f}", f"{val_acc:.4f}"])
    print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f}")

import os, shutil, random
from pathlib import Path

SRC_TRAIN = Path("/content/drive/MyDrive/retina_data/retinal_fundus_images/train")
SRC_VAL   = Path("/content/drive/MyDrive/retina_data/retinal_fundus_images/val")
DST_TRAIN = Path("data/train")
DST_VAL   = Path("data/val")

def copy_subset(src_root, dst_root, n_per_class):
    dst_root.mkdir(parents=True, exist_ok=True)
    for cls in os.listdir(src_root):
        src_cls = src_root / cls
        dst_cls = dst_root / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        images = os.listdir(src_cls)
        for img in random.sample(images, min(n_per_class, len(images))):
            shutil.copy(src_cls / img, dst_cls / img)

copy_subset(SRC_TRAIN, DST_TRAIN, 100)
copy_subset(SRC_VAL, DST_VAL, 40)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 224
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

train_ds = datasets.ImageFolder("data/train", transform=transform)
val_ds   = datasets.ImageFolder("data/val",   transform=transform)

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds, BATCH_SIZE)

num_classes = len(train_ds.classes)
print('Classes:', train_ds.classes)

import torch, timm
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
metric = MulticlassAccuracy(num_classes=num_classes).to(device)
scaler = torch.cuda.amp.GradScaler()

EPOCHS = 30  # should complete in ~90 minutes

import csv
from pathlib import Path

CKPT_DIR = Path('/content/drive/MyDrive/retina_ckpt_2h')
CKPT_DIR.mkdir(exist_ok=True, parents=True)
LOGFILE = CKPT_DIR / 'log.csv'

if not LOGFILE.exists():
    with open(LOGFILE, 'w') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_acc'])

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)

    train_loss = total_loss / len(train_ds)

    model.eval()
    metric.reset()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            metric.update(out, y)
    val_acc = metric.compute().item()

    with open(LOGFILE, 'a') as f:
        csv.writer(f).writerow([epoch+1, f'{train_loss:.4f}', f'{val_acc:.4f}'])

if (epoch + 1) % 5 == 0:
    ckpt_path = CKPT_DIR / f"vit_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt_path)


    print(f'Epoch {epoch+1:02}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}')

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

print("Loaded Test Data. Classes:", test_ds.classes)

torch.save(model.state_dict(), "/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt")

import timm, torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(test_ds.classes)

model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt"))
model.eval()



import numpy as np
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

all_preds = []
all_labels = []

with torch.no_grad(), torch.cuda.amp.autocast():
    for x, y in test_dl:
        x = x.to(device)
        out = model(x)
        probs = F.softmax(out, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print(classification_report(all_labels, all_preds, target_names=test_ds.classes))

print(classification_report(all_labels, all_preds, target_names=test_ds.classes))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_ds.classes, yticklabels=test_ds.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Test Set")
plt.show()

with open("/content/drive/MyDrive/test_report.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(all_labels, all_preds, target_names=test_ds.classes))

cm = confusion_matrix(all_labels, all_preds)
classes = test_ds.classes

import seaborn as sns, matplotlib.pyplot as plt
import numpy as np, os

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Test Set")
plt.tight_layout()

img_path = "/content/drive/MyDrive/retina_ckpt_2h/confusion_matrix.png"
plt.savefig(img_path)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/drive/MyDrive/retina_ckpt_2h/log.csv")

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color='tab:red')
ax1.plot(df['epoch'], df['train_loss'], color='tab:red', label='Train Loss')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('Val Accuracy', color='tab:blue')
ax2.plot(df['epoch'], df['val_acc'], color='tab:blue', label='Val Accuracy')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title("Training Loss & Validation Accuracy Curve")
fig.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

df.plot(kind='bar', figsize=(8, 5), ylim=(0, 1), colormap='tab10')
plt.title("Per-Class Precision, Recall, and F1-score")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.grid(True, axis='y')
plt.tight_layout()

plt.savefig("/content/drive/MyDrive/per_class_bar_plot.png", dpi=300)
plt.show()

plt.savefig("/content/drive/MyDrive/retina_ckpt_2h/loss_accuracy_curve.png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

df.plot(kind='bar', figsize=(8, 5), ylim=(0, 1), colormap='tab10')
plt.title("Per-Class Precision, Recall, and F1-score")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.grid(True, axis='y')
plt.tight_layout()

plt.savefig("/content/drive/MyDrive/per_class_bar_plot.png", dpi=300)
plt.show()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder(
    "/content/drive/MyDrive/retina_data/retinal_fundus_images/test",
    transform=test_tfms
)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes  # ['Glaucoma', 'Normal', 'Diabetic'] or similar order

all_labels = []
all_preds = []
model.eval()
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

df.plot(kind='bar', figsize=(8,5), ylim=(0,1), colormap='tab10')
plt.title("Per-Class Precision, Recall, F1-Score")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()

plt.savefig("/content/drive/MyDrive/per_class_bar_plot.png", dpi=300)
plt.show()

import torch, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=3)
model.load_state_dict(torch.load("/content/drive/MyDrive/retina_ckpt_2h/ckpt_epoch_30.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]
df.plot(kind='bar', figsize=(8,5), ylim=(0,1), colormap='tab10')
plt.title("Per-Class Precision, Recall, F1-score")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_bar_plot.png", dpi=300)
plt.show()


import torch, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=3)
model.load_state_dict(torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]
df.plot(kind='bar', figsize=(8,5), ylim=(0,1), colormap='tab10')
plt.title("Per-Class Precision, Recall, F1-score")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_bar_plot.png", dpi=300)
plt.show()


import torch, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt")
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]
df.plot(kind='bar', figsize=(8,5), ylim=(0,1), colormap='tab10')
plt.title("Per-Class Precision, Recall, F1-score")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_bar_plot.png", dpi=300)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(font_scale=1.2)
plt.figure(figsize=(8, 4))
sns.heatmap(df, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True)
plt.title("Per-Class Precision, Recall, F1-score (ViT)", fontsize=14)
plt.ylabel("Class")
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_heatmap.png", dpi=300)
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

df.plot(kind='bar', ax=axs[0], colormap='Accent', ylim=(0,1))
axs[0].set_title("Per-Class Precision, Recall, F1-score")
axs[0].set_ylabel("Score")
axs[0].legend(loc='lower right')
axs[0].grid(axis='y')

table_data = np.round(df.values, 2)
table = axs[1].table(cellText=table_data,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
axs[1].axis('off')

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_table_bar_combo.png", dpi=300)
plt.show()

from sklearn.metrics import classification_report
import pandas as pd

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

import torch, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt")
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

df.plot(kind='barh', figsize=(10, 5), colormap='Set2')
plt.title("Per-Class Precision, Recall, and F1-score (ViT)")
plt.xlabel("Score")
plt.xlim(0, 1)
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_barh_plot.png", dpi=300)
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

df.plot(kind='bar', ax=axs[0], colormap='Accent', ylim=(0,1))
axs[0].set_title("Per-Class Precision, Recall, F1-score")
axs[0].set_ylabel("Score")
axs[0].legend(loc='lower right')
axs[0].grid(axis='y')

table_data = np.round(df.values, 2)
table = axs[1].table(cellText=table_data,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
axs[1].axis('off')

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_table_bar_combo.png", dpi=300)
plt.show()

import torch, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt/vit_latest")
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

df.plot(kind='bar', ax=axs[0], colormap='Accent', ylim=(0, 1))
axs[0].set_title("Per-Class Precision, Recall, F1-score (ViT)")
axs[0].set_ylabel("Score")
axs[0].legend(loc='lower right')
axs[0].grid(axis='y')

table_data = np.round(df.values, 2)
table = axs[1].table(cellText=table_data,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
axs[1].axis('off')

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_table_bar_combo.png", dpi=300)
plt.show()


import torch, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt")
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
df.plot(kind='bar', ax=axs[0], colormap='Accent', ylim=(0, 1))
axs[0].set_title("Per-Class Precision, Recall, F1-score (ViT)")
axs[0].set_ylabel("Score")
axs[0].legend(loc='lower right')
axs[0].grid(axis='y')
table_data = np.round(df.values, 2)
table = axs[1].table(cellText=table_data,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
axs[1].axis('off')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_table_bar_combo.png", dpi=300)
plt.show()


import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt")
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

attn_maps = []
def save_attn(module, input, output):
    attn_maps.append(output)

hook = model.blocks[-1].attn.attn_drop.register_forward_hook(save_attn)

with torch.no_grad():
    output = model(input_tensor)
    pred_class = torch.argmax(output).item()

attn = attn_maps[0].squeeze(0)
attn_mean = attn.mean(0)
cls_attn = attn_mean[0, 1:].reshape(14, 14).cpu().numpy()
cls_attn = cv2.resize(cls_attn, (224, 224))
cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

img_np = np.array(image.resize((224, 224)))
heatmap = cv2.applyColorMap(np.u)

import os
print(os.path.exists("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt"))

!ls /content/drive/MyDrive/retina_ckpt_2h

from google.colab import drive
drive.mount('/content/drive')





!ls /content/drive/MyDrive

import torch, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt", map_location=device)
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

df.plot(kind='bar', figsize=(8, 5), ylim=(0, 1), colormap='Set2')
plt.title("Per-Class Precision, Recall, and F1-score (ViT)")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_vit_metrics.png", dpi=300)
plt.show()


import torch, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=3)
model.load_state_dict(torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt"))
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

df.plot(kind='bar', figsize=(8, 5), ylim=(0, 1), colormap='Set2')
plt.title("Per-Class Precision, Recall, and F1-score (ViT)")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_vit_metrics.png", dpi=300)
plt.show()


import torch, timm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=3)

state_dict = torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt", map_location=device)
for key in ['head.weight', 'head.bias']:
    if key in state_dict:
        del state_dict[key]

model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

df.plot(kind='bar', figsize=(8, 5), ylim=(0, 1), colormap='Set2')
plt.title("Per-Class Precision, Recall, and F1-score (ViT)")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_vit_metrics.png", dpi=300)
plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes
n_classes = len(classes)

all_labels = []
all_probs = []

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

y_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    axs[0].plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")

axs[0].plot([0, 1], [0, 1], 'k--')
axs[0].set_title("ROC Curves (One-vs-Rest)")
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")
axs[0].legend(loc="lower right")
axs[0].grid(True)

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
    axs[1].plot(recall, precision, label=f"{classes[i]}")

axs[1].set_title("Precision-Recall Curves (One-vs-Rest)")
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].legend(loc="lower left")
axs[1].grid(True)

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/roc_pr_curves.png", dpi=300)
plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes
true_class_count = len(classes)

all_labels = []
all_probs = []

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

pred_class_count = all_probs.shape[1]
y_true_bin = label_binarize(all_labels, classes=list(range(pred_class_count)))

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for i in range(pred_class_count):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    label = classes[i] if i < true_class_count else f"Class {i}"
    axs[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")

axs[0].plot([0, 1], [0, 1], 'k--')
axs[0].set_title("ROC Curves (One-vs-Rest)")
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")
axs[0].legend(loc="lower right")
axs[0].grid(True)

for i in range(pred_class_count):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
    label = classes[i] if i < true_class_count else f"Class {i}"
    axs[1].plot(recall, precision, label=label)

axs[1].set_title("Precision-Recall Curves (One-vs-Rest)")
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].legend(loc="lower left")
axs[1].grid(True)

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/roc_pr_curves.png", dpi=300)
plt.show()


import torch, timm
import numpy as np, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

model.eval()
features, labels = [], []

with torch.no_grad():
    for x, y in test_dl:
        x = x.to(device)
        feat = model.forward_features(x)  # shape: (batch, 192)
        features.append(feat.cpu().numpy())
        labels.extend(y.cpu().numpy())

features = np.concatenate(features, axis=0)
labels = np.array(labels)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
proj = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))
palette = sns.color_palette("hsv", len(classes))
sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=[classes[i] for i in labels], palette=palette, s=50, alpha=0.8)
plt.title("t-SNE of ViT CLS Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/vit_tsne_plot.png", dpi=300)
plt.show()


import torch, timm
import numpy as np, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import seaborn as sns

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

model.eval()
features, labels = [], []

with torch.no_grad():
    for x, y in test_dl:
        x = x.to(device)
        features.append(feat.cpu().numpy())
        labels.extend(y.cpu().numpy())

features = np.concatenate(features, axis=0)
labels = np.array(labels)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=42)
proj = tsne.fit_transform(features)

plt.figure(figsize=(8, 6))
palette = sns.color_palette("hsv", len(classes))
sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=[classes[i] for i in labels], palette=palette, s=50, alpha=0.8)
plt.title("t-SNE of ViT CLS Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/vit_tsne_plot.png", dpi=300)
plt.show()


import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
raw_tfms = transforms.Resize((224, 224))
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=tfms)
raw_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=raw_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

attn_maps = []
def save_attn(module, input, output):
    attn_maps.append(output)

hook = model.blocks[-1].attn.attn_drop.register_forward_hook(save_attn)

imgs, cams = [], []
count = 0

with torch.no_grad():
    for i, (x, _) in enumerate(test_dl):
        if count >= n_imgs: break
        input_tensor = x.to(device)
        raw_img = np.array(raw_ds[i][0])  # Unnormalized image
        attn_maps.clear()

        _ = model(input_tensor)
        attn = attn_maps[0].squeeze(0).mean(0)  # avg over heads
        cls_attn = attn[0, 1:].reshape(14, 14).cpu().numpy()
        cls_attn = cv2.resize(cls_attn, (224, 224))
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)

        imgs.append(raw_img)
        cams.append(overlay)
        count += 1

n_cols = 3
fig, axs = plt.subplots(n_imgs, 2, figsize=(6, 2*n_imgs))
for i in range(n_imgs):
    axs[i, 0].imshow(imgs[i])
    axs[i, 0].axis('off')
    axs[i, 0].set_title("Original")

    axs[i, 1].imshow(cams[i])
    axs[i, 1].axis('off')
    axs[i, 1].set_title("Grad-CAM Overlay")

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/vit_gradcam_grid.png", dpi=300)
plt.show()


import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
raw_tfms = transforms.Compose([transforms.Resize((224, 224))])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=tfms)
raw_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=raw_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

attn_maps = []

def hook_fn(module, input, output):
    attn_maps.append(output)

hook = model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

n_imgs = 9
imgs, cams = [], []
count = 0

with torch.no_grad():
    for i, (x, _) in enumerate(test_dl):
        if count >= n_imgs:
            break
        input_tensor = x.to(device)
        raw_img = np.array(raw_ds[i][0])

        attn_maps.clear()
        _ = model(input_tensor)

        if len(attn_maps) == 0:
            print("⚠️ Attention hook failed!")
            continue

        attn = attn_maps[0].squeeze(0).mean(0)  # (tokens, tokens)
        cls_attn = attn[0, 1:].reshape(14, 14).cpu().numpy()
        cls_attn = cv2.resize(cls_attn, (224, 224))
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)

        imgs.append(raw_img)
        cams.append(overlay)
        count += 1

fig, axs = plt.subplots(n_imgs, 2, figsize=(6, 2 * n_imgs))
for i in range(n_imgs):
    axs[i, 0].imshow(imgs[i])
    axs[i, 0].axis('off')
    axs[i, 0].set_title("Original")

    axs[i, 1].imshow(cams[i])
    axs[i, 1].axis('off')
    axs[i, 1].set_title("Grad-CAM Overlay")

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/vit_gradcam_grid.png", dpi=300)
plt.show()


import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
raw_tfms = transforms.Compose([transforms.Resize((224, 224))])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=tfms)
raw_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=raw_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

attn_maps = []
def hook_fn(module, input, output):
    attn_maps.append(output)

hook = model.blocks[-1].attn.attn_drop.register_forward_hook(hook_fn)

n_imgs = 9
imgs, cams = [], []
count = 0

with torch.no_grad():
    for i, (x, _) in enumerate(test_dl):
        if count >= n_imgs:
            break
        input_tensor = x.to(device)
        raw_img = np.array(raw_ds[i][0])
        attn_maps.clear()
        _ = model(input_tensor)

        if len(attn_maps) == 0:
            continue

        attn = attn_maps[0].squeeze(0).mean(0)  # (tokens, tokens)
        if attn.shape[0] <= 1:
            continue

        cls_attn = attn[0, 1:].reshape(14, 14).cpu().numpy()
        cls_attn = cv2.resize(cls_attn, (224, 224))
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)

        imgs.append(raw_img)
        cams.append(overlay)
        count += 1

if len(imgs) == 0:
    print(" No valid Grad-CAM overlays were generated.")
else:
    fig, axs = plt.subplots(len(imgs), 2, figsize=(6, 2 * len(imgs)))
    for i in range(len(imgs)):
        axs[i, 0].imshow(imgs[i])
        axs[i, 0].axis('off')
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(cams[i])
        axs[i, 1].axis('off')
        axs[i, 1].set_title("Grad-CAM Overlay")

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/vit_gradcam_grid.png", dpi=300)
    plt.show()

import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
raw_tfms = transforms.Compose([transforms.Resize((224, 224))])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=tfms)
raw_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=raw_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

attn_maps = []

def get_attention_hook(module, input, output):
    if hasattr(module, 'attn_drop'):
        attn_maps.append(module.attn_drop(input[0]))  # manually apply dropout if needed

hook = model.blocks[-1].attn.register_forward_hook(get_attention_hook)

n_imgs = 9
imgs, cams = [], []
count = 0

with torch.no_grad():
    for i, (x, _) in enumerate(test_dl):
        if count >= n_imgs:
            break
        input_tensor = x.to(device)
        raw_img = np.array(raw_ds[i][0])
        attn_maps.clear()
        _ = model(input_tensor)

        if len(attn_maps) == 0:
            continue

        attn = attn_maps[0].squeeze(0).mean(0)  # shape: (tokens, tokens)
        if attn.shape[0] <= 1:
            continue

        cls_attn = attn[0, 1:].reshape(14, 14).cpu().numpy()
        cls_attn = cv2.resize(cls_attn, (224, 224))
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)

        imgs.append(raw_img)
        cams.append(overlay)
        count += 1

if len(imgs) == 0:
else:
    fig, axs = plt.subplots(len(imgs), 2, figsize=(6, 2 * len(imgs)))
    for i in range(len(imgs)):
        axs[i, 0].imshow(imgs[i])
        axs[i, 0].axis('off')
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(cams[i])
        axs[i, 1].axis('off')
        axs[i, 1].set_title("Grad-CAM Overlay")

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/vit_gradcam_grid_fixed.png", dpi=300)
    plt.show()


import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
raw_tfms = transforms.Compose([transforms.Resize((224, 224))])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=tfms)
raw_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=raw_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

attn_maps = []
def hook_fn(module, input, output):
    attn_maps.append(output)

hook = model.blocks[-1].attn.proj.register_forward_hook(hook_fn)

n_imgs = 9
imgs, cams = [], []
count = 0

with torch.no_grad():
    for i, (x, _) in enumerate(test_dl):
        if count >= n_imgs:
            break
        input_tensor = x.to(device)
        raw_img = np.array(raw_ds[i][0])
        attn_maps.clear()
        _ = model(input_tensor)

        if len(attn_maps) == 0:
            continue

        attn = attn_maps[0].squeeze()


        if attn.ndim == 2 and attn.shape[0] > 1:
            cls_attn = attn[0, 1:].reshape(14, 14).cpu().numpy()
        elif attn.ndim == 1 and attn.shape[0] == 196:  # 14x14 tokens
            cls_attn = attn.reshape(14, 14).cpu().numpy()
        else:
            continue

        cls_attn = cv2.resize(cls_attn, (224, 224))
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)

        imgs.append(raw_img)
        cams.append(overlay)
        count += 1

if len(imgs) == 0:
else:
    fig, axs = plt.subplots(len(imgs), 2, figsize=(6, 2 * len(imgs)))
    for i in range(len(imgs)):
        axs[i, 0].imshow(imgs[i])
        axs[i, 0].axis('off')
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(cams[i])
        axs[i, 1].axis('off')
        axs[i, 1].set_title("Grad-CAM Overlay")

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/vit_gradcam_grid_fixed.png", dpi=300)
    plt.show()

import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import math

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
raw_tfms = transforms.Compose([transforms.Resize((224, 224))])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=tfms)
raw_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=raw_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

attn_maps = []
def hook_fn(module, input, output):
    attn_maps.append(output)

hook = model.blocks[-1].attn.proj.register_forward_hook(hook_fn)

n_imgs = 9
imgs, cams = [], []
count = 0

with torch.no_grad():
    for i, (x, _) in enumerate(test_dl):
        if count >= n_imgs:
            break
        input_tensor = x.to(device)
        raw_img = np.array(raw_ds[i][0])
        attn_maps.clear()
        _ = model(input_tensor)

        if len(attn_maps) == 0:
            continue

        attn = attn_maps[0].squeeze()
        if attn.ndim != 2 or attn.shape[0] <= 1:
            continue

        patch_attn = attn[0, 1:].cpu().numpy()
        num_patches = patch_attn.shape[0]
        grid_size = int(math.sqrt(num_patches))

        if grid_size * grid_size != num_patches:
            print(f" Skipping: can't reshape {num_patches} into square grid.")
            continue

        cls_attn = patch_attn.reshape(grid_size, grid_size)
        cls_attn = cv2.resize(cls_attn, (224, 224))
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)

        imgs.append(raw_img)
        cams.append(overlay)
        count += 1

if len(imgs) == 0:
else:
    fig, axs = plt.subplots(len(imgs), 2, figsize=(6, 2 * len(imgs)))
    for i in range(len(imgs)):
        axs[i, 0].imshow(imgs[i])
        axs[i, 0].axis('off')
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(cams[i])
        axs[i, 1].axis('off')
        axs[i, 1].set_title("Grad-CAM Overlay")

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/vit_gradcam_grid_final.png", dpi=300)
    plt.show()

import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import math

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
raw_tfms = transforms.Compose([transforms.Resize((224, 224))])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=tfms)
raw_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=raw_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

attn_scores = []

def patched_forward(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn_scores.append(attn.detach().cpu())
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.attn_drop(x)
    return x

model.blocks[-1].attn.forward = patched_forward.__get__(model.blocks[-1].attn, type(model.blocks[-1].attn))

n_imgs = 9
imgs, cams = [], []
count = 0

with torch.no_grad():
    for i, (x, _) in enumerate(test_dl):
        if count >= n_imgs:
            break
        input_tensor = x.to(device)
        raw_img = np.array(raw_ds[i][0])
        attn_scores.clear()
        _ = model(input_tensor)

        if len(attn_scores) == 0:
            continue

        attn = attn_scores[0].mean(1)[0]  # average heads, get [tokens x tokens]
        patch_attn = attn[0, 1:].numpy()  # CLS token attention to all patches
        num_patches = patch_attn.shape[0]
        grid_size = int(math.sqrt(num_patches))

        if grid_size * grid_size != num_patches:
            print(f"⚠️ Skipping {num_patches} tokens — cannot reshape.")
            continue

        cls_attn = patch_attn.reshape(grid_size, grid_size)
        cls_attn = cv2.resize(cls_attn, (224, 224))
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)

        imgs.append(raw_img)
        cams.append(overlay)
        count += 1

if len(imgs) == 0:
else:
    fig, axs = plt.subplots(len(imgs), 2, figsize=(6, 2 * len(imgs)))
    for i in range(len(imgs)):
        axs[i, 0].imshow(imgs[i])
        axs[i, 0].axis('off')
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(cams[i])
        axs[i, 1].axis('off')
        axs[i, 1].set_title("Grad-CAM Overlay")

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/vit_gradcam_grid_final.png", dpi=300)
    plt.show()

import torch, cv2, math
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
raw_tfms = transforms.Compose([transforms.Resize((224, 224))])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=tfms)
raw_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=raw_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

attn_scores = []

    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn_scores.append(attn.detach().cpu())
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.attn_drop(x)
    return x

model.blocks[-1].attn.forward = patched_forward.__get__(model.blocks[-1].attn, type(model.blocks[-1].attn))

n_imgs = 9
imgs, cams = [], []
count = 0

with torch.no_grad():
    for i, (x, _) in enumerate(test_dl):
        if count >= n_imgs:
            break
        input_tensor = x.to(device)
        raw_img = np.array(raw_ds[i][0])
        attn_scores.clear()
        _ = model(input_tensor)

        if len(attn_scores) == 0:
            continue

        attn = attn_scores[0].mean(1)[0]  # (tokens, tokens)
        patch_attn = attn[0, 1:].numpy()
        num_patches = patch_attn.shape[0]
        grid_size = int(math.sqrt(num_patches))

        if grid_size * grid_size != num_patches:
            print(f"⚠️ Skipped: {num_patches} tokens not square.")
            continue

        cls_attn = patch_attn.reshape(grid_size, grid_size)
        cls_attn = cv2.resize(cls_attn, (224, 224))
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)

        imgs.append(raw_img)
        cams.append(overlay)
        count += 1

if len(imgs) == 0:
else:
    fig, axs = plt.subplots(len(imgs), 2, figsize=(6, 2 * len(imgs)))
    for i in range(len(imgs)):
        axs[i, 0].imshow(imgs[i])
        axs[i, 0].axis('off')
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(cams[i])
        axs[i, 1].axis('off')
        axs[i, 1].set_title("Grad-CAM Overlay")

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/vit_gradcam_grid_final.png", dpi=300)
    plt.show()

import torch, numpy as np, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
raw_tfms = transforms.Compose([
    transforms.Resize((224, 224))
])

test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
raw_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=raw_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
classes = test_ds.classes

misclassified = []
model.eval()

with torch.no_grad():
    for i, (x, label) in enumerate(test_dl):
        img_raw = raw_ds[i][0]
        x = x.to(device)
        output = model(x)
        pred = output.argmax(dim=1).item()
        true = label.item()
        if pred != true:
            misclassified.append((img_raw, classes[true], classes[pred]))

n_show = min(12, len(misclassified))
if n_show == 0:
else:
    fig, axs = plt.subplots(3, 4, figsize=(16, 10))
    for idx, (img, true_lbl, pred_lbl) in enumerate(misclassified[:12]):
        row, col = divmod(idx, 4)
        axs[row, col].imshow(img)
        axs[row, col].axis("off")
        axs[row, col].set_title(f"True: {true_lbl}\nPred: {pred_lbl}", fontsize=10)

    for i in range(n_show, 12):
        row, col = divmod(i, 4)
        axs[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/vit_misclassified_grid.png", dpi=300)
    plt.show()
    print("📛 Saved misclassified image grid to /content/drive/MyDrive/vit_misclassified_grid.png")

import torch, timm
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=11)
model.load_state_dict(torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt", map_location=device))
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes

all_labels, all_preds = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
df = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

df.plot(kind='bar', figsize=(12,6), ylim=(0,1), colormap='Set2')
plt.title("Per-Class Precision, Recall, and F1-score (ViT)")
plt.ylabel("Score")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/per_class_viT_full.png", dpi=300)
plt.show()


import torch, timm
import numpy as np, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=11)
model.load_state_dict(torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt", map_location=device))
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
classes = test_ds.classes
n_classes = len(classes)

all_labels, all_probs = [], []
with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_probs = np.vstack(all_probs)
y_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    axs[0].plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")
axs[0].plot([0, 1], [0, 1], 'k--')
axs[0].set_title("Multi-Class ROC Curve")
axs[0].set_xlabel("False Positive Rate")
axs[0].set_ylabel("True Positive Rate")
axs[0].legend(loc='lower right')
axs[0].grid(True)

for i in range(n_classes):
    prec, rec, _ = precision_recall_curve(y_true_bin[:, i], all_probs[:, i])
    pr_auc = auc(rec, prec)
    axs[1].plot(rec, prec, label=f"{classes[i]} (AUC = {pr_auc:.2f})")
axs[1].set_title("Multi-Class Precision-Recall Curve")
axs[1].set_xlabel("Recall")
axs[1].set_ylabel("Precision")
axs[1].legend(loc='lower left')
axs[1].grid(True)

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/roc_pr_viT_full.png", dpi=300)
plt.show()


import torch, timm, cv2, numpy as np, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=11)
model.load_state_dict(torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt", map_location=device))
model.eval().to(device)

attn_maps = []
def hook_fn(module, input, output):
    attn_maps.append(output.detach())

for blk in model.blocks[-1:]:
    blk.attn.attn_drop.register_forward_hook(hook_fn)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset_path = "/content/drive/MyDrive/retina_data/retinal_fundus_images/test"
test_ds = datasets.ImageFolder(dataset_path, transform=transform)
class_indices = {v: k for k, v in test_ds.class_to_idx.items()}
class_samples = {}

for img_path, label in test_ds.samples:
    if label not in class_samples:
        class_samples[label] = img_path
    if len(class_samples) == 11:
        break

imgs, heatmaps = [], []
for label, img_path in sorted(class_samples.items()):
    img_orig = Image.open(img_path).convert("RGB").resize((224, 224))
    input_tensor = transform(img_orig).unsqueeze(0).to(device)
    attn_maps.clear()

    _ = model(input_tensor)
    if not attn_maps:
        continue

    attn = attn_maps[0].squeeze(0).mean(0)  # (num_heads → mean)
    cls_attn = attn[0, 1:].reshape(14, 14).cpu().numpy()
    cls_attn = cv2.resize(cls_attn, (224, 224))
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

    img_np = np.array(img_orig) / 255.0
    heat = cv2.applyColorMap(np.uint8(255 * cls_attn), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(255 * img_np), 0.6, heat, 0.4, 0)

    imgs.append(img_np)
    heatmaps.append(overlay / 255.0)

n_classes = len(imgs)  # Actual number of loaded images (should be 11)
fig, axs = plt.subplots(n_classes, 2, figsize=(6, 3 * n_classes))

for i in range(n_classes):
    axs[i, 0].imshow(imgs[i])
    axs[i, 0].axis("off")
    axs[i, 0].set_title(f"Original - {class_indices[i]}")

    axs[i, 1].imshow(heatmaps[i])
    axs[i, 1].axis("off")
    axs[i, 1].set_title(f"Grad-CAM - {class_indices[i]}")

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/vit_gradcam_grid_11classes.png", dpi=300)
plt.show()

import torch, timm, numpy as np, matplotlib.pyplot as plt, cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms.functional import to_tensor
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=11)
model.load_state_dict(torch.load("/content/drive/MyDrive/retina_ckpt_2h/vit_latest.pt", map_location=device))
model.to(device)
model.eval()

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
test_ds = datasets.ImageFolder("/content/drive/MyDrive/retina_data/retinal_fundus_images/test", transform=test_tfms)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
class_names = test_ds.classes

attn_maps = []
def hook_fn(module, input, output):
    attn_maps.append(output)
for blk in model.blocks:
    blk.attn.register_forward_hook(lambda self, input, output: attn_maps.append(output))


class_samples = OrderedDict()
for img_path, label in test_ds.samples:
    if label not in class_samples:
        class_samples[label] = img_path
    if len(class_samples) == 11:
        break

imgs, heatmaps, class_labels = [], [], []
for cls_idx, img_path in class_samples.items():
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    input_tensor = test_tfms(img).unsqueeze(0).to(device)

    attn_maps.clear()
    _ = model(input_tensor)

    try:
        attn = attn_maps[0].squeeze(0).mean(0)
        cls_attn = attn[0, 1:].reshape(14, 14).cpu().detach().numpy()
        cls_attn = cv2.resize(cls_attn, (224, 224))
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

        img_np = np.array(img) / 255.0
        cmap = plt.get_cmap('jet')(cls_attn)[..., :3]
        overlay = (0.5 * img_np + 0.5 * cmap)
        imgs.append(img_np)
        heatmaps.append(overlay)
        class_labels.append(class_names[cls_idx])
    except:
        continue

n = len(imgs)
if n == 0:
else:
    fig, axs = plt.subplots(n, 2, figsize=(6, 3 * n))
    for i in range(n):
        axs[i, 0].imshow(imgs[i])
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f"Original - {class_labels[i]}")

        axs[i, 1].imshow(heatmaps[i])
        axs[i, 1].axis('off')
        axs[i, 1].set_title(f"Grad-CAM - {class_labels[i]}")

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/vit_gradcam_grid_11classes.png", dpi=300)
    plt.show()

from PIL import Image
import matplotlib.pyplot as plt

img_path, label = test_ds.samples[0]
print("Testing image:", img_path)

img = Image.open(img_path).convert("RGB").resize((224, 224))
input_tensor = test_tfms(img).unsqueeze(0).to(device)

attn_maps.clear()

_ = model(input_tensor)

if len(attn_maps) == 0:
else:
    print("Attn shape:", attn_maps[0].shape)  # Expect [1, heads, tokens, tokens]

    attn = attn_maps[0].squeeze(0).mean(0)
    cls_attn = attn[0, 1:].reshape(14, 14).cpu().numpy()
    plt.imshow(cls_attn, cmap='jet')
    plt.title("Sample Grad-CAM Attention")
    plt.show()
