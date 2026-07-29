import timm
import torch
from config import MODEL_NAME, NUM_CLASSES


def create_model(num_classes: int = None, pretrained: bool = True):
    num_classes = num_classes or NUM_CLASSES
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
    return model


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    model.to(device)
    return model
