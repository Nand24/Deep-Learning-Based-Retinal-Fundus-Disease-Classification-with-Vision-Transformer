import torch
import numpy as np


def get_attention_rollout(model, x, discard_ratio=0.0):
    """Compute a simple attention rollout for ViT models from timm."""
    model.eval()
    x = x.to(next(model.parameters()).device)

    if not hasattr(model, "blocks"):
        raise AttributeError("Model does not expose transformer blocks for attention rollout.")

    # forward through patch embedding and positional encoding
    x = model.patch_embed(x)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    attn_matrices = []
    for block in model.blocks:
        x, attn = block(x, return_attention=True)
        attn = attn.mean(dim=1)
        attn = attn[:, :, 0, 1:]
        attn_matrices.append(attn.detach())

    rollout = torch.eye(attn_matrices[0].size(-1) + 1, device=x.device)
    for attn in attn_matrices:
        attn = attn + torch.eye(attn.size(-1), device=attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout

    return rollout[0, 0].reshape(int(np.sqrt(rollout.size(-1))), -1).cpu().numpy()


def normalize_map(cam):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam
