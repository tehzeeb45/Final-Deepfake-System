"""New ViT-Large/CLIP wrapper — trained from scratch on DeepShield dataset.

Architecture : timm 'vit_large_patch14_clip_224', num_classes=2
Checkpoint   : best_vit_large_clip.pth
               keys: 'state_dict', 'val_auc'

Normalization: NONE — training notebook used raw float() conversion only.
               (IndexedDeepShieldDataset: img = torch.from_numpy(...).float())
               Input images must be in [0, 1] float range (CHW). No mean/std
               normalization is applied — matches training exactly.

P(Fake) = softmax(logits, dim=1)[:, 1]

Note: ~307M-parameter ViT-Large; inference is run in bs=16 chunks to avoid OOM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from ..config import Config, resolve

# Exact timm tag used during training (no suffix variant)
_TIMM_TAG = "vit_large_patch14_clip_224"


class NewViTLargeClip(nn.Module):
    """timm ViT-Large/14 CLIP with 2-class classification head.

    NO normalization applied — matches training exactly.
    Callers pass plain [0, 1] float tensors (CHW).
    """

    def __init__(self):
        super().__init__()
        import timm
        self.vit = timm.create_model(_TIMM_TAG, pretrained=False, num_classes=2)
        self.target_size = 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) float tensor in [0, 1] range. No normalization applied."""
        return self.vit(x)

    def predict_proba(self, faces) -> np.ndarray:
        """Return per-frame P(Fake) scores (batched, bs=16).

        Accepts:
          - np.ndarray of shape (N, C, H, W) in [0, 1] float32
          - list of (H, W, C) uint8 face crops
        """
        if isinstance(faces, np.ndarray) and faces.ndim == 4:
            all_frames = torch.from_numpy(faces).float()
        else:
            frames = []
            for f in faces:
                t = torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
                frames.append(t)
            all_frames = torch.stack(frames)

        device = next(self.parameters()).device
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(all_frames), 16):
                batch = all_frames[start:start + 16].to(device)
                logits = self(batch)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs)


def load_new_vit_large_clip(cfg: Config, device=None):
    """Load the DeepShield-trained ViT-Large/CLIP checkpoint."""
    from . import get_device

    ckpt_path = resolve(cfg.models.new_vit_large_clip.checkpoint)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"new_vit_large_clip checkpoint missing: {ckpt_path}\n"
            "Put best_vit_large_clip.pth in models/."
        )

    model = NewViTLargeClip()

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and "ema_state_dict" in raw:
        sd = raw["ema_state_dict"]
        print("[new_vit_large_clip] Loaded EMA weights from checkpoint.")
    elif isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        print("[new_vit_large_clip] Loaded state_dict weights from checkpoint.")
    else:
        sd = raw
        print("[new_vit_large_clip] WARNING: raw checkpoint used (no state_dict key found).")

    missing, unexpected = model.vit.load_state_dict(sd, strict=True)
    if missing:
        raise RuntimeError(f"Missing new_vit_large_clip weights: {missing[:5]}")

    device = device or get_device(cfg.models.device)
    model.to(device).eval()
    return model
