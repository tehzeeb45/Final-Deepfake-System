"""New EfficientNet-B5 wrapper — trained from scratch on DeepShield dataset.

Architecture: timm 'tf_efficientnet_b5.ns_jft_in1k', num_classes=2 (Real=0, Fake=1).
Checkpoint format: {'state_dict': ..., 'val_auc': ...}
P(Fake) = softmax(logits, dim=1)[:, 1]

Input: 456x456, ImageNet mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config, resolve

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class NewEfficientNetB5(nn.Module):
    """timm EfficientNet-B5 with a 2-class classifier head.

    Input size: 456x456, ImageNet normalised.
    """

    def __init__(self):
        super().__init__()
        import timm
        self.net = timm.create_model(
            "tf_efficientnet_b5.ns_jft_in1k", pretrained=False, num_classes=2
        )
        self.target_size = 456

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ImageNet normalisation: mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)
        mean = _IMAGENET_MEAN.to(x.device)
        std  = _IMAGENET_STD.to(x.device)
        x = (x - mean) / std
        return self.net(x)

    def predict_proba(self, faces) -> np.ndarray:
        """Return per-frame P(Fake) scores as a numpy array.

        Accepts [N,3,H,W] float32 (0..1) or list of HWC uint8 crops.
        forward() applies ImageNet normalisation internally.
        """
        if isinstance(faces, np.ndarray) and faces.ndim == 4:
            tensor = torch.from_numpy(faces).float()
        else:
            frames = []
            for f in faces:
                t = torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
                frames.append(t)
            tensor = torch.stack(frames)

        tensor = tensor.to(next(self.parameters()).device)
        with torch.no_grad():
            logits = self.forward(tensor)   # norm applied inside forward()
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy()

    @property
    def grad_cam_target_layer(self) -> nn.Module:
        return self.net.conv_head


def load_new_efficientnet(cfg: Config, device=None):
    """Load the new DeepShield-trained EfficientNet-B5 checkpoint."""
    from . import get_device

    ckpt_path = resolve(cfg.models.new_efficientnet.checkpoint)
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(
            f"new_efficientnet checkpoint missing: {ckpt_path}\n"
            "Put best_efficientnet_b5.pth in the models/results(Eff)/ folder."
        )

    model = NewEfficientNetB5()
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    missing, unexpected = model.net.load_state_dict(sd, strict=True)
    if missing:
        raise RuntimeError(f"Missing new_efficientnet weights: {missing[:5]}")

    device = device or get_device(cfg.models.device)
    model.to(device).eval()
    return model
