"""Tiny Siamese encoder shared by training (experiments) and inference.

4 conv stages -> 64-d L2-normalised embedding. ~60k parameters, CPU-friendly,
no pretrained weights. Input: 96x96 grayscale patches, per-patch standardised.
"""
from __future__ import annotations

import numpy as np


def make_model(dim: int = 64):
    import torch.nn as nn
    import torch.nn.functional as tF

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.a = nn.Conv2d(c, c, 3, padding=1)
            self.b = nn.Conv2d(c, c, 3, padding=1)
            self.r = nn.ReLU()

        def forward(self, x):
            return self.r(x + self.b(self.r(self.a(x))))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),   # 48
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 24
                Res(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 12
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim))

        def forward(self, x):
            return tF.normalize(self.net(x), dim=1)

    return Encoder()


def prep_u8(u8: np.ndarray):
    """uint8 [N,H,W] -> standardised float tensor [N,1,H,W]."""
    import torch
    x = torch.from_numpy(u8.astype(np.float32) / 255.0).unsqueeze(1)
    return (x - x.mean(dim=(2, 3), keepdim=True)) / (x.std(dim=(2, 3), keepdim=True) + 1e-6)
