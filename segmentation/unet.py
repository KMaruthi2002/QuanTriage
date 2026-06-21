"""A compact 2-D U-Net for tumor localization (segmentation).

U-Net is the standard architecture for biomedical segmentation: an encoder that
downsamples to capture context, a decoder that upsamples to recover resolution,
and skip connections that preserve fine detail. The output is a per-pixel map of
"is this pixel tumor?" — i.e. *where* the cancer is.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """A small U-Net. `base` controls width (capacity vs. speed)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.d1 = _DoubleConv(in_ch, base)
        self.d2 = _DoubleConv(base, base * 2)
        self.d3 = _DoubleConv(base * 2, base * 4)
        self.bottleneck = _DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        b = self.bottleneck(self.pool(c3))
        x = self.u3(torch.cat([self.up3(b), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.head(x)  # logits; apply sigmoid for probabilities


def get_device() -> torch.device:
    """Prefer Apple-Silicon GPU (MPS), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
