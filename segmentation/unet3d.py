"""A 3-D U-Net for volumetric brain-tumor segmentation.

The same encoder/decoder idea as the 2-D model, but with 3-D convolutions so the
network sees whole sub-volumes at once and learns through-plane context (how the
tumor connects across slices). Trained on patches because full volumes are large.
Runs on the Apple-Silicon GPU via PyTorch's MPS backend.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv3d(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 4, base: int = 16):
        super().__init__()
        self.d1 = _DoubleConv3d(in_ch, base)
        self.d2 = _DoubleConv3d(base, base * 2)
        self.d3 = _DoubleConv3d(base * 2, base * 4)
        self.bottleneck = _DoubleConv3d(base * 4, base * 8)
        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, 2, stride=2)
        self.u3 = _DoubleConv3d(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.u2 = _DoubleConv3d(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.u1 = _DoubleConv3d(base * 2, base)
        self.head = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        b = self.bottleneck(self.pool(c3))
        x = self.u3(torch.cat([self.up3(b), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.head(x)
