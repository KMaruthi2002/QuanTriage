"""Datasets for tumor segmentation.

Two sources:
  * ``FolderSegDataset``, point it at an images/ + masks/ layout (the common
    Kaggle format) and it loads grayscale image/mask pairs. This is how your own
    Kaggle data plugs in.
  * ``make_synthetic``, generates synthetic "scans" with bright blob "tumors"
    and ground-truth masks, purely to validate that the U-Net + MPS training
    pipeline works end-to-end before real medical data is wired in. Clearly NOT
    medical data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FolderSegDataset(Dataset):
    """Grayscale (image, mask) pairs from two parallel folders.

    Files are matched by sorted order; masks are binarized at >0. Images are
    resized to ``size`` and scaled to [0, 1].
    """

    def __init__(self, images_dir, masks_dir, size: int = 128):
        from PIL import Image

        self.Image = Image
        self.size = size
        self.imgs = sorted(Path(images_dir).glob("*"))
        self.masks = sorted(Path(masks_dir).glob("*"))
        if len(self.imgs) != len(self.masks):
            raise ValueError("image/mask counts differ")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.Image.open(self.imgs[i]).convert("L").resize((self.size, self.size))
        msk = self.Image.open(self.masks[i]).convert("L").resize((self.size, self.size))
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None]
        y = torch.from_numpy((np.asarray(msk) > 0).astype(np.float32))[None]
        return x, y


class _TensorSeg(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def make_synthetic(n: int = 240, size: int = 128, seed: int = 0):
    """Synthetic scans: textured background + bright elliptical 'tumor' blobs.

    Returns (train_ds, val_ds). For PIPELINE VALIDATION ONLY, not medical data.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    X = np.zeros((n, 1, size, size), np.float32)
    Y = np.zeros((n, 1, size, size), np.float32)
    for k in range(n):
        bg = rng.normal(0.3, 0.05, (size, size)).astype(np.float32)
        mask = np.zeros((size, size), np.float32)
        for _ in range(rng.integers(1, 3)):
            cy, cx = rng.integers(size * 0.25, size * 0.75, 2)
            ry, rx = rng.integers(8, 22, 2)
            ang = rng.uniform(0, np.pi)
            xr = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
            yr = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
            blob = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1
            bg[blob] += rng.uniform(0.4, 0.6)
            mask[blob] = 1.0
        X[k, 0] = np.clip(bg, 0, 1)
        Y[k, 0] = mask
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    cut = int(n * 0.85)
    return _TensorSeg(Xt[:cut], Yt[:cut]), _TensorSeg(Xt[cut:], Yt[cut:])
