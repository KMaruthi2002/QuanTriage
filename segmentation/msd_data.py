"""Turn the 3-D multimodal MSD Brain-Tumour volumes into 2-D training slices.

The Medical Segmentation Decathlon "Task01_BrainTumour" set stores, per case:
  * a 4-D image  (H, W, D, 4 MRI modalities: FLAIR, T1w, T1gd, T2w)
  * a 3-D label  (H, W, D) with tumor sub-regions (edema / non-enhancing / enhancing)

For a first localizer we take the **FLAIR** modality (best for whole-tumor extent),
binarize the label to **tumor vs. not**, and extract axial slices that contain
tumor. The result plugs into the same U-Net training loop as everything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from seg_data import _TensorSeg


def _resize(arr, size, mask=False):
    from skimage.transform import resize

    order = 0 if mask else 1
    out = resize(arr, (size, size), order=order, preserve_range=True, anti_aliasing=not mask)
    return out


def build_msd_slice_dataset(
    root, n_volumes: int = 60, max_slices_per_vol: int = 24, size: int = 128,
    min_tumor_px: int = 60, modality: int = 0, val_frac: float = 0.15, seed: int = 42,
):
    """Build (train_ds, val_ds) of 2-D (FLAIR slice, tumor mask) pairs.

    The split is done by *volume* (no patient leaks across train/val).
    """
    import nibabel as nib

    root = Path(root)
    img_dir = root / "imagesTr"
    lbl_dir = root / "labelsTr"
    # MSD files start with BRATS_...; skip hidden files
    cases = sorted(p for p in img_dir.glob("*.nii.gz") if not p.name.startswith("."))
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    cases = cases[:n_volumes]

    n_val = max(1, int(len(cases) * val_frac))
    val_cases = set(cases[:n_val])

    tr_X, tr_Y, va_X, va_Y = [], [], [], []
    for path in cases:
        lbl_path = lbl_dir / path.name
        if not lbl_path.exists():
            continue
        img = nib.load(str(path)).get_fdata()           # (H, W, D, 4)
        lbl = nib.load(str(lbl_path)).get_fdata()        # (H, W, D)
        flair = img[..., modality]
        tumor = (lbl > 0).astype(np.float32)
        depth = flair.shape[2]
        tumor_slices = [d for d in range(depth) if tumor[:, :, d].sum() >= min_tumor_px]
        if not tumor_slices:
            continue
        if len(tumor_slices) > max_slices_per_vol:
            tumor_slices = list(rng.choice(tumor_slices, max_slices_per_vol, replace=False))
        bucket_X = va_X if path in val_cases else tr_X
        bucket_Y = va_Y if path in val_cases else tr_Y
        for d in tumor_slices:
            sl = flair[:, :, d].astype(np.float32)
            sl = (sl - sl.min()) / (np.ptp(sl) + 1e-6)
            m = (tumor[:, :, d] > 0).astype(np.float32)
            bucket_X.append(_resize(sl, size)[None])
            bucket_Y.append((_resize(m, size, mask=True) > 0.5).astype(np.float32)[None])

    def stack(xs, ys):
        if not xs:
            return None
        return _TensorSeg(torch.from_numpy(np.array(xs, np.float32)),
                          torch.from_numpy(np.array(ys, np.float32)))

    train_ds, val_ds = stack(tr_X, tr_Y), stack(va_X, va_Y)
    return train_ds, val_ds, {"train_slices": len(tr_X), "val_slices": len(va_X),
                              "volumes": len(cases)}
