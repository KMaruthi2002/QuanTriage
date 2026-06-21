"""Run the shipped multi-modal U-Net on a real MSD volume and return a 3-D
predicted-tumor render, used to wire localization into the app's 3-D tab.

Gracefully no-ops if the MSD data isn't present locally (it's a 7 GB download,
not shipped in the repo); the app then falls back to the shipped static images.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
MSD_ROOT = ROOT / "data_cache" / "msd" / "Task01_BrainTumour"
MM_CKPT = ROOT / "models" / "unet_multimodal_brats.pt"



def has_msd() -> bool:
    return (MSD_ROOT / "imagesTr").exists() and MM_CKPT.exists()


def list_cases(n: int = 12) -> list[str]:
    cases = sorted(p.name for p in (MSD_ROOT / "imagesTr").glob("*.nii.gz")
                   if not p.name.startswith("."))
    return cases[:n]


def predict_wt(case_name: str):
    """Return (flair, whole_tumor_mask, spacing) for one MSD case."""
    import nibabel as nib
    import torch
    from skimage.transform import resize

    from qml_oncology.segmentation.multimodal import _norm_modality
    from qml_oncology.segmentation.unet import UNet, get_device

    dev = get_device()
    ck = torch.load(str(MM_CKPT), map_location=dev)
    model = UNet(in_ch=4, out_ch=4, base=ck["base"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    path = MSD_ROOT / "imagesTr" / case_name
    img = nib.load(str(path)).get_fdata()
    spacing = np.array(nib.load(str(path)).header.get_zooms()[:3], dtype=float)
    flair = img[..., 0].astype(np.float32)
    mods = [_norm_modality(img[..., c]) for c in range(4)]
    H, W, D = flair.shape

    pred = np.zeros((H, W, D), np.uint8)
    for d in range(D):
        chans = np.stack([resize(m[:, :, d], (128, 128), preserve_range=True,
                                 anti_aliasing=True) for m in mods]).astype(np.float32)
        with torch.no_grad():
            cls = model(torch.from_numpy(chans)[None].to(dev)).argmax(1)[0].cpu().numpy()
        pred[:, :, d] = (resize(cls, (H, W), order=0, preserve_range=True) > 0).astype(np.uint8)
    return flair, pred, spacing


def predicted_figure(case_name: str):
    """Interactive 3-D figure of the predicted tumor on the brain for a case."""
    from qml_oncology.segmentation.predict_seg import render_predicted_3d

    flair, mask, spacing = predict_wt(case_name)
    fig = render_predicted_3d(flair, mask, spacing)
    voxels = int(mask.sum())
    vol_mm3 = voxels * float(np.prod(spacing))
    return fig, {"case": case_name, "tumor_voxels": voxels, "tumor_volume_mm3": round(vol_mm3, 1)}
