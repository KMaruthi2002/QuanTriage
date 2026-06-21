"""Multi-modal, multi-class brain-tumor segmentation (2-D).

Upgrades the FLAIR-only / binary localizer to the real clinical setting:

  * **4 input modalities**, FLAIR, T1, T1ce (t1gd), T2, stacked as channels.
  * **4 output classes**, background, edema, non-enhancing, enhancing tumor.

Reported with the standard BraTS regions computed from the predicted class map:
  * WT (whole tumor)  = {edema, non-enh, enh}      labels {1,2,3}
  * TC (tumor core)   = {non-enh, enh}             labels {2,3}
  * ET (enhancing)    = {enh}                      labels {3}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from qml_oncology.segmentation.unet import UNet, get_device

CKPT_DIR = Path.cwd() / "results" / "checkpoints"
REGIONS = {"WT": (1, 2, 3), "TC": (2, 3), "ET": (3,)}


class _MMSeg(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y  # X:(N,4,H,W) float32, Y:(N,H,W) int64

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def _norm_modality(vol):
    """Z-score a modality over its non-zero (brain) voxels."""
    m = vol[vol > 0]
    if m.size == 0:
        return vol
    return (vol - m.mean()) / (m.std() + 1e-6)


def build_multimodal_dataset(root, n_volumes=60, max_slices_per_vol=20, size=128,
                             min_tumor_px=80, val_frac=0.15, seed=42):
    import nibabel as nib
    from skimage.transform import resize

    root = Path(root)
    cases = sorted(p for p in (root / "imagesTr").glob("*.nii.gz")
                   if not p.name.startswith("."))
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    cases = cases[:n_volumes]
    n_val = max(1, int(len(cases) * val_frac))
    val_set = set(cases[:n_val])

    tr_X, tr_Y, va_X, va_Y = [], [], [], []
    for path in cases:
        lbl_path = root / "labelsTr" / path.name
        if not lbl_path.exists():
            continue
        img = nib.load(str(path)).get_fdata()          # (H,W,D,4)
        lbl = nib.load(str(lbl_path)).get_fdata().astype(np.int64)  # (H,W,D)
        mods = [_norm_modality(img[..., c]) for c in range(4)]
        depth = lbl.shape[2]
        slices = [d for d in range(depth) if (lbl[:, :, d] > 0).sum() >= min_tumor_px]
        if not slices:
            continue
        if len(slices) > max_slices_per_vol:
            slices = list(rng.choice(slices, max_slices_per_vol, replace=False))
        is_val = path in val_set
        bX = va_X if is_val else tr_X
        bY = va_Y if is_val else tr_Y
        for d in slices:
            chans = np.stack([resize(m[:, :, d], (size, size), preserve_range=True,
                                     anti_aliasing=True) for m in mods]).astype(np.float32)
            y = resize(lbl[:, :, d], (size, size), order=0, preserve_range=True,
                       anti_aliasing=False).astype(np.int64)
            bX.append(chans)
            bY.append(y)

    def stack(xs, ys):
        return _MMSeg(torch.from_numpy(np.array(xs, np.float32)),
                      torch.from_numpy(np.array(ys, np.int64)))

    return stack(tr_X, tr_Y), stack(va_X, va_Y), {
        "train": len(tr_X), "val": len(va_X), "volumes": len(cases)}


def region_dice(pred_cls, target_cls, labels):
    p = np.isin(pred_cls, labels)
    g = np.isin(target_cls, labels)
    s = p.sum() + g.sum()
    return 2 * (p & g).sum() / (s + 1e-6) if s else 1.0


def _dice_loss_mc(logits, target, n_cls, eps=1.0):
    """Soft multi-class Dice loss over foreground classes (handles imbalance)."""
    prob = torch.softmax(logits, 1)
    oh = torch.nn.functional.one_hot(target, n_cls).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * oh).sum(dims)
    union = prob.sum(dims) + oh.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice[1:].mean()  # ignore background


def train_multimodal(train_ds, val_ds, epochs=25, batch_size=12, lr=1e-3, base=32,
                     device=None, on_epoch=None, ckpt_name="unet_multimodal.pt"):
    device = device or get_device()
    model = UNet(in_ch=4, out_ch=4, base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=batch_size)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        run = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = ce(out, y) + _dice_loss_mc(out, y, 4)
            loss.backward()
            opt.step()
            run += loss.item() * len(x)
        train_loss = run / len(train_ds)

        model.eval()
        reg = {r: [] for r in REGIONS}
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(device)).argmax(1).cpu().numpy()
                yt = y.numpy()
                for b in range(len(pred)):
                    for r, labs in REGIONS.items():
                        reg[r].append(region_dice(pred[b], yt[b], labs))
        dice = {r: float(np.mean(v)) for r, v in reg.items()}
        mean_dice = float(np.mean(list(dice.values())))
        if mean_dice > best:
            best = mean_dice
            torch.save({"model": model.state_dict(), "base": base, "in_ch": 4,
                        "out_ch": 4, "dice": dice}, CKPT_DIR / ckpt_name)
        stats = {"epoch": epoch, "epochs": epochs, "train_loss": train_loss,
                 "WT": dice["WT"], "TC": dice["TC"], "ET": dice["ET"],
                 "mean_dice": mean_dice, "best": best}
        history.append(stats)
        if on_epoch:
            on_epoch(stats)
        else:
            print(f"epoch {epoch:2d}/{epochs}  loss={train_loss:.3f}  "
                  f"WT={dice['WT']:.3f} TC={dice['TC']:.3f} ET={dice['ET']:.3f} "
                  f"(best mean {best:.3f})")
    return model, history


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--msd_root", default="data_cache/msd/Task01_BrainTumour")
    ap.add_argument("--n_volumes", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=22)
    args = ap.parse_args()
    dev = get_device()
    print(f"Device: {dev}")
    print(f"Extracting multi-modal slices from {args.n_volumes} volumes…")
    tr, va, info = build_multimodal_dataset(args.msd_root, n_volumes=args.n_volumes)
    print(f"  {info['volumes']} volumes -> {info['train']} train / {info['val']} val slices")
    _, hist = train_multimodal(tr, va, epochs=args.epochs, device=dev)
    b = max(hist, key=lambda h: h["mean_dice"])
    print(f"\nBest: WT={b['WT']:.3f}  TC={b['TC']:.3f}  ET={b['ET']:.3f}")
