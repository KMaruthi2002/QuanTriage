"""Train the 3-D U-Net on MSD brain-tumour volumes (patch-based, multi-modal).

Volumes are preloaded and normalized; training samples tumor-centered 3-D patches.
Validation uses sliding-window inference over whole volumes and reports the BraTS
regions (WT/TC/ET). GPU-accelerated on Apple Silicon (MPS).

    python segmentation/train3d.py --n_volumes 24 --epochs 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from multimodal import REGIONS, _norm_modality, region_dice
from unet import get_device
from unet3d import UNet3D

CKPT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
P = 96  # patch size


def _load_volumes(root, n_volumes, seed):
    import nibabel as nib

    root = Path(root)
    cases = sorted(p for p in (root / "imagesTr").glob("*.nii.gz")
                   if not p.name.startswith("."))
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    vols = []
    for path in cases[:n_volumes]:
        lbl_p = root / "labelsTr" / path.name
        if not lbl_p.exists():
            continue
        img = nib.load(str(path)).get_fdata()
        lbl = nib.load(str(lbl_p)).get_fdata().astype(np.int8)
        mods = np.stack([_norm_modality(img[..., c]) for c in range(4)]).astype(np.float16)
        vols.append((mods, lbl))  # mods:(4,H,W,D)  lbl:(H,W,D)
    return vols


def _crop(center, dim, p):
    s = int(np.clip(center - p // 2, 0, max(0, dim - p)))
    return s, s + p


class PatchDS(Dataset):
    def __init__(self, vols, n_patches, seed=0):
        self.vols, self.n = vols, n_patches
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        mods, lbl = self.vols[self.rng.integers(len(self.vols))]
        H, W, D = lbl.shape
        tum = np.argwhere(lbl > 0)
        if len(tum) and self.rng.random() < 0.8:
            cy, cx, cz = tum[self.rng.integers(len(tum))]
        else:
            cy, cx, cz = (self.rng.integers(H), self.rng.integers(W), self.rng.integers(D))
        ys, ye = _crop(cy, H, P)
        xs, xe = _crop(cx, W, P)
        zs, ze = _crop(cz, D, P)
        x = mods[:, ys:ye, xs:xe, zs:ze].astype(np.float32)
        y = lbl[ys:ye, xs:xe, zs:ze].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def _dice_loss3d(logits, target, n_cls=4, eps=1.0):
    prob = torch.softmax(logits, 1)
    oh = torch.nn.functional.one_hot(target, n_cls).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = (prob * oh).sum(dims)
    union = prob.sum(dims) + oh.sum(dims)
    return 1 - ((2 * inter + eps) / (union + eps))[1:].mean()


@torch.no_grad()
def predict_volume(model, mods, device, stride=64):
    """Sliding-window inference over a full volume -> predicted class map."""
    model.eval()
    _, H, W, D = mods.shape
    logits = torch.zeros(4, H, W, D)
    counts = torch.zeros(1, H, W, D)
    ys = list(range(0, max(1, H - P + 1), stride)) + [H - P]
    xs = list(range(0, max(1, W - P + 1), stride)) + [W - P]
    zs = list(range(0, max(1, D - P + 1), stride)) + [D - P]
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            for z in sorted(set(zs)):
                patch = mods[:, y:y + P, x:x + P, z:z + P].astype(np.float32)
                out = model(torch.from_numpy(patch)[None].to(device)).cpu()[0]
                logits[:, y:y + P, x:x + P, z:z + P] += out
                counts[:, y:y + P, x:x + P, z:z + P] += 1
    return (logits / counts.clamp(min=1)).argmax(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msd_root", default="data_cache/msd/Task01_BrainTumour")
    ap.add_argument("--n_volumes", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patches_per_epoch", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--base", type=int, default=16)
    args = ap.parse_args()

    dev = get_device()
    print(f"Device: {dev}")
    print(f"Preloading {args.n_volumes} volumes…")
    vols = _load_volumes(args.msd_root, args.n_volumes, seed=42)
    n_val = max(2, len(vols) // 6)
    train_vols, val_vols = vols[n_val:], vols[:n_val]
    print(f"  {len(train_vols)} train / {len(val_vols)} val volumes")

    model = UNet3D(in_ch=4, out_ch=4, base=args.base).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    ce = nn.CrossEntropyLoss()
    ds = PatchDS(train_vols, args.patches_per_epoch)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        run = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = ce(out, y) + _dice_loss3d(out, y)
            loss.backward()
            opt.step()
            run += loss.item()
        # validation: sliding-window region Dice
        reg = {r: [] for r in REGIONS}
        for mods, lbl in val_vols:
            pred = predict_volume(model, mods, dev)
            for r, labs in REGIONS.items():
                reg[r].append(region_dice(pred, lbl, labs))
        dice = {r: float(np.mean(v)) for r, v in reg.items()}
        mean = float(np.mean(list(dice.values())))
        if mean > best:
            best = mean
            torch.save({"model": model.state_dict(), "base": args.base,
                        "dice": dice}, CKPT_DIR / "unet3d_brats.pt")
        print(f"epoch {epoch:2d}/{args.epochs}  loss={run / len(dl):.3f}  "
              f"WT={dice['WT']:.3f} TC={dice['TC']:.3f} ET={dice['ET']:.3f} "
              f"(best mean {best:.3f})")
    print(f"\nBest mean region Dice: {best:.3f}  ->  {CKPT_DIR / 'unet3d_brats.pt'}")


if __name__ == "__main__":
    main()
