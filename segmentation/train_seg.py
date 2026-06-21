"""Train the U-Net tumor-localization model — GPU-accelerated on Apple Silicon.

Live per-epoch logging (train loss, val loss, val Dice), best-checkpoint saving.

    python segmentation/train_seg.py                       # synthetic pipeline check
    python segmentation/train_seg.py --images_dir IMG --masks_dir MASKS --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seg_data import FolderSegDataset, make_synthetic
from unet import UNet, get_device

CKPT_DIR = Path(__file__).resolve().parent.parent / "results" / "checkpoints"


def dice_score(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def dice_loss(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def train(train_ds, val_ds, epochs=25, batch_size=16, lr=1e-3, base=32,
          device=None, on_epoch=None, ckpt_name="unet_best.pt"):
    """Train a U-Net. `on_epoch(stats: dict)` is called after each epoch (for a
    live UI). Returns (model, history)."""
    device = device or get_device()
    model = UNet(base=base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
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
            loss = bce(out, y) + dice_loss(out, y)
            loss.backward()
            opt.step()
            run += loss.item() * len(x)
        train_loss = run / len(train_ds)

        model.eval()
        vloss, vdice, nb = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vloss += (bce(out, y) + dice_loss(out, y)).item()
                vdice += dice_score(out, y)
                nb += 1
        vloss, vdice = vloss / nb, vdice / nb

        if vdice > best:
            best = vdice
            torch.save({"model": model.state_dict(), "base": base, "epoch": epoch,
                        "val_dice": vdice}, CKPT_DIR / ckpt_name)
        stats = {"epoch": epoch, "epochs": epochs, "train_loss": train_loss,
                 "val_loss": vloss, "val_dice": vdice, "best_dice": best}
        history.append(stats)
        if on_epoch:
            on_epoch(stats)
        else:
            print(f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={vloss:.4f}  val_Dice={vdice:.3f}  (best {best:.3f})")
    return model, history


def main():
    ap = argparse.ArgumentParser(description="Train U-Net tumor segmentation")
    ap.add_argument("--images_dir")
    ap.add_argument("--masks_dir")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    dev = get_device()
    print(f"Device: {dev}  (Apple-Silicon GPU)" if dev.type == "mps" else f"Device: {dev}")
    if args.images_dir and args.masks_dir:
        full = FolderSegDataset(args.images_dir, args.masks_dir, size=args.size)
        cut = int(len(full) * 0.85)
        train_ds = torch.utils.data.Subset(full, range(cut))
        val_ds = torch.utils.data.Subset(full, range(cut, len(full)))
        print(f"Real dataset: {len(full)} image/mask pairs")
    else:
        print("No dataset given — using SYNTHETIC data to validate the pipeline.")
        train_ds, val_ds = make_synthetic(size=args.size)

    _, history = train(train_ds, val_ds, epochs=args.epochs,
                       batch_size=args.batch_size, base=args.base, device=dev)
    print(f"\nBest val Dice: {max(h['val_dice'] for h in history):.3f}")
    print(f"Checkpoint -> {CKPT_DIR / 'unet_best.pt'}")


if __name__ == "__main__":
    main()
