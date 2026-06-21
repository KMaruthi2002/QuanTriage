"""Run the trained U-Net to localize tumors, and visualize *where* they are.

  * ``segment_image``, per-pixel tumor probability for one 2-D scan slice.
  * ``segment_volume``, segment a whole 3-D volume slice-by-slice (-> 3-D mask).
  * ``render_predicted_3d``, 3-D surface of the PREDICTED tumor (reuses the
    imaging renderer), so a predicted mask flows straight into the 3-D view.
  * ``demo_overlay``, sanity panel (input / ground truth / prediction).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "imaging"))
from unet import UNet, get_device  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
# prefer the shipped real-MRI-trained model; fall back to a freshly trained one
CKPT = _ROOT / "models" / "unet_brats.pt"
if not CKPT.exists():
    CKPT = _ROOT / "results" / "checkpoints" / "unet_best.pt"


def load_model(ckpt_path=CKPT, device=None):
    device = device or get_device()
    ck = torch.load(ckpt_path, map_location=device)
    model = UNet(base=ck.get("base", 32)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, device, ck


def segment_image(model, image2d: np.ndarray, device, size: int = 128) -> np.ndarray:
    """Tumor probability map (same H×W as input) for one grayscale slice."""
    h, w = image2d.shape
    img = image2d.astype(np.float32)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    x = torch.from_numpy(img)[None, None]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))
    prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
    return prob[0, 0].cpu().numpy()


def segment_volume(model, volume: np.ndarray, device, axis: int = 2, thr: float = 0.5):
    """Segment a 3-D volume slice-by-slice along `axis`; return a binary mask."""
    vol = np.moveaxis(volume, axis, 0)
    out = np.stack([segment_image(model, sl, device) for sl in vol])
    return np.moveaxis((out > thr).astype(np.uint8), 0, axis)


def render_predicted_3d(img, pred_mask, spacing, show_organ: bool = True):
    """Build a 3-D Plotly figure of the predicted tumor ON the organ.

    Reuses the imaging renderer: a translucent organ surface for context plus the
    predicted tumor mesh, colored by MRI intensity, so you see *where* the
    predicted tumor sits inside the organ.
    """
    import plotly.graph_objects as go
    from tumor3d import _vertex_intensity, brain_context_mesh, tumor_mesh

    data = []
    if show_organ:
        ctx = brain_context_mesh(img, spacing)
        if ctx is not None:
            data.append(ctx)
    verts, faces = tumor_mesh(pred_mask, spacing)
    vcol = _vertex_intensity(img, verts, spacing)
    data.append(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=vcol, colorscale="Hot", showscale=True, name="predicted tumor"))
    fig = go.Figure(data=data)
    fig.update_layout(title="Predicted tumor on the organ (U-Net)", height=600,
                      scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0))
    return fig


def demo_overlay(out_path):
    """Predict synthetic val images and save an input/truth/prediction panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from seg_data import make_synthetic

    _, val = make_synthetic(seed=1)
    model, device, ck = load_model()
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for row in range(3):
        x, y = val[row]
        prob = segment_image(model, x[0].numpy(), device)
        for ax, im, title in zip(
            axes[row],
            [x[0].numpy(), y[0].numpy(), prob > 0.5],
            ["input scan", "ground truth", "U-Net prediction"],
        ):
            ax.imshow(im, cmap="gray" if title == "input scan" else "magma")
            ax.set_title(title if row == 0 else "")
            ax.axis("off")
    fig.suptitle(f"Tumor localization, best val Dice {ck['val_dice']:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return ck["val_dice"]


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "assets" / "segmentation_demo.png"
    dice = demo_overlay(out)
    print(f"val Dice {dice:.3f} -> saved localization panel to {out}")
