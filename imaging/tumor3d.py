"""Real medical-imaging tumor rendering.

Loads an actual brain-MRI volume with a radiologist-drawn tumor segmentation
(the open ``brain1`` sample shipped with pyRadiomics), reconstructs the tumor as
a true 3-D surface via marching cubes, quantifies its "state" with real radiomics
features (volume, surface area, sphericity, elongation, intensity heterogeneity),
and builds an interactive, animated Plotly figure.

This is a *real* segmentation rendered in 3-D -- not a stylized shape. What it is
NOT: a quantum prediction on this scan. Connecting the QML model to imaging is a
separate research step (see the project README roadmap); here the quantum model
remains the tabular diagnostic and the imaging module is an honest tumor viewer +
quantifier.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data"
_BASE = "https://raw.githubusercontent.com/AIM-Harvard/pyradiomics/master/data"
_FILES = {"image": "brain1_image.nrrd", "label": "brain1_label.nrrd"}


def ensure_sample() -> tuple[Path, Path]:
    """Download the real MRI + tumor mask on first use (≈2 MB), then cache."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in _FILES.items():
        p = SAMPLE_DIR / name
        if not p.exists():
            urllib.request.urlretrieve(f"{_BASE}/{name}", p)
        paths[key] = p
    return paths["image"], paths["label"]


def load_volume():
    """Return (image, label, spacing_mm) for the real scan."""
    import nrrd

    img_path, lbl_path = ensure_sample()
    img, hdr = nrrd.read(str(img_path))
    lbl, _ = nrrd.read(str(lbl_path))
    # voxel spacing (mm) = norm of each row of the 'space directions' matrix
    sd = hdr.get("space directions")
    if sd is not None:
        sd = np.array(sd, dtype=float)
        spacing = np.linalg.norm(np.nan_to_num(sd), axis=1)
        spacing = np.where(spacing > 0, spacing, 1.0)
    else:
        spacing = np.ones(3)
    return img.astype(np.float32), (lbl > 0).astype(np.uint8), spacing


def tumor_mesh(label: np.ndarray, spacing: np.ndarray):
    """Marching-cubes surface of the tumor in physical (mm) coordinates."""
    from skimage import measure

    verts, faces, _normals, _vals = measure.marching_cubes(
        label.astype(float), level=0.5, spacing=tuple(spacing)
    )
    return verts, faces


def _surface_area(verts: np.ndarray, faces: np.ndarray) -> float:
    tris = verts[faces]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def tumor_features(img, label, spacing, verts, faces) -> dict:
    """Real radiomics-style quantification of the tumor 'state'."""
    voxel_vol = float(np.prod(spacing))
    n_vox = int(label.sum())
    volume = n_vox * voxel_vol
    area = _surface_area(verts, faces)
    # sphericity: 1.0 = perfect sphere; lower = more irregular (a malignancy cue)
    sphericity = (np.pi ** (1 / 3) * (6 * volume) ** (2 / 3)) / area if area else 0.0

    # shape anisotropy via PCA of the tumor voxel cloud (scaled to mm)
    coords = np.argwhere(label > 0).astype(float) * spacing
    coords -= coords.mean(0)
    eig = np.sort(np.linalg.eigvalsh(np.cov(coords.T)))[::-1]
    eig = np.clip(eig, 1e-9, None)
    elongation = float(np.sqrt(eig[1] / eig[0]))
    flatness = float(np.sqrt(eig[2] / eig[0]))

    inten = img[label > 0]
    return {
        "tumor_volume_mm3": round(volume, 1),
        "surface_area_mm2": round(area, 1),
        "sphericity": round(float(sphericity), 3),
        "elongation": round(elongation, 3),
        "flatness": round(flatness, 3),
        "voxel_count": n_vox,
        "mean_intensity": round(float(inten.mean()), 1),
        "intensity_heterogeneity": round(float(inten.std()), 1),
        "max_diameter_mm": round(float(np.linalg.norm(coords.max(0) - coords.min(0))), 1),
    }


def _vertex_intensity(img, verts, spacing):
    """Sample MRI intensity at each mesh vertex (for surface coloring)."""
    idx = np.round(verts / spacing).astype(int)
    for a in range(3):
        idx[:, a] = np.clip(idx[:, a], 0, img.shape[a] - 1)
    return img[idx[:, 0], idx[:, 1], idx[:, 2]]


def brain_context_mesh(img, spacing, opacity: float = 0.12):
    """A translucent organ (brain) surface for context, downsampled to stay light.

    Returns a Plotly Mesh3d (or None if it can't be built), so a tumor can be
    shown sitting *on the organ* rather than floating in space.
    """
    import plotly.graph_objects as go
    from skimage import measure

    ds = img[::2, ::2, :]
    sp_ds = spacing * np.array([2, 2, 1])
    thr = float(np.percentile(img[img > 0], 55))
    try:
        bv, bf, _, _ = measure.marching_cubes(ds, level=thr, spacing=tuple(sp_ds))
    except (ValueError, RuntimeError):
        return None
    return go.Mesh3d(
        x=bv[:, 0], y=bv[:, 1], z=bv[:, 2],
        i=bf[:, 0], j=bf[:, 1], k=bf[:, 2],
        color="lightgray", opacity=opacity, hoverinfo="skip",
        name="organ", showscale=False,
    )


def build_figure(animate: bool = True, show_brain: bool = True):
    """Interactive, animated Plotly figure of the real tumor (+ brain context)."""
    import plotly.graph_objects as go
    from skimage import measure

    img, label, spacing = load_volume()
    verts, faces = tumor_mesh(label, spacing)
    feats = tumor_features(img, label, spacing, verts, faces)
    vcol = _vertex_intensity(img, verts, spacing)

    data = []

    # translucent brain context (downsampled so it stays light)
    if show_brain:
        ctx = brain_context_mesh(img, spacing)
        if ctx is not None:
            data.append(ctx)

    # the tumor itself, surface-colored by MRI intensity
    data.append(
        go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=vcol, colorscale="Hot", opacity=1.0,
            name="tumor", showscale=True,
            colorbar=dict(title="MRI<br>intensity"),
            lighting=dict(ambient=0.45, diffuse=0.8, specular=0.3),
        )
    )

    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"Real brain-tumor segmentation · volume {feats['tumor_volume_mm3']:.0f} mm³ · "
              f"sphericity {feats['sphericity']:.2f}",
        scene=dict(
            xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )

    # rotation animation: orbit the camera around the tumor
    if animate:
        center = verts.mean(0)
        r = 2.2
        frames = []
        for t in np.linspace(0, 2 * np.pi, 48, endpoint=False):
            frames.append(
                go.Frame(
                    layout=dict(
                        scene_camera=dict(
                            eye=dict(x=r * np.cos(t), y=r * np.sin(t), z=0.7)
                        )
                    )
                )
            )
        fig.frames = frames
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons", showactive=False, x=0.05, y=0.05,
                    buttons=[
                        dict(
                            label="▶ Rotate",
                            method="animate",
                            args=[None, dict(frame=dict(duration=70, redraw=True),
                                             transition=dict(duration=0),
                                             fromcurrent=True, mode="immediate")],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[[None], dict(frame=dict(duration=0, redraw=False),
                                               mode="immediate")],
                        ),
                    ],
                )
            ]
        )
    return fig, feats


def write_html(path: Path):
    """Save a standalone interactive HTML you can open in any browser."""
    fig, feats = build_figure(animate=True, show_brain=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", auto_play=False)
    return feats


if __name__ == "__main__":
    out = ROOT.parent / "results" / "tumor_3d.html"
    feats = write_html(out)
    print("Real tumor radiomics:")
    for k, v in feats.items():
        print(f"  {k:24} {v}")
    print(f"\nWrote interactive 3-D render -> {out}")
