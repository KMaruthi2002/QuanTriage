"""Bridge: connect the variational quantum classifier to the 3-D imaging module.

The breast-cancer quantum classifier already runs on *image-derived* features -
the Wisconsin "radius / texture / perimeter / concavity…" values are radiomics
computed from cell-nucleus images. This module extracts the 3-D analogues from a
real tumor segmentation and feeds them into the SAME quantum machinery, so the
imaging pipeline and the quantum model are literally wired together:

    segmentation ──▶ radiomics vector ──▶ AngleEmbedding ──▶ quantum circuit ──▶ output

Two honest caveats baked in:
  * A *validated* imaging-based prediction needs a quantum model trained on a
    matched, labeled radiomics dataset (e.g. a Kaggle/TCIA set), that is what
    ``QuantumRadiomicsClassifier`` is for. Plug your data into it to train.
  * Without that, we do NOT fake a diagnosis. We show (a) the real radiomics,
    (b) a transparent rule-based risk indicator, and (c) the actual quantum
    circuit those features feed, i.e. the connection, not a clinical claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qml_oncology.quantum.classifier import QuantumClassifier  # noqa: E402

# the 3-D radiomics features (one per qubit) extracted from a segmentation
RADIOMICS_KEYS = [
    "tumor_volume_mm3", "surface_area_mm2", "sphericity", "elongation",
    "flatness", "mean_intensity", "intensity_heterogeneity", "max_diameter_mm",
]


def radiomics_vector(feats: dict) -> np.ndarray:
    """Ordered feature vector from a tumor_features() dict."""
    return np.array([feats[k] for k in RADIOMICS_KEYS], dtype=float)


def heuristic_risk(feats: dict) -> dict:
    """A transparent, rule-based concern indicator from radiomics.

    NOT a diagnosis and NOT the quantum model, just an interpretable score from
    well-known radiomics cues, so there is an honest readout to show.
    Irregular shape (low sphericity), heterogeneous texture, and large size all
    raise the indicator.
    """
    # each component mapped to 0..1 with simple, stated thresholds
    irregularity = float(np.clip(1.0 - feats["sphericity"], 0, 1))
    heterogeneity = float(np.clip(feats["intensity_heterogeneity"] / 300.0, 0, 1))
    size = float(np.clip(feats["max_diameter_mm"] / 100.0, 0, 1))
    components = {"shape irregularity": irregularity,
                  "texture heterogeneity": heterogeneity,
                  "size": size}
    score = float(np.clip(0.5 * irregularity + 0.3 * heterogeneity + 0.2 * size, 0, 1))
    return {"score": score, "components": components}


def encode_circuit_drawing(vector: np.ndarray, n_layers: int = 2, seed: int = 0) -> str:
    """Draw the quantum circuit that the imaging radiomics feed into.

    This demonstrates the literal data flow (imaging features -> angle encoding ->
    entangling layers -> measurement). Weights are random here, so the numeric
    output is NOT meaningful, only the connection/structure is.
    """
    n = len(vector)
    dev = qml.device("default.qubit", wires=n)
    # normalize features to sensible rotation angles (z-score then squash)
    v = (vector - vector.mean()) / (vector.std() + 1e-9)
    v = np.tanh(v) * (np.pi / 2)

    @qml.qnode(dev)
    def circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(n))
        qml.StronglyEntanglingLayers(weights, wires=range(n))
        return qml.expval(qml.PauliZ(0))

    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n)
    rng = np.random.default_rng(seed)
    w = pnp.array(rng.normal(0, 0.1, shape), requires_grad=False)
    return qml.draw(circuit, decimals=None, show_matrices=False, max_length=110)(w, v)


class QuantumRadiomicsClassifier:
    """Trainable imaging-radiomics quantum classifier.

    Point ``fit`` at a labeled radiomics table (e.g. exported from a Kaggle
    dataset or produced by segmenting many cases) and it trains the same
    variational quantum classifier used elsewhere in the project. This is the
    real path to an imaging-based quantum diagnostic, it just needs data.
    """

    def __init__(self, n_layers: int = 4, epochs: int = 40, seed: int = 42):
        self.n_layers = n_layers
        self.epochs = epochs
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.model_ = QuantumClassifier(
            n_qubits=X.shape[1], n_layers=self.n_layers, epochs=self.epochs,
            seed=self.seed, verbose=True,
        )
        self.model_.fit(Xs, y)
        return self

    def analyze(self, vector: np.ndarray) -> float:
        """Malignancy probability for a single radiomics vector (after fit)."""
        Xs = self.scaler_.transform(vector.reshape(1, -1))
        return float(self.model_.predict_proba(Xs)[0, 1])
