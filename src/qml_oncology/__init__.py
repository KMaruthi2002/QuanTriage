"""qml-oncology: quantum machine learning + medical imaging for oncology.

This top-level import exposes the **core** formulations, which need only the base
dependencies (PennyLane + scikit-learn):

* Quantum classifiers, ``QuantumClassifier`` (binary), ``MultiClassQuantumClassifier``
  (tumor type), and ``FlexibleQuantumClassifier`` (swappable angle / amplitude /
  data-reuploading encodings and simulator / real-hardware backends via ``make_device``).
* The clinically-honest evaluation toolkit, ``compute_metrics``,
  ``cost_sensitive_threshold``, ``selective_prediction``, ``noise_robustness``,
  ``feature_importance``.
* A registry of real cancer datasets, ``load_dataset`` / ``DATASETS``.

Imaging, segmentation, the PyTorch hybrid, and the real-hardware plugins live in
submodules that need the optional extras::

    pip install qml-oncology[imaging]    # U-Nets, radiomics, 3-D rendering (torch, nibabel, ...)
    pip install qml-oncology[hardware]   # IBM Quantum via pennylane-qiskit
    pip install qml-oncology[all]        # everything

    from qml_oncology.quantum.hybrid import HybridQuantumClassifier   # needs [imaging] (torch)
    from qml_oncology.segmentation.multimodal import train_multimodal # needs [imaging]
    from qml_oncology.imaging.tumor3d import build_figure             # needs [imaging]

**Educational / research software. Not a validated medical device.**
"""

from qml_oncology.data.datasets import DATASETS, load_dataset
from qml_oncology.evaluation import (
    compute_metrics,
    cost_sensitive_threshold,
    feature_importance,
    noise_robustness,
    selective_prediction,
)
from qml_oncology.quantum.classifier import QuantumClassifier
from qml_oncology.quantum.flexible import FlexibleQuantumClassifier, make_device
from qml_oncology.quantum.multiclass import MultiClassQuantumClassifier

__version__ = "0.1.0"

__all__ = [
    "QuantumClassifier",
    "MultiClassQuantumClassifier",
    "FlexibleQuantumClassifier",
    "make_device",
    "compute_metrics",
    "cost_sensitive_threshold",
    "selective_prediction",
    "noise_robustness",
    "feature_importance",
    "load_dataset",
    "DATASETS",
    "__version__",
]
