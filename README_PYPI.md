# qml-oncology

**Quantum machine learning + medical imaging for oncology.** Reusable, clinically-honest
formulations: variational quantum classifiers (swappable encodings and simulator / real-hardware
backends), a multi-cancer-type classifier, a selective-prediction / cost-sensitive evaluation
toolkit, a radiomics-to-quantum bridge, and U-Net tumor localization with 3-D rendering.

Extracted from the [QuanTriage](https://github.com/KMaruthi2002/QuanTriage) project so others can
build on the same methods.

> Educational / research software. **Not a validated medical device.**

## Install

```bash
pip install qml-oncology                 # core: quantum classifiers + eval toolkit + datasets
pip install qml-oncology[imaging]        # + U-Nets, radiomics, 3-D rendering (torch, nibabel, ...)
pip install qml-oncology[hardware]       # + IBM Quantum via pennylane-qiskit
pip install qml-oncology[all]            # everything
```

## Quick start

### A clinically-honest quantum classifier

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from qml_oncology import (FlexibleQuantumClassifier, compute_metrics,
                          cost_sensitive_threshold, selective_prediction)

X, y = load_breast_cancer(return_X_y=True)
y = (y == 0).astype(int)                       # positive class = malignant
Xtr, Xte, ytr, yte = train_test_split(X[:, :6], y, stratify=y, random_state=0)

clf = FlexibleQuantumClassifier(encoding="reupload", n_qubits=6, n_layers=4, epochs=25)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:, 1]

# tune a threshold that treats a missed cancer as 10x worse than a false alarm
thr = cost_sensitive_threshold(yte, proba, cost_ratio=10)
print(compute_metrics(yte, proba, thr))        # accuracy, sensitivity, specificity, AUC, ...

# defer low-confidence cases to a human instead of guessing
print(selective_prediction(yte, proba, confidence=0.85))
```

### Encodings, bigger models, real hardware

```python
# data re-uploading or amplitude encoding, on the fast C++ simulator
m = FlexibleQuantumClassifier(encoding="amplitude", n_qubits=4, device="lightning.qubit")

# train on a simulator, then run the SAME trained circuit on real hardware
from qml_oncology import make_device   # device factory
m.predict_on_device(Xte, "qiskit.remote", shots=1024)   # IBM Quantum (needs your token)
m.predict_on_device(Xte, "braket.aws.qubit", shots=1024)  # AWS Braket (needs AWS creds)
```

### Multi-cancer typing (5 tumor types from gene expression)

```python
from qml_oncology import load_dataset, MultiClassQuantumClassifier
ds = load_dataset("pan_cancer_rnaseq", n_features=10)    # downloads TCGA data on first use
clf = MultiClassQuantumClassifier(n_classes=ds.n_classes, n_qubits=10, epochs=35).fit(ds.X_train, ds.y_train)
```

### Imaging (needs `[imaging]`): tumor localization + 3-D render

```python
from qml_oncology.segmentation.multimodal import build_multimodal_dataset, train_multimodal
from qml_oncology.imaging.tumor3d import build_figure          # interactive 3-D tumor render
from qml_oncology.imaging.bridge import radiomics_vector       # radiomics -> quantum circuit
```

## What's inside

| Module | What |
|---|---|
| `qml_oncology.quantum` | variational, multi-class, flexible (encodings + backends), and PyTorch hybrid classifiers |
| `qml_oncology.evaluation` | cost-sensitive thresholding, selective prediction, noise robustness, permutation importance |
| `qml_oncology.data` | dataset registry (breast cancer, TCGA pan-cancer RNA-seq) |
| `qml_oncology.imaging` | radiomics, 3-D rendering, radiomics-to-quantum bridge |
| `qml_oncology.segmentation` | 2-D / 3-D U-Nets, multi-modal MRI loaders, training |

Full app, visuals, and docs: <https://github.com/KMaruthi2002/QuanTriage>

## License

MIT
