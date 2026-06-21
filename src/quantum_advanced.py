"""Showcase of the advanced quantum options, every 'Extending it' knob, live.

Runs on the breast-cancer data and demonstrates:
  1. Encodings / ansätze: angle vs. data-reuploading vs. amplitude embedding.
  2. Bigger model on the fast C++ simulator (lightning.qubit).
  3. PyTorch hybrid (quantum TorchLayer + classical layers).
  4. Real-hardware path: run a trained circuit on the qiskit Aer sampling
     backend, and the exact code to target IBM Quantum / AWS Braket.

    python src/quantum_advanced.py
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hybrid_torch import HybridQuantumClassifier
from quantum_flexible import FlexibleQuantumClassifier


def prep(n_features, seed=42):
    raw = load_breast_cancer()
    y = (raw.target == 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(raw.data, y, test_size=0.25,
                                          random_state=seed, stratify=y)
    sel = SelectKBest(f_classif, k=n_features).fit(Xtr, ytr)
    sc = StandardScaler().fit(sel.transform(Xtr))
    return (sc.transform(sel.transform(Xtr)), sc.transform(sel.transform(Xte)), ytr, yte)


def hr(t):
    print("\n" + "=" * 66 + f"\n{t}\n" + "=" * 66)


def main():
    results = {}

    hr("1 · ENCODINGS / ANSÄTZE  (6 qubits, 4 layers)")
    Xtr, Xte, ytr, yte = prep(6)
    for enc, kw in [("angle", {}), ("reupload", {"reuploads": 3})]:
        m = FlexibleQuantumClassifier(encoding=enc, n_qubits=6, n_layers=4,
                                      epochs=25, seed=42, verbose=False, **kw)
        m.fit(Xtr, ytr)
        acc = accuracy_score(yte, m.predict(Xte))
        results[enc] = acc
        print(f"  {enc:10s} -> test accuracy {acc:.3f}")

    # amplitude embedding packs 2^n features into n qubits (log compression)
    Xtr16, Xte16, ytr, yte = prep(16)
    ma = FlexibleQuantumClassifier(encoding="amplitude", n_qubits=4, n_layers=4,
                                   epochs=25, seed=42, verbose=False)
    ma.fit(Xtr16, ytr)
    acc = accuracy_score(yte, ma.predict(Xte16))
    results["amplitude"] = acc
    print(f"  {'amplitude':10s} -> test accuracy {acc:.3f}  (16 features in 4 qubits)")

    hr("2 · BIGGER MODEL on the fast C++ simulator (lightning.qubit)")
    Xtr8, Xte8, ytr, yte = prep(8)
    t0 = time.time()
    big = FlexibleQuantumClassifier(encoding="angle", n_qubits=8, n_layers=6,
                                    device="lightning.qubit", epochs=20,
                                    seed=42, verbose=False)
    big.fit(Xtr8, ytr)
    acc = accuracy_score(yte, big.predict(Xte8))
    results["big(8q,6L)"] = acc
    print(f"  8 qubits, 6 layers -> test accuracy {acc:.3f}  "
          f"(trained in {time.time() - t0:.1f}s on lightning.qubit)")

    hr("3 · PYTORCH HYBRID  (Linear -> quantum TorchLayer -> Linear)")
    Xtr, Xte, ytr, yte = prep(6)
    hyb = HybridQuantumClassifier(n_qubits=6, n_layers=3, epochs=30, verbose=False)
    hyb.fit(Xtr, ytr)
    acc = accuracy_score(yte, hyb.predict(Xte))
    results["hybrid"] = acc
    print(f"  hybrid -> test accuracy {acc:.3f}")

    hr("4 · REAL-HARDWARE PATH  (train on simulator, run on a backend)")
    # re-run the trained angle model on the qiskit Aer sampling backend
    Xtr, Xte, ytr, yte = prep(6)
    sim = FlexibleQuantumClassifier(encoding="angle", n_qubits=6, n_layers=4,
                                    epochs=25, seed=42, verbose=False).fit(Xtr, ytr)
    sub = slice(0, 20)  # small subset, sampling backends run one circuit per shot-set
    proba = sim.predict_on_device(Xte[sub], "qiskit.aer", shots=1024)
    acc = accuracy_score(yte[sub], proba[:, 1] >= 0.5)
    print(f"  same model on qiskit.aer (1024 shots, 20 test cases) -> accuracy {acc:.3f}")
    print("\n  To run on REAL hardware, just swap the device (needs your credentials):")
    print('    # IBM Quantum:')
    print('    model.predict_on_device(X, "qiskit.remote", shots=1024)  '
          '# backend="ibm_brisbane", token=...')
    print('    # AWS Braket:')
    print('    model.predict_on_device(X, "braket.aws.qubit", shots=1024)  # device_arn=...')

    hr("SUMMARY")
    for k, v in results.items():
        print(f"  {k:14s} {v:.3f}")
    print("\nDone. ✅  (educational/research, not a medical device)")


if __name__ == "__main__":
    main()
