"""Train and evaluate the quantum classifier across MULTIPLE cancer types.

By default this runs on the TCGA pan-cancer RNA-seq dataset and classifies a
gene-expression profile into one of five tumor types (breast, kidney, lung,
prostate, colon), with a classical baseline for honest comparison.

    python src/multicancer.py
    python src/multicancer.py --dataset breast_cancer --features 6
    python src/multicancer.py --features 10 --qubits 10 --epochs 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from qml_oncology.data.datasets import TCGA_FULL_NAMES, load_dataset
from qml_oncology.quantum.multiclass import MultiClassQuantumClassifier

RESULTS = Path.cwd() / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Multi-cancer quantum classifier")
    p.add_argument("--dataset", default="pan_cancer_rnaseq",
                   choices=["pan_cancer_rnaseq", "breast_cancer"])
    p.add_argument("--features", type=int, default=10, help="genes/features = qubits")
    p.add_argument("--qubits", type=int, default=10)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--weight-power", type=float, default=0.5,
                   help="class-imbalance correction strength (0=off, 0.5=soft, 1=full)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def plot_confusion(cm, labels, path, title):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("predicted tumor type")
    ax.set_ylabel("actual tumor type")
    ax.set_title(title)
    thr = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    RESULTS.mkdir(exist_ok=True)

    print("=" * 68)
    print("MULTI-CANCER QUANTUM CLASSIFIER")
    print("=" * 68)
    print(f"Loading dataset: {args.dataset} (downloads on first use)…")
    ds = load_dataset(args.dataset, n_features=args.features, seed=args.seed)
    pretty = [f"{c} ({TCGA_FULL_NAMES.get(c, c)})" for c in ds.class_names]
    print(f"\n{ds.name}")
    print(f"  task     : {ds.task}  ({ds.n_classes} classes)")
    print(f"  classes  : {', '.join(pretty)}")
    print(f"  train/test: {len(ds.y_train)}/{len(ds.y_test)}   features: {len(ds.feature_names)}")

    print("\nTraining quantum classifier…")
    qmodel = MultiClassQuantumClassifier(
        n_classes=ds.n_classes, n_qubits=max(args.qubits, ds.n_classes),
        n_layers=args.layers, epochs=args.epochs, seed=args.seed, verbose=True,
        weight_power=args.weight_power,
    )
    qmodel.fit(ds.X_train, ds.y_train)
    q_pred = qmodel.predict(ds.X_test)
    q_acc = accuracy_score(ds.y_test, q_pred)

    print("\nTraining classical baseline (logistic regression)…")
    clf = LogisticRegression(max_iter=3000).fit(ds.X_train, ds.y_train)
    c_acc = accuracy_score(ds.y_test, clf.predict(ds.X_test))

    print("\n" + "=" * 68)
    print("RESULTS")
    print("=" * 68)
    print(f"Quantum test accuracy : {q_acc:.3f}")
    print(f"Classical baseline    : {c_acc:.3f}")
    print("\nPer-tumor-type report (quantum):")
    print(classification_report(ds.y_test, q_pred, target_names=ds.class_names, zero_division=0))

    cm = confusion_matrix(ds.y_test, q_pred, labels=range(ds.n_classes))
    out = RESULTS / f"multicancer_confusion_{args.dataset}.png"
    plot_confusion(cm, ds.class_names, out,
                   f"Quantum multi-cancer classifier, acc {q_acc:.2f}")

    summary = {
        "dataset": ds.key, "name": ds.name, "classes": ds.class_names,
        "features": ds.feature_names, "quantum_accuracy": float(q_acc),
        "classical_accuracy": float(c_acc),
        "confusion_matrix": cm.tolist(),
    }
    with open(RESULTS / f"multicancer_{args.dataset}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved confusion matrix -> {out}")
    print("Done. ✅")


if __name__ == "__main__":
    main()
