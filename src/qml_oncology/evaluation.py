"""Evaluation: metrics, cost-sensitive thresholding, selective prediction,
noise robustness, interpretability, and all the plots.

This module is where QuanTriage earns its "clinically honest" framing. Instead
of reporting a single accuracy number, it answers the questions a clinician
would actually ask:
  * How many real cancers does it miss?           (sensitivity / cost tuning)
  * Can it tell me when it isn't sure?            (selective prediction)
  * Does it still work on noisy real hardware?    (noise robustness)
  * Which measurements is it relying on?          (interpretability)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
)

from qml_oncology.quantum.classifier import noisy_probabilities


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Threshold probabilities and compute the clinically relevant metrics.

    Positive class = malignant, so:
      * sensitivity (recall) = fraction of real cancers caught
      * specificity          = fraction of healthy patients correctly cleared
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity)
        else 0.0
    )
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def cost_sensitive_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, cost_ratio: float = 10.0
) -> float:
    """Pick the threshold minimizing cost = cost_ratio * FN + 1 * FP.

    A missed cancer (false negative) is `cost_ratio` times worse than a false
    alarm (false positive). This pushes the operating point toward high
    sensitivity, which is what a screening tool should do.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = cost_ratio * fn + fp
        if cost < best_cost:
            best_cost, best_t = cost, t
    return float(best_t)


# --------------------------------------------------------------------------- #
# Unique angle #1: selective prediction / abstention
# --------------------------------------------------------------------------- #
def selective_prediction(
    y_true: np.ndarray, y_prob: np.ndarray, confidence: float = 0.85
) -> dict:
    """A model that refers low-confidence cases to a human.

    Confidence = max(p, 1 - p). Cases below `confidence` are *abstained* on
    (referred to a doctor); we report accuracy on the cases the model did keep.
    Also returns the full risk-coverage curve for plotting.
    """
    conf = np.maximum(y_prob, 1 - y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    accept = conf >= confidence
    n = len(y_true)
    n_accept = int(accept.sum())
    coverage = n_accept / n if n else 0.0
    acc_accept = float(np.mean(y_pred[accept] == y_true[accept])) if n_accept else 0.0
    acc_all = float(np.mean(y_pred == y_true))

    # risk-coverage curve: sweep the confidence cutoff
    cutoffs = np.linspace(0.5, 1.0, 51)
    cov_curve, acc_curve = [], []
    for c in cutoffs:
        m = conf >= c
        if m.sum() == 0:
            cov_curve.append(0.0)
            acc_curve.append(np.nan)
        else:
            cov_curve.append(float(m.mean()))
            acc_curve.append(float(np.mean(y_pred[m] == y_true[m])))

    return {
        "confidence_threshold": confidence,
        "coverage": coverage,
        "n_referred_to_doctor": n - n_accept,
        "n_total": n,
        "accuracy_on_accepted": acc_accept,
        "accuracy_without_abstention": acc_all,
        "curve": {"cutoffs": cutoffs.tolist(), "coverage": cov_curve, "accuracy": acc_curve},
    }


# --------------------------------------------------------------------------- #
# Unique angle #3: robustness under quantum noise
# --------------------------------------------------------------------------- #
def noise_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: list[float],
    threshold: float,
    max_samples: int = 120,
    seed: int = 42,
) -> dict:
    """Re-evaluate the trained model on a noisy (mixed-state) simulator.

    Subsamples the test set because density-matrix simulation is heavier.
    Returns accuracy and sensitivity at each depolarizing-noise level.
    """
    rng = np.random.default_rng(seed)
    if len(X_test) > max_samples:
        idx = rng.choice(len(X_test), size=max_samples, replace=False)
        Xs, ys = X_test[idx], y_test[idx]
    else:
        Xs, ys = X_test, y_test

    rows = []
    for p in noise_levels:
        if p == 0.0:
            # noiseless reference straight from the trained estimator
            probs = model.predict_proba(Xs)[:, 1]
        else:
            probs = noisy_probabilities(model, Xs, p)
        m = compute_metrics(ys, probs, threshold)
        rows.append(
            {
                "noise": float(p),
                "accuracy": m["accuracy"],
                "sensitivity": m["sensitivity_recall"],
            }
        )
    return {"n_samples": int(len(Xs)), "levels": rows}


# --------------------------------------------------------------------------- #
# Unique angle #4: interpretability
# --------------------------------------------------------------------------- #
def feature_importance(
    model, X_test: np.ndarray, y_test: np.ndarray, feature_names: list[str], seed: int = 42
) -> dict:
    """Permutation importance of each named feature on the quantum model.

    Shuffle one feature column at a time and measure how much test accuracy
    drops -- the bigger the drop, the more the model relies on that measurement.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=8, random_state=seed, scoring="accuracy"
    )
    order = np.argsort(result.importances_mean)[::-1]
    return {
        "features": [feature_names[i] for i in order],
        "importance_mean": [float(result.importances_mean[i]) for i in order],
        "importance_std": [float(result.importances_std[i]) for i in order],
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_training_curve(history: dict, path: Path):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history["loss"]) + 1)
    ax1.plot(epochs, history["loss"], color="#c0392b", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["acc"], color="#2980b9", label="train accuracy")
    ax2.set_ylabel("train accuracy", color="#2980b9")
    ax1.set_title("Quantum classifier training")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(y_true, curves: dict, path: Path):
    """curves: name -> y_prob array."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC, quantum vs. classical")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: dict, path: Path):
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(mat, cmap="Blues")
    labels = ["benign", "malignant"]
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Confusion matrix (quantum)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_selective(sel: dict, path: Path):
    cov = np.array(sel["curve"]["coverage"])
    acc = np.array(sel["curve"]["accuracy"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cov, acc, marker="o", ms=3, color="#27ae60")
    ax.set_xlabel("coverage (fraction the model decides)")
    ax.set_ylabel("accuracy on accepted cases")
    ax.set_title("Selective prediction, accuracy rises as it abstains more")
    ax.invert_xaxis()  # left = decide everything, right = abstain more
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_noise(noise: dict, path: Path):
    levels = [r["noise"] for r in noise["levels"]]
    acc = [r["accuracy"] for r in noise["levels"]]
    sens = [r["sensitivity"] for r in noise["levels"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(levels, acc, marker="o", label="accuracy")
    ax.plot(levels, sens, marker="s", label="sensitivity")
    ax.set_xlabel("depolarizing noise strength")
    ax.set_ylabel("score")
    ax.set_title(f"Robustness under quantum noise (n={noise['n_samples']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(imp: dict, path: Path):
    feats = imp["features"][::-1]
    means = imp["importance_mean"][::-1]
    stds = imp["importance_std"][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(feats, means, xerr=stds, color="#8e44ad")
    ax.set_xlabel("drop in accuracy when shuffled (importance)")
    ax.set_title("Which measurements the quantum model relies on")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
