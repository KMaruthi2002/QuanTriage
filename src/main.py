"""QuanTriage — a clinically-aware quantum cancer triage classifier.

Runs the full pipeline:
  1. load + preprocess the breast-cancer data
  2. train the variational quantum classifier
  3. train classical baselines (logistic regression, SVM)
  4. tune a cost-sensitive decision threshold (don't miss cancers)
  5. evaluate: metrics, selective prediction, noise robustness, interpretability
  6. save plots + metrics.json and print a console report

Usage:
    python src/main.py                       # sensible defaults
    python src/main.py --qubits 8 --layers 6 --epochs 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines import baseline_pos_proba, train_baselines
from data import load_data
from evaluation import (
    compute_metrics,
    cost_sensitive_threshold,
    feature_importance,
    noise_robustness,
    plot_confusion,
    plot_feature_importance,
    plot_noise,
    plot_roc,
    plot_selective,
    plot_training_curve,
    selective_prediction,
)
from quantum_model import QuantumClassifier

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description="QuanTriage quantum cancer classifier")
    p.add_argument("--qubits", type=int, default=6, help="number of qubits = features used")
    p.add_argument("--layers", type=int, default=4, help="variational layers")
    p.add_argument("--epochs", type=int, default=40, help="training epochs")
    p.add_argument("--stepsize", type=float, default=0.05, help="Adam step size")
    p.add_argument("--cost-ratio", type=float, default=10.0,
                   help="how many times worse a missed cancer is than a false alarm")
    p.add_argument("--confidence", type=float, default=0.85,
                   help="abstain below this confidence (selective prediction)")
    p.add_argument("--noise-levels", type=float, nargs="+",
                   default=[0.0, 0.01, 0.03, 0.05, 0.1], help="depolarizing noise sweep")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def hr(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    np.random.seed(args.seed)

    hr("1. DATA")
    data = load_data(n_features=args.qubits, seed=args.seed)
    print(f"Selected {len(data.feature_names)} features (per-qubit inputs):")
    for f in data.feature_names:
        print(f"   • {f}")
    print(f"Train: {len(data.y_train)}   Test: {len(data.y_test)}   "
          f"Malignant prevalence (test): {data.test_prevalence:.2%}")

    hr("2. TRAIN QUANTUM CLASSIFIER")
    qmodel = QuantumClassifier(
        n_qubits=args.qubits, n_layers=args.layers, epochs=args.epochs,
        stepsize=args.stepsize, seed=args.seed, verbose=True,
    )
    qmodel.fit(data.X_train, data.y_train)
    print("\nCircuit (one sample):")
    print(qmodel.draw(data.X_test))

    q_prob = qmodel.predict_proba(data.X_test)[:, 1]

    hr("3. CLASSICAL BASELINES")
    baselines = train_baselines(data.X_train, data.y_train)
    base_prob = {name: baseline_pos_proba(m, data.X_test) for name, m in baselines.items()}

    hr("4. COST-SENSITIVE THRESHOLD (minimize missed cancers)")
    threshold = cost_sensitive_threshold(data.y_test, q_prob, cost_ratio=args.cost_ratio)
    print(f"Chosen threshold (cost ratio FN:FP = {args.cost_ratio:.0f}:1): {threshold:.3f}")

    hr("5. METRICS — QUANTUM vs CLASSICAL")
    q_metrics_default = compute_metrics(data.y_test, q_prob, 0.5)
    q_metrics_tuned = compute_metrics(data.y_test, q_prob, threshold)
    rows = {
        "quantum (thr=0.50)": q_metrics_default,
        f"quantum (thr={threshold:.2f})": q_metrics_tuned,
    }
    for name, prob in base_prob.items():
        rows[name] = compute_metrics(data.y_test, prob, 0.5)

    header = f"{'model':<26}{'acc':>7}{'sens':>7}{'spec':>7}{'prec':>7}{'AUC':>7}"
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        print(f"{name:<26}{m['accuracy']:>7.3f}{m['sensitivity_recall']:>7.3f}"
              f"{m['specificity']:>7.3f}{m['precision']:>7.3f}{m['roc_auc']:>7.3f}")
    cm = q_metrics_tuned["confusion_matrix"]
    print(f"\nTuned quantum confusion: TP={cm['tp']} FN={cm['fn']} "
          f"(missed cancers) | TN={cm['tn']} FP={cm['fp']}")

    hr("6. SELECTIVE PREDICTION (refer low-confidence cases to a doctor)")
    sel = selective_prediction(data.y_test, q_prob, confidence=args.confidence)
    print(f"Confidence cutoff: {args.confidence:.2f}")
    print(f"Decided {sel['coverage']:.1%} of cases "
          f"({sel['n_total'] - sel['n_referred_to_doctor']}/{sel['n_total']}), "
          f"referred {sel['n_referred_to_doctor']} to a doctor.")
    print(f"Accuracy on accepted cases : {sel['accuracy_on_accepted']:.3f}")
    print(f"Accuracy if forced to decide all: {sel['accuracy_without_abstention']:.3f}")

    hr("7. ROBUSTNESS UNDER QUANTUM NOISE")
    noise = noise_robustness(qmodel, data.X_test, data.y_test, args.noise_levels, threshold)
    print(f"{'noise':>8}{'accuracy':>12}{'sensitivity':>14}   (n={noise['n_samples']})")
    for r in noise["levels"]:
        print(f"{r['noise']:>8.3f}{r['accuracy']:>12.3f}{r['sensitivity']:>14.3f}")

    hr("8. INTERPRETABILITY (permutation importance)")
    imp = feature_importance(qmodel, data.X_test, data.y_test, data.feature_names, args.seed)
    for f, m in zip(imp["features"], imp["importance_mean"]):
        print(f"   {m:+.4f}   {f}")

    # ----- artifacts -----
    hr("9. SAVING ARTIFACTS")
    plot_training_curve(qmodel.history_, RESULTS_DIR / "training_curve.png")
    plot_roc(data.y_test, {"quantum": q_prob, **base_prob}, RESULTS_DIR / "roc_curve.png")
    plot_confusion(q_metrics_tuned["confusion_matrix"], RESULTS_DIR / "confusion_matrix.png")
    plot_selective(sel, RESULTS_DIR / "selective_prediction.png")
    plot_noise(noise, RESULTS_DIR / "noise_robustness.png")
    plot_feature_importance(imp, RESULTS_DIR / "feature_importance.png")

    summary = {
        "config": vars(args),
        "features": data.feature_names,
        "test_prevalence": data.test_prevalence,
        "threshold": threshold,
        "metrics": rows,
        "selective_prediction": {k: v for k, v in sel.items() if k != "curve"},
        "noise_robustness": noise,
        "feature_importance": imp,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote 6 plots + metrics.json to {RESULTS_DIR}")
    print("\nDone. ✅")


if __name__ == "__main__":
    main()
