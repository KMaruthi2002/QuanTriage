"""QuanTriage — interactive Streamlit dashboard for the quantum cancer classifier.

Run with:
    .venv/bin/streamlit run app.py

The heavy work (training the quantum circuit, computing noise robustness and
feature importance) is cached, so it runs once and the UI stays responsive.
The clinical controls in the sidebar (cost ratio, confidence cutoff) only touch
cheap recomputations, so they update live.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "imaging"))

import numpy as np
import streamlit as st

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

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

st.set_page_config(page_title="QuanTriage", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- #
# Cached heavy computation
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training the quantum classifier (one-time, ~30s)…")
def load_bundle(qubits: int, layers: int, epochs: int, seed: int):
    data = load_data(n_features=qubits, seed=seed)
    qmodel = QuantumClassifier(
        n_qubits=qubits, n_layers=layers, epochs=epochs, seed=seed, verbose=False
    )
    qmodel.fit(data.X_train, data.y_train)
    baselines = train_baselines(data.X_train, data.y_train)
    q_prob = qmodel.predict_proba(data.X_test)[:, 1]
    base_prob = {n: baseline_pos_proba(m, data.X_test) for n, m in baselines.items()}
    return data, qmodel, q_prob, base_prob


@st.cache_resource(show_spinner="Reconstructing the real tumor in 3-D…")
def tumor_scan():
    """Build the real-imaging 3-D tumor figure + radiomics (cached)."""
    from tumor3d import build_figure

    return build_figure(animate=True, show_brain=True)


@st.cache_resource(show_spinner="Computing robustness & interpretability (one-time)…")
def static_artifacts(qubits: int, layers: int, epochs: int, seed: int):
    data, qmodel, q_prob, base_prob = load_bundle(qubits, layers, epochs, seed)
    thr = cost_sensitive_threshold(data.y_test, q_prob, 10.0)
    plot_training_curve(qmodel.history_, RESULTS / "training_curve.png")
    plot_roc(data.y_test, {"quantum": q_prob, **base_prob}, RESULTS / "roc_curve.png")
    noise = noise_robustness(
        qmodel, data.X_test, data.y_test, [0.0, 0.01, 0.03, 0.05, 0.1], thr
    )
    plot_noise(noise, RESULTS / "noise_robustness.png")
    imp = feature_importance(qmodel, data.X_test, data.y_test, data.feature_names, seed)
    plot_feature_importance(imp, RESULTS / "feature_importance.png")
    return noise, imp, thr


# --------------------------------------------------------------------------- #
# Sidebar — controls
# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 QuanTriage")
st.sidebar.caption("A clinically-aware **quantum** cancer triage classifier (PennyLane).")

st.sidebar.subheader("Clinical settings")
cost_ratio = st.sidebar.slider(
    "Cost of a missed cancer vs. a false alarm", 1.0, 20.0, 10.0, 1.0,
    help="Higher = the model works harder not to miss any malignancy (raises sensitivity).",
)
confidence_cut = st.sidebar.slider(
    "Refer-to-doctor confidence cutoff", 0.50, 0.99, 0.85, 0.01,
    help="Cases below this confidence are abstained on and referred to a human.",
)

with st.sidebar.expander("Model (changing retrains, ~30s)"):
    qubits = st.slider("Qubits / features", 4, 8, 6)
    layers = st.slider("Variational layers", 2, 6, 4)
    epochs = st.slider("Training epochs", 10, 80, 40, 5)
    seed = st.number_input("Random seed", value=42, step=1)

st.sidebar.info("Educational research demo — **not a medical device.**")

# --------------------------------------------------------------------------- #
# Load everything (cached)
# --------------------------------------------------------------------------- #
data, qmodel, q_prob, base_prob = load_bundle(qubits, layers, epochs, int(seed))
noise, imp, _default_thr = static_artifacts(qubits, layers, epochs, int(seed))

# threshold depends on the live cost-ratio slider (cheap to recompute)
threshold = cost_sensitive_threshold(data.y_test, q_prob, cost_ratio)

st.title("QuanTriage")
st.markdown(
    "A hybrid **quantum machine learning** model that flags breast tumors as malignant or "
    "benign — and, unlike most QML demos, **refers uncertain cases to a doctor**, is **tuned "
    "not to miss cancer**, **survives quantum noise**, and **explains its reasoning**."
)

tab_predict, tab_perf, tab_trust, tab_scan = st.tabs(
    ["🩺 Predict a patient", "📊 Model performance", "🛡️ Trust & robustness",
     "🧠 3-D tumor scan"]
)

# --------------------------------------------------------------------------- #
# Tab 1 — single-patient prediction
# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Enter a patient's measurements")

    # preset buttons set the sliders via session_state
    c1, c2, c3 = st.columns(3)
    if c1.button("Load example malignant"):
        for s in data.feature_stats:
            st.session_state[f"f_{s['name']}"] = round(s["mean_malignant"], 3)
    if c2.button("Load example benign"):
        for s in data.feature_stats:
            st.session_state[f"f_{s['name']}"] = round(s["mean_benign"], 3)
    if c3.button("Reset to typical"):
        for s in data.feature_stats:
            st.session_state[f"f_{s['name']}"] = round(s["median"], 3)

    left, right = st.columns([3, 2])
    raw_values = []
    with left:
        for s in data.feature_stats:
            key = f"f_{s['name']}"
            # pad the slider range a little beyond observed train min/max
            span = s["max"] - s["min"]
            lo, hi = s["min"] - 0.1 * span, s["max"] + 0.1 * span
            val = st.slider(
                s["name"], float(lo), float(hi),
                float(st.session_state.get(key, round(s["median"], 3))),
                key=key,
            )
            raw_values.append(val)

    # predict
    x_scaled = data.transform_raw(raw_values)
    p_mal = float(qmodel.predict_proba(x_scaled)[0, 1])
    confidence = max(p_mal, 1 - p_mal)

    with right:
        st.markdown("### Result")
        st.metric("Probability of malignancy", f"{p_mal:.1%}")
        st.progress(p_mal)
        st.metric("Model confidence", f"{confidence:.1%}")

        if confidence < confidence_cut:
            st.warning("⚠️ **Refer to a doctor** — the model is not confident enough to decide.")
        elif p_mal >= threshold:
            st.error("🔴 **Flag as malignant** — recommend clinical follow-up.")
        else:
            st.success("🟢 **Likely benign.**")
        st.caption(
            f"Decision threshold {threshold:.2f} (cost ratio {cost_ratio:.0f}:1) · "
            f"abstain below {confidence_cut:.0%} confidence."
        )

# --------------------------------------------------------------------------- #
# Tab 2 — performance vs. classical
# --------------------------------------------------------------------------- #
with tab_perf:
    st.subheader("Quantum vs. classical baselines (held-out test set)")

    rows = {
        "Quantum (thr 0.50)": compute_metrics(data.y_test, q_prob, 0.5),
        f"Quantum (cost-tuned {threshold:.2f})": compute_metrics(data.y_test, q_prob, threshold),
    }
    for name, prob in base_prob.items():
        rows[name.replace("_", " ").title()] = compute_metrics(data.y_test, prob, 0.5)

    table = {
        "model": list(rows.keys()),
        "accuracy": [f"{m['accuracy']:.3f}" for m in rows.values()],
        "sensitivity": [f"{m['sensitivity_recall']:.3f}" for m in rows.values()],
        "specificity": [f"{m['specificity']:.3f}" for m in rows.values()],
        "AUC": [f"{m['roc_auc']:.3f}" for m in rows.values()],
    }
    st.dataframe(table, width="stretch", hide_index=True)

    tuned = compute_metrics(data.y_test, q_prob, threshold)
    cm = tuned["confusion_matrix"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Sensitivity (cancers caught)", f"{tuned['sensitivity_recall']:.1%}")
    m2.metric("Missed cancers (FN)", cm["fn"])
    m3.metric("False alarms (FP)", cm["fp"])

    g1, g2 = st.columns(2)
    plot_confusion(cm, RESULTS / "confusion_live.png")
    g1.image(str(RESULTS / "confusion_live.png"), caption="Quantum confusion @ cost-tuned threshold")
    g2.image(str(RESULTS / "roc_curve.png"), caption="ROC — quantum vs. classical")
    st.image(str(RESULTS / "training_curve.png"), caption="Quantum training curve")

# --------------------------------------------------------------------------- #
# Tab 3 — trust & robustness
# --------------------------------------------------------------------------- #
with tab_trust:
    st.subheader("Selective prediction — knowing when to defer")
    sel = selective_prediction(data.y_test, q_prob, confidence=confidence_cut)
    s1, s2, s3 = st.columns(3)
    s1.metric("Cases decided (coverage)", f"{sel['coverage']:.0%}")
    s2.metric("Accuracy on accepted", f"{sel['accuracy_on_accepted']:.1%}")
    s3.metric("Referred to a doctor", sel["n_referred_to_doctor"])
    st.caption(
        f"If forced to decide every case, accuracy would be "
        f"{sel['accuracy_without_abstention']:.1%}. Abstaining trades coverage for safety."
    )
    plot_selective(sel, RESULTS / "selective_live.png")
    st.image(str(RESULTS / "selective_live.png"))

    st.divider()
    c1, c2 = st.columns(2)
    c1.subheader("Robustness under quantum noise")
    c1.image(str(RESULTS / "noise_robustness.png"))
    c1.caption("How the trained model holds up as simulated hardware decoherence rises.")
    c2.subheader("What the model relies on")
    c2.image(str(RESULTS / "feature_importance.png"))
    c2.caption("Permutation importance over real cell-nucleus measurements.")

# --------------------------------------------------------------------------- #
# Tab 4 — real 3-D tumor scan
# --------------------------------------------------------------------------- #
with tab_scan:
    st.subheader("Real tumor, reconstructed in 3-D")
    st.markdown(
        "An **actual brain-MRI volume with a radiologist-drawn tumor segmentation** "
        "(the open `brain1` sample), reconstructed as a true 3-D surface via marching cubes and "
        "colored by MRI intensity. Drag to rotate, or hit **▶ Rotate** to orbit it."
    )
    fig, feats = tumor_scan()
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Tumor state — quantified from the real segmentation (radiomics)")
    a, b, c, d = st.columns(4)
    a.metric("Volume", f"{feats['tumor_volume_mm3']:.0f} mm³")
    b.metric("Max diameter", f"{feats['max_diameter_mm']:.0f} mm")
    c.metric("Sphericity", f"{feats['sphericity']:.2f}", help="1.0 = perfect sphere; lower = more irregular")
    d.metric("Intensity heterogeneity", f"{feats['intensity_heterogeneity']:.0f}")
    e, f_, g, h = st.columns(4)
    e.metric("Surface area", f"{feats['surface_area_mm2']:.0f} mm²")
    f_.metric("Elongation", f"{feats['elongation']:.2f}")
    g.metric("Flatness", f"{feats['flatness']:.2f}")
    h.metric("Tumor voxels", f"{feats['voxel_count']:,}")

    st.info(
        "**Honest scope note.** This is a *real* segmentation rendered in 3-D and quantified with "
        "real radiomics — not a stylized shape. It is **not** a quantum prediction on this scan: "
        "the quantum model on the other tabs is a tabular diagnostic. Linking the quantum model to "
        "imaging (classifying directly from radiomics/voxels) is the project's research roadmap."
    )
