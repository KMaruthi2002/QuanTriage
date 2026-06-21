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
from quantum_imaging import encode_circuit_drawing, heuristic_risk, radiomics_vector
from datasets import TCGA_FULL_NAMES, load_dataset
from multiclass_model import MultiClassQuantumClassifier

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


@st.cache_resource(show_spinner="Training multi-cancer model (downloads ~72 MB on first use)…")
def multicancer_train(features, qubits, layers, epochs, weight_power, seed):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.linear_model import LogisticRegression

    ds = load_dataset("pan_cancer_rnaseq", n_features=features, seed=seed)
    qm = MultiClassQuantumClassifier(
        n_classes=ds.n_classes, n_qubits=max(qubits, ds.n_classes), n_layers=layers,
        epochs=epochs, seed=seed, verbose=False, weight_power=weight_power,
    )
    qm.fit(ds.X_train, ds.y_train)
    q_pred = qm.predict(ds.X_test)
    q_acc = accuracy_score(ds.y_test, q_pred)
    clf = LogisticRegression(max_iter=3000).fit(ds.X_train, ds.y_train)
    c_acc = accuracy_score(ds.y_test, clf.predict(ds.X_test))
    report = classification_report(ds.y_test, q_pred, target_names=ds.class_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(ds.y_test, q_pred, labels=range(ds.n_classes))
    return dict(class_names=ds.class_names, q_acc=q_acc, c_acc=c_acc,
                report=report, cm=cm, history=qm.history_, features=ds.feature_names)


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
    "A local **quantum + imaging** platform for cancer ML: classify the **cancer type** across "
    "five tumor types, **localize** tumors in real brain MRI with a 3-D render, and train on your "
    "own data — plus a clinically-honest **breast-cancer diagnosis** demo that defers to a doctor "
    "when unsure. Everything runs on-device."
)

tab_types, tab_scan, tab_loc, tab_predict, tab_perf, tab_trust = st.tabs(
    ["🧬 Cancer types", "🧠 3-D tumor scan", "🎯 Localization (train)",
     "🩺 Breast: predict", "📊 Breast: performance", "🛡️ Breast: trust"]
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

    st.divider()
    st.markdown("#### 🔗 Quantum analysis of this tumor (imaging → quantum)")
    st.markdown(
        "The imaging radiomics above are fed straight into the quantum machinery — the same "
        "angle-encoding + entangling circuit the classifier uses elsewhere."
    )
    feats_vec = radiomics_vector(feats)
    risk = heuristic_risk(feats)
    rc1, rc2 = st.columns([2, 3])
    with rc1:
        st.metric("Radiomics risk indicator", f"{risk['score']:.0%}",
                  help="Transparent rule-based score — NOT a diagnosis and NOT the quantum model.")
        st.progress(risk["score"])
        for name, val in risk["components"].items():
            st.caption(f"{name}: {val:.0%}")
    with rc2:
        st.caption("The tumor's radiomics, angle-encoded into the quantum circuit:")
        st.code(encode_circuit_drawing(feats_vec), language="text")

    st.warning(
        "**Honest scope.** The 3-D scan and radiomics are *real*. The risk indicator is a "
        "transparent heuristic, **not** the quantum model and **not** a diagnosis. The circuit "
        "shows the real data flow (imaging → quantum), but a *valid* imaging-based quantum "
        "prediction requires training `QuantumRadiomicsClassifier` on a matched, labeled radiomics "
        "dataset (the training-studio / Kaggle path). **Not a medical device.**"
    )

    st.divider()
    st.markdown("#### 🎯 U-Net localization — predicted tumor in 3-D")
    sys.path.insert(0, str(ROOT / "segmentation"))
    from localize_msd import has_msd, list_cases, predicted_figure

    if has_msd():
        st.caption("Run the shipped multi-modal U-Net on a real MSD brain-MRI case and render the "
                   "**predicted** tumor on the brain.")
        case = st.selectbox("MSD case", list_cases(12))
        if st.button("🎯 Localize in 3-D"):
            with st.spinner("Segmenting the volume on the GPU…"):
                fig_p, meta = predicted_figure(case)
            st.plotly_chart(fig_p, use_container_width=True)
            st.caption(f"Predicted tumor volume ≈ {meta['tumor_volume_mm3']:,.0f} mm³ "
                       f"({meta['tumor_voxels']:,} voxels) · case {meta['case']}")
    else:
        st.caption("MSD data not present locally (7 GB, not shipped) — showing the shipped result. "
                   "Run `python segmentation/multimodal.py` after downloading MSD to do this live.")
        st.image(str(ROOT / "assets" / "msd_tumor_on_organ.png"),
                 caption="Predicted tumor on the brain (held-out MRI, Dice 0.85)")

# --------------------------------------------------------------------------- #
# Tab 5 — multi-cancer types
# --------------------------------------------------------------------------- #
with tab_types:
    st.subheader("Classifying multiple cancer types")
    st.markdown(
        "Beyond breast cancer: a **multi-class quantum classifier** on the real **TCGA pan-cancer "
        "RNA-seq** dataset, classifying a gene-expression profile into one of five tumor types — "
        "**breast, kidney, lung, prostate, colon**."
    )
    if st.button("🧬 Train / load multi-cancer model", help="Downloads ~72 MB on first use; ~1–2 min."):
        st.session_state["mc_go"] = True
    if st.session_state.get("mc_go"):
        res = multicancer_train(10, 10, 4, 35, 0.5, int(seed))
        m1, m2 = st.columns(2)
        m1.metric("Quantum accuracy (5-way)", f"{res['q_acc']:.1%}")
        m2.metric("Classical baseline", f"{res['c_acc']:.1%}")
        rows = {"tumor type": [], "precision": [], "recall": [], "f1": []}
        for c in res["class_names"]:
            r = res["report"][c]
            rows["tumor type"].append(f"{c} ({TCGA_FULL_NAMES.get(c, c)})")
            rows["precision"].append(f"{r['precision']:.2f}")
            rows["recall"].append(f"{r['recall']:.2f}")
            rows["f1"].append(f"{r['f1-score']:.2f}")
        st.dataframe(rows, width="stretch", hide_index=True)
        import numpy as _np
        cm = _np.array(res["cm"])
        import plotly.graph_objects as _go
        heat = _go.Figure(_go.Heatmap(
            z=cm, x=res["class_names"], y=res["class_names"], colorscale="Blues",
            text=cm, texttemplate="%{text}", showscale=True))
        heat.update_layout(title="Confusion matrix (actual ↓ vs predicted →)",
                           xaxis_title="predicted", yaxis_title="actual", height=420)
        st.plotly_chart(heat, use_container_width=True)
        st.caption("Colon (COAD) is the hardest — smallest class, transcriptomically close to the "
                   "other adenocarcinomas. An honest limitation.")
    else:
        st.info("Click the button above to train the 5-type quantum classifier on real "
                "gene-expression data.")

# --------------------------------------------------------------------------- #
# Tab 6 — tumor localization (live U-Net training on the GPU)
# --------------------------------------------------------------------------- #
with tab_loc:
    st.subheader("Train a U-Net to localize tumors — live, on your GPU")
    sys.path.insert(0, str(ROOT / "segmentation"))
    msd_root = ROOT / "data_cache" / "msd" / "Task01_BrainTumour"
    has_msd = msd_root.exists()

    from unet import get_device

    dev = get_device()
    st.caption(f"Compute device: **{dev.type.upper()}**"
               + ("  (Apple-Silicon GPU 🚀)" if dev.type == "mps" else ""))

    sources = ["Synthetic (fast pipeline check)"]
    if has_msd:
        sources.insert(0, "MSD Brain-Tumour (real MRI)")
    else:
        st.info("MSD Brain-Tumour data not found yet — synthetic only until the download finishes.")
    source = st.radio("Training data", sources, horizontal=True)
    c1, c2 = st.columns(2)
    epochs = c1.slider("Epochs", 5, 40, 15)
    n_vol = c2.slider("MSD volumes (real only)", 10, 120, 50, 10,
                      disabled=not source.startswith("MSD"))

    if st.button("▶ Start training"):
        st.session_state["seg_go"] = True
    if st.session_state.get("seg_go"):
        with st.spinner("Preparing data…"):
            if source.startswith("MSD"):
                from msd_data import build_msd_slice_dataset
                train_ds, val_ds, info = build_msd_slice_dataset(
                    str(msd_root), n_volumes=n_vol, seed=42)
                st.write(f"Real MRI: {info['train_slices']} train / {info['val_slices']} "
                         f"val slices from {info['volumes']} patients")
            else:
                from seg_data import make_synthetic
                train_ds, val_ds = make_synthetic()

        prog = st.progress(0.0)
        status = st.empty()
        chart = st.empty()
        hist: list[dict] = []

        def on_epoch(s):
            hist.append(s)
            prog.progress(s["epoch"] / s["epochs"])
            status.markdown(
                f"**epoch {s['epoch']}/{s['epochs']}** · train loss `{s['train_loss']:.3f}` · "
                f"val loss `{s['val_loss']:.3f}` · **val Dice `{s['val_dice']:.3f}`** "
                f"(best `{s['best_dice']:.3f}`)")
            chart.line_chart(
                {"epoch": [h["epoch"] for h in hist],
                 "val Dice": [h["val_dice"] for h in hist],
                 "train loss": [h["train_loss"] for h in hist]},
                x="epoch")

        from train_seg import train as train_unet

        model, history = train_unet(train_ds, val_ds, epochs=epochs, on_epoch=on_epoch, device=dev)
        best = max(h["val_dice"] for h in history)
        st.success(f"Training complete — best validation Dice **{best:.3f}**. "
                   f"Checkpoint saved to results/checkpoints/.")

        # show a few predictions
        import matplotlib.pyplot as _plt
        from predict_seg import segment_image
        st.markdown("#### Predictions (input · ground truth · U-Net)")
        figp, axp = _plt.subplots(2, 3, figsize=(8, 5.5))
        for r in range(2):
            x, y = val_ds[r]
            pr = segment_image(model, x[0].numpy(), dev)
            for ax, im, t in zip(axp[r], [x[0].numpy(), y[0].numpy(), pr > 0.5],
                                 ["input", "truth", "prediction"]):
                ax.imshow(im, cmap="gray" if t == "input" else "magma")
                ax.set_title(t if r == 0 else ""); ax.axis("off")
        figp.tight_layout()
        st.pyplot(figp)

    st.warning("Educational/research prototype — **not a medical device.** Synthetic Dice only "
               "validates the pipeline; real accuracy comes from training on real MRI.")
