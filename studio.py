"""🔬 Quantum Training Studio — bring your own data, train the quantum model live.

Load a CSV (e.g. a dataset you downloaded from Kaggle), pick the target column,
and train the variational quantum classifier while watching every epoch, the
loss/accuracy curves, and the live state of the model (weight norm, per-layer
stats, the circuit). Save a checkpoint when you're happy.

    .venv/bin/streamlit run studio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pennylane as qml
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from multiclass_model import MultiClassQuantumClassifier
from quantum_model import QuantumClassifier

CKPT_DIR = ROOT / "results" / "checkpoints"
st.set_page_config(page_title="Quantum Training Studio", page_icon="🔬", layout="wide")

st.title("🔬 Quantum Training Studio")
st.caption("Bring your own data (e.g. a Kaggle CSV) and train the quantum classifier live — "
           "watch every epoch and the state of the model.")


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
st.subheader("1 · Data")
source = st.radio("Source", ["Upload CSV (Kaggle, etc.)", "Built-in demo: breast cancer"],
                  horizontal=True)

df = None
if source.startswith("Upload"):
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
else:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.rename(columns={"target": "diagnosis"})

if df is None:
    st.info("Upload a CSV to begin, or switch to the built-in demo dataset.")
    st.stop()

st.dataframe(df.head(), width="stretch", height=180)
st.caption(f"{len(df):,} rows × {df.shape[1]} columns")


# --------------------------------------------------------------------------- #
# 2. Configure
# --------------------------------------------------------------------------- #
st.subheader("2 · Configure")
cols = list(df.columns)
target = st.selectbox("Target column (what to predict)", cols, index=len(cols) - 1)

y_raw = df[target]
le = LabelEncoder()
y = le.fit_transform(y_raw.astype(str))
class_names = list(le.classes_)
n_classes = len(class_names)

numeric = df.drop(columns=[target]).select_dtypes(include=[np.number])
if numeric.shape[1] < 2 or n_classes < 2:
    st.error("Need ≥2 numeric feature columns and ≥2 classes in the target.")
    st.stop()

task = "binary" if n_classes == 2 else f"multiclass ({n_classes})"
c1, c2, c3, c4 = st.columns(4)
c1.metric("Task", task)
c2.metric("Classes", n_classes)
max_q = int(min(12, numeric.shape[1]))
qubits = c2.slider("Qubits / features", 2, max_q, min(8 if n_classes > 2 else 6, max_q))
layers = c3.slider("Layers", 2, 6, 4)
epochs = c4.slider("Epochs", 5, 60, 25)
st.caption(f"Classes: {', '.join(map(str, class_names))}  ·  "
           f"{numeric.shape[1]} numeric features available")


# --------------------------------------------------------------------------- #
# 3. Train (live)
# --------------------------------------------------------------------------- #
st.subheader("3 · Train")
if st.button("▶ Train quantum model"):
    st.session_state["go"] = True
if st.session_state.get("go"):
    X = numeric.to_numpy(dtype=float)
    # impute any NaNs, select top-k features, scale, split
    X = np.nan_to_num(X, nan=np.nanmean(X) if np.isfinite(np.nanmean(X)) else 0.0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    k = min(qubits, X.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
    feat_names = [numeric.columns[i] for i in sel.get_support(indices=True)]
    scaler = StandardScaler().fit(sel.transform(Xtr))
    Xtr_s, Xte_s = scaler.transform(sel.transform(Xtr)), scaler.transform(sel.transform(Xte))

    prog = st.progress(0.0)
    status = st.empty()
    left, right = st.columns([3, 2])
    chart = left.empty()
    state = right.empty()
    hist: list[dict] = []

    def on_epoch(s):
        hist.append(s)
        prog.progress(s["epoch"] / s["epochs"])
        status.markdown(
            f"**epoch {s['epoch']}/{s['epochs']}** · loss `{s['loss']:.4f}` · "
            f"train acc `{s['acc']:.3f}` · ‖weights‖ `{s['weight_norm']:.2f}`")
        chart.line_chart({"epoch": [h["epoch"] for h in hist],
                          "loss": [h["loss"] for h in hist],
                          "train acc": [h["acc"] for h in hist]}, x="epoch")
        state.markdown(
            f"**Model state**\n\n- epoch: {s['epoch']}/{s['epochs']}\n"
            f"- loss: `{s['loss']:.4f}`\n- train accuracy: `{s['acc']:.3f}`\n"
            f"- weight L2 norm: `{s['weight_norm']:.3f}`")

    if n_classes == 2:
        model = QuantumClassifier(n_qubits=k, n_layers=layers, epochs=epochs,
                                  seed=42, verbose=False, on_epoch=on_epoch)
    else:
        model = MultiClassQuantumClassifier(n_classes=n_classes, n_qubits=max(k, n_classes),
                                            n_layers=layers, epochs=epochs, seed=42,
                                            verbose=False, on_epoch=on_epoch)
    model.fit(Xtr_s, ytr)

    # --- results ---
    pred = model.predict(Xte_s)
    acc = accuracy_score(yte, pred)
    st.success(f"Done · test accuracy **{acc:.3f}** on {len(yte)} held-out rows")

    r1, r2 = st.columns(2)
    cm = confusion_matrix(yte, pred, labels=range(n_classes))
    import plotly.graph_objects as go
    heat = go.Figure(go.Heatmap(z=cm, x=class_names, y=class_names, colorscale="Blues",
                                text=cm, texttemplate="%{text}"))
    heat.update_layout(title="Confusion matrix", xaxis_title="predicted",
                       yaxis_title="actual", height=380)
    r1.plotly_chart(heat, use_container_width=True)

    # model state: weights + circuit
    w = np.asarray(model.weights_)
    r2.markdown("**Final model state**")
    r2.write({"weight tensor shape": list(w.shape),
              "weight L2 norm": round(float(np.linalg.norm(w)), 3),
              "weight mean": round(float(w.mean()), 4),
              "weight std": round(float(w.std()), 4),
              "features used": feat_names})
    try:
        x0 = np.asarray(Xte_s[:1])
        drawing = qml.draw(model.circuit_, decimals=None, show_matrices=False,
                           max_length=90)(model.weights_, x0)
        r2.code(drawing, language="text")
    except Exception:
        pass

    # checkpoint
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if n_classes == 2:
        ckpt = CKPT_DIR / "studio_model.npz"
        model.save_checkpoint(ckpt)
        with open(ckpt, "rb") as fh:
            st.download_button("⬇ Download checkpoint", fh, file_name="studio_model.npz")

st.divider()
st.caption("Educational/research tool — **not a medical device.** Trains the same variational "
           "quantum classifier used across the project, on any tabular data you provide.")
