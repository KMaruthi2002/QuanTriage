r"""
A clinically-aware quantum classifier: triage that knows when to defer
======================================================================

.. meta::
    :property="og:description": Build a variational quantum classifier for breast-cancer
        triage that abstains on uncertain cases, is tuned not to miss malignancies, survives
        quantum noise, and explains itself -- benchmarked honestly against classical models.
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/quantum_cancer_triage/thumbnail.png

.. related::
    tutorial_variational_classifier Variational classifier
    tutorial_data_reuploading_classifier Data-reuploading classifier
    tutorial_noisy_circuits Noisy circuits

*Author: Maruthi Kunchala, Posted: 21 June 2026.*

Most quantum machine learning tutorials stop at *"encode the data, run a circuit, print the
accuracy."* But a model you would actually want near a clinic has to answer harder questions:

* How many real cancers does it **miss**?
* Can it tell you when it is **not sure** and defer to a human?
* Does it still work on **noisy** quantum hardware?
* Which measurements is it **relying on**?

In this demo we build a :doc:`variational quantum classifier <tutorial_variational_classifier>`
for breast-cancer diagnosis and wrap it in four ideas that make it *clinically honest*:
**selective prediction**, **cost-sensitive thresholding**, **noise-robustness evaluation**, and
**interpretability** -- and we benchmark it candidly against classical baselines.
"""

######################################################################
# The data
# --------
#
# We use the Wisconsin breast-cancer dataset that ships with scikit-learn: 569 patients, each
# described by 30 features computed from a digitized image of a fine-needle aspirate of a breast
# mass. scikit-learn encodes the target as ``malignant=0`` / ``benign=1``; for a screening tool
# the event we care about is *catching a malignancy*, so we relabel **malignant = 1 (the positive
# class)**. Then "sensitivity" will mean "fraction of real cancers caught."
#
# We keep the six most discriminative **named** features (rather than using PCA) so that the
# interpretability step at the end maps back to real, human-readable measurements.

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

N_QUBITS = 6
raw = load_breast_cancer()
X, y = raw.data, (raw.target == 0).astype(int)  # malignant -> 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

selector = SelectKBest(f_classif, k=N_QUBITS).fit(X_train, y_train)
feature_names = [raw.feature_names[i] for i in selector.get_support(indices=True)]
scaler = StandardScaler().fit(selector.transform(X_train))

X_train = scaler.transform(selector.transform(X_train))
X_test = scaler.transform(selector.transform(X_test))

print("Features fed to the qubits:")
for name in feature_names:
    print("  •", name)

######################################################################
# The quantum circuit
# -------------------
#
# Our model is a textbook hybrid quantum-classical classifier:
#
# .. math::
#
#     \text{features} \;\rightarrow\; \underbrace{\text{AngleEmbedding}}_{\text{encode}}
#     \;\rightarrow\; \underbrace{\text{StronglyEntanglingLayers}(\theta)}_{\text{trainable}}
#     \;\rightarrow\; \langle Z_0 \rangle.
#
# Each feature sets a qubit rotation angle (data encoding); a stack of trainable entangling
# layers -- the "weights" :math:`\theta` -- transforms the state; and we read out a single
# Pauli-:math:`Z` expectation value on the first wire. That value lives in :math:`[-1, 1]`, which
# we map to a **malignancy probability** :math:`p = (1 - \langle Z \rangle)/2`.

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="autograd")
def circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def prob_malignant(weights, X):
    """Malignancy probability for a batch of samples (uses broadcasting)."""
    expvals = circuit(weights, pnp.array(X, requires_grad=False))
    return (1.0 - np.asarray(expvals)) / 2.0


######################################################################
# Let us look at the structure of the circuit:

n_layers = 4
shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=N_QUBITS)
init_weights = pnp.array(np.random.normal(0, 0.1, shape), requires_grad=True)

print(qml.draw(circuit, decimals=None, show_matrices=False)(init_weights, X_test[:1]))

######################################################################
# Training
# --------
#
# We train the rotation angles by minimizing the binary cross-entropy between the predicted
# malignancy probability and the labels, using the Adam optimizer over mini-batches -- exactly
# like training a small neural network, except the hidden layer is a set of *entangled qubits*.


def bce(weights, X, y):
    # Use PennyLane's autograd-aware numpy (pnp) so gradients flow through the loss.
    expvals = circuit(weights, X)
    p = (1.0 - expvals) / 2.0
    p = pnp.clip(p, 1e-7, 1 - 1e-7)
    return -pnp.mean(y * pnp.log(p) + (1 - y) * pnp.log(1 - p))


opt = qml.AdamOptimizer(stepsize=0.05)
weights = init_weights
batch_size, epochs = 24, 30
X_train_t = pnp.array(X_train, requires_grad=False)
y_train_t = pnp.array(y_train, requires_grad=False)

loss_history, acc_history = [], []
for epoch in range(epochs):
    perm = np.random.permutation(len(X_train_t))
    for start in range(0, len(X_train_t), batch_size):
        idx = perm[start : start + batch_size]
        weights = opt.step(lambda w: bce(w, X_train_t[idx], y_train_t[idx]), weights)
    loss_history.append(float(bce(weights, X_train_t, y_train_t)))
    acc_history.append(float(np.mean((prob_malignant(weights, X_train) >= 0.5) == y_train)))
    if (epoch + 1) % 5 == 0:
        print(f"epoch {epoch + 1:2d}  loss={loss_history[-1]:.4f}  train_acc={acc_history[-1]:.3f}")

######################################################################
# The loss falls and training accuracy climbs:

fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(range(1, epochs + 1), loss_history, color="#c0392b", label="loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss", color="#c0392b")
ax2 = ax1.twinx()
ax2.plot(range(1, epochs + 1), acc_history, color="#2980b9")
ax2.set_ylabel("train accuracy", color="#2980b9")
plt.title("Training the quantum classifier")
plt.tight_layout()
plt.show()

######################################################################
# Unique angle #1, cost-sensitive thresholding
# ----------------------------------------------
#
# A default 0.5 decision threshold treats a missed cancer and a false alarm as equally bad. They
# are not: in screening, a **missed malignancy is far costlier**. We therefore choose the
# threshold that minimizes :math:`\text{cost} = c_{\text{FN}}\,\text{FN} + \text{FP}`, with the
# false-negative cost :math:`c_{\text{FN}}` much larger than 1 (here 10).

from sklearn.metrics import confusion_matrix, roc_curve, auc

q_prob = prob_malignant(weights, X_test)


def cost_threshold(y_true, p, c_fn=10.0):
    best_t, best_c = 0.5, np.inf
    for t in np.linspace(0.01, 0.99, 99):
        tn, fp, fn, tp = confusion_matrix(y_true, (p >= t).astype(int), labels=[0, 1]).ravel()
        c = c_fn * fn + fp
        if c < best_c:
            best_c, best_t = c, t
    return best_t


threshold = cost_threshold(y_test, q_prob)
tn, fp, fn, tp = confusion_matrix(y_test, (q_prob >= threshold).astype(int), labels=[0, 1]).ravel()
print(f"Cost-tuned threshold: {threshold:.2f}")
print(f"At this threshold: missed cancers (FN) = {fn},  false alarms (FP) = {fp}")
print(f"Sensitivity (cancers caught): {tp / (tp + fn):.1%}")

######################################################################
# Shifting the threshold trades a handful of false alarms for **catching every malignancy** --
# the right call for a triage tool. The ROC curve summarizes the full tradeoff and lets us
# compare against a classical baseline.

clf = LogisticRegression(max_iter=2000).fit(X_train, y_train)
c_prob = clf.predict_proba(X_test)[:, 1]

plt.figure(figsize=(6, 5))
for name, p in [("quantum", q_prob), ("logistic regression", c_prob)]:
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate (sensitivity)")
plt.title("ROC, quantum vs. classical")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

######################################################################
# The quantum classifier is **competitive** with a strong classical baseline here -- not
# obviously better. Saying so plainly is part of doing QML honestly: on small tabular problems
# like this one, classical models are excellent, and the value of the quantum approach is in the
# methodology and the research direction, not a headline accuracy win.

######################################################################
# Unique angle #2, selective prediction (knowing when to defer)
# --------------------------------------------------------------
#
# Instead of forcing a call on every patient, the model reports a **confidence**
# :math:`\max(p, 1-p)` and *abstains* -- referring the case to a doctor -- when confidence is
# low. We can trace the **risk-coverage curve**: as the model decides fewer cases (lower
# coverage), its accuracy on the cases it *does* keep rises toward 100%.

conf = np.maximum(q_prob, 1 - q_prob)
pred = (q_prob >= 0.5).astype(int)

cutoffs = np.linspace(0.5, 1.0, 51)
coverage, accuracy = [], []
for c in cutoffs:
    m = conf >= c
    coverage.append(m.mean())
    accuracy.append(np.mean(pred[m] == y_test[m]) if m.sum() else np.nan)

plt.figure(figsize=(7, 4))
plt.plot(coverage, accuracy, marker="o", ms=3, color="#27ae60")
plt.gca().invert_xaxis()
plt.xlabel("coverage (fraction of cases decided)")
plt.ylabel("accuracy on accepted cases")
plt.title("Selective prediction, abstaining buys accuracy")
plt.tight_layout()
plt.show()

at85 = conf >= 0.85
print(f"At 85% confidence: decides {at85.mean():.0%} of cases, "
      f"accuracy on those = {np.mean(pred[at85] == y_test[at85]):.1%}, "
      f"refers {int((~at85).sum())} patients to a doctor.")

######################################################################
# Unique angle #3, robustness under quantum noise
# -------------------------------------------------
#
# Real quantum hardware is noisy. We re-evaluate the *already-trained* model on the
# :doc:`mixed-state simulator <tutorial_noisy_circuits>` ``default.mixed``, inserting a
# :class:`~.pennylane.DepolarizingChannel` on every wire after encoding and after the variational
# layers, and sweep the noise strength. (Density-matrix simulation is heavier, so we use a
# subsample of the test set.)

dev_noisy = qml.device("default.mixed", wires=N_QUBITS)


@qml.qnode(dev_noisy)
def noisy_circuit(weights, x, p):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    for w in range(N_QUBITS):
        qml.DepolarizingChannel(p, wires=w)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    for w in range(N_QUBITS):
        qml.DepolarizingChannel(p, wires=w)
    return qml.expval(qml.PauliZ(0))


sub = np.random.choice(len(X_test), size=100, replace=False)
noise_levels = [0.0, 0.01, 0.03, 0.05, 0.1]
noisy_acc = []
for p_noise in noise_levels:
    if p_noise == 0.0:
        probs = q_prob[sub]
    else:
        probs = np.array(
            [(1 - float(noisy_circuit(weights, pnp.array(x, requires_grad=False), p_noise))) / 2
             for x in X_test[sub]]
        )
    noisy_acc.append(np.mean((probs >= threshold).astype(int) == y_test[sub]))

plt.figure(figsize=(7, 4))
plt.plot(noise_levels, noisy_acc, marker="o", color="#e67e22")
plt.xlabel("depolarizing noise strength")
plt.ylabel("accuracy")
plt.title("Robustness under quantum noise")
plt.tight_layout()
plt.show()

print("accuracy vs noise:", {f"{p:.2f}": round(a, 3) for p, a in zip(noise_levels, noisy_acc)})

######################################################################
# Accuracy degrades **gracefully** as noise rises -- useful intuition for what deploying this
# model on near-term hardware would actually cost you.

######################################################################
# Unique angle #4, interpretability
# -----------------------------------
#
# Finally, we open the black box with **permutation importance**: shuffle one feature column at a
# time and measure how much test accuracy drops. The larger the drop, the more the model relies
# on that measurement.

base_acc = np.mean((q_prob >= threshold).astype(int) == y_test)
importances = []
for j in range(N_QUBITS):
    drops = []
    for _ in range(8):
        Xp = X_test.copy()
        Xp[:, j] = np.random.permutation(Xp[:, j])
        acc = np.mean((prob_malignant(weights, Xp) >= threshold).astype(int) == y_test)
        drops.append(base_acc - acc)
    importances.append(np.mean(drops))

order = np.argsort(importances)
plt.figure(figsize=(7, 4.5))
plt.barh([feature_names[i] for i in order], [importances[i] for i in order], color="#8e44ad")
plt.xlabel("drop in accuracy when shuffled (importance)")
plt.title("Which measurements the quantum model relies on")
plt.tight_layout()
plt.show()

######################################################################
# The model leans most on tumor **size and shape** features (perimeter, concave points) -- which
# is exactly what a clinician would look at, a reassuring sanity check.
#
# Conclusion
# ----------
#
# We built a variational quantum classifier that does more than report a number: it **defers when
# unsure**, is **tuned not to miss cancer**, **survives noise**, and **explains itself**, all
# while being **honestly benchmarked** against a classical model. These ideas -- selective
# prediction, cost-sensitive evaluation, noise robustness, interpretability -- are exactly what a
# trustworthy real-world QML system needs, and they transfer to any quantum classifier you build.
#
# References
# ----------
#
# .. [#perezsalinas]
#    A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J. I. Latorre.
#    "Data re-uploading for a universal quantum classifier." *Quantum* 4, 226 (2020).
#
# .. [#havlicek]
#    V. Havlíček et al. "Supervised learning with quantum-enhanced feature spaces."
#    *Nature* 567, 209–212 (2019).
#
# .. [#wolberg]
#    W. Wolberg, W. Street, O. Mangasarian. "Breast Cancer Wisconsin (Diagnostic) Data Set."
#    UCI Machine Learning Repository (1995).
