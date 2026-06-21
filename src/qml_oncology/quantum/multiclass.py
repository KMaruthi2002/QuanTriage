"""A multi-class variational quantum classifier.

Extends the binary model to N classes: we measure Pauli-Z on the first N qubits,
treat those N expectation values as class logits, and train with softmax
cross-entropy. This is what lets the same quantum approach classify *which* of
several tumor types a gene-expression profile belongs to (e.g. breast vs. lung
vs. kidney vs. prostate vs. colon).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin


class MultiClassQuantumClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_classes, n_qubits=8, n_layers=4, epochs=30,
                 stepsize=0.05, batch_size=32, seed=42, verbose=True,
                 class_weight="balanced", weight_power=0.75, on_epoch=None):
        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.stepsize = stepsize
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        # "balanced" weights the loss inversely to class frequency so rare tumor
        # types (e.g. colon) are not ignored; None = unweighted. weight_power
        # tempers the strength: 0 = unweighted, 0.5 = sqrt-soft, 1 = full balanced.
        self.class_weight = class_weight
        self.weight_power = weight_power
        self.on_epoch = on_epoch

    def _build(self):
        if self.n_qubits < self.n_classes:
            raise ValueError("n_qubits must be >= n_classes (one readout qubit per class)")
        dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(weights, x):
            qml.AngleEmbedding(x, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]

        self.circuit_ = circuit

    def _logits(self, weights, X):
        # circuit returns n_classes arrays each (batch,) -> stack to (batch, n_classes)
        outs = self.circuit_(weights, X)
        return pnp.stack(outs, axis=1)

    def fit(self, X, y):
        self.classes_ = np.arange(self.n_classes)
        self._build()
        rng = np.random.default_rng(self.seed)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        weights = pnp.array(rng.normal(0, 0.1, shape), requires_grad=True)

        X_t = pnp.array(X, requires_grad=False)
        onehot = pnp.array(np.eye(self.n_classes)[y], requires_grad=False)
        # per-sample weights (sklearn "balanced" formula): total / (n_classes * count)
        if self.class_weight == "balanced":
            # tempered balanced weights (weight_power): lifts rare classes (colon)
            # without letting them starve the majority class (breast).
            counts = np.bincount(y, minlength=self.n_classes)
            cw = (len(y) / (self.n_classes * np.maximum(counts, 1))) ** self.weight_power
            sw = pnp.array(cw[y], requires_grad=False)
        else:
            sw = pnp.array(np.ones(len(y)), requires_grad=False)
        opt = qml.AdamOptimizer(stepsize=self.stepsize)

        def cost(w, xb, oh, wb):
            logits = self._logits(w, xb)
            logits = logits - pnp.max(logits, axis=1, keepdims=True)
            logp = logits - pnp.log(pnp.sum(pnp.exp(logits), axis=1, keepdims=True))
            return -pnp.sum(wb * pnp.sum(oh * logp, axis=1)) / pnp.sum(wb)

        n = len(X_t)
        self.history_ = {"loss": [], "acc": []}
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                weights = opt.step(lambda w: cost(w, X_t[idx], onehot[idx], sw[idx]), weights)
            self.weights_ = weights
            loss = float(cost(weights, X_t, onehot, sw))
            acc = float(np.mean(self.predict(X) == y))
            self.history_["loss"].append(loss)
            self.history_["acc"].append(acc)
            if self.on_epoch:
                self.on_epoch({"epoch": epoch + 1, "epochs": self.epochs,
                               "loss": loss, "acc": acc,
                               "weight_norm": float(np.linalg.norm(np.asarray(weights)))})
            if self.verbose:
                print(f"  epoch {epoch + 1:2d}/{self.epochs}  loss={loss:.4f}  train_acc={acc:.3f}")
        self.weights_ = weights
        return self

    def predict_proba(self, X):
        logits = np.asarray(self._logits(self.weights_, pnp.array(X, requires_grad=False)))
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
