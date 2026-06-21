"""A fully configurable quantum classifier — every knob exposed.

Lets you choose:
  * **encoding / ansatz**: angle embedding, amplitude embedding, or a
    data-reuploading circuit (re-embed the data between trainable layers for more
    expressivity — Pérez-Salinas et al. 2020).
  * **model size**: any number of qubits and layers (and re-upload depth).
  * **device / backend**: local simulators (`default.qubit`, the fast C++
    `lightning.qubit`) or real-hardware-style backends (`qiskit.aer` sampling,
    IBM Quantum via `qiskit.remote`, AWS Braket) — see ``make_device``.

Train on a fast simulator, then run the *same trained circuit* on real hardware
with ``predict_on_device`` — the standard QML workflow.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin

ENCODINGS = ("angle", "amplitude", "reupload")


def make_device(name: str = "default.qubit", wires: int = 4, shots=None, **kwargs):
    """Device factory covering simulators and real-hardware backends.

    Examples
    --------
    Local simulators:
        make_device("default.qubit", 6)
        make_device("lightning.qubit", 12)        # fast C++ simulator
        make_device("qiskit.aer", 6, shots=1024)  # local sampling simulator

    Real IBM Quantum hardware (needs an IBM Quantum API token):
        make_device("qiskit.remote", 5, shots=1024,
                    backend="ibm_brisbane", token="YOUR_IBMQ_TOKEN")

    Real AWS Braket hardware (needs AWS credentials configured):
        make_device("braket.aws.qubit", 5, shots=1024,
                    device_arn="arn:aws:braket:...:device/qpu/ionq/Aria-1")
    """
    return qml.device(name, wires=wires, shots=shots, **kwargs)


def _prob(expval):
    return (1.0 - expval) / 2.0


class FlexibleQuantumClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, encoding="angle", n_qubits=6, n_layers=4, reuploads=2,
                 device="default.qubit", shots=None, epochs=30, stepsize=0.05,
                 batch_size=24, seed=42, verbose=True, on_epoch=None):
        self.encoding = encoding
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reuploads = reuploads
        self.device = device
        self.shots = shots
        self.epochs = epochs
        self.stepsize = stepsize
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.on_epoch = on_epoch

    # -- circuit construction ------------------------------------------------
    def _weight_shape(self):
        base = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        return (self.reuploads, *base) if self.encoding == "reupload" else base

    def _make_qnode(self, device_name=None, shots=None):
        nq, enc, reup = self.n_qubits, self.encoding, self.reuploads
        dev = make_device(device_name or self.device, nq, shots if shots is not None else self.shots)

        @qml.qnode(dev, interface="autograd", diff_method="best")
        def circuit(weights, x):
            if enc == "amplitude":
                qml.AmplitudeEmbedding(x, wires=range(nq), normalize=True, pad_with=0.0)
                qml.StronglyEntanglingLayers(weights, wires=range(nq))
            elif enc == "reupload":
                for r in range(reup):
                    qml.AngleEmbedding(x, wires=range(nq))
                    qml.StronglyEntanglingLayers(weights[r], wires=range(nq))
            else:  # angle
                qml.AngleEmbedding(x, wires=range(nq))
                qml.StronglyEntanglingLayers(weights, wires=range(nq))
            return qml.expval(qml.PauliZ(0))

        return circuit

    # -- sklearn API ---------------------------------------------------------
    def fit(self, X, y):
        assert self.encoding in ENCODINGS, f"encoding must be one of {ENCODINGS}"
        self.classes_ = np.array([0, 1])
        self.circuit_ = self._make_qnode()
        rng = np.random.default_rng(self.seed)
        weights = pnp.array(rng.normal(0, 0.1, self._weight_shape()), requires_grad=True)
        Xt = pnp.array(X, requires_grad=False)
        yt = pnp.array(y, requires_grad=False)
        opt = qml.AdamOptimizer(stepsize=self.stepsize)

        def cost(w, xb, yb):
            p = _prob(self.circuit_(w, xb))
            p = pnp.clip(p, 1e-7, 1 - 1e-7)
            return -pnp.mean(yb * pnp.log(p) + (1 - yb) * pnp.log(1 - p))

        n = len(Xt)
        self.history_ = {"loss": [], "acc": []}
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                weights = opt.step(lambda w: cost(w, Xt[idx], yt[idx]), weights)
            self.weights_ = weights
            loss = float(cost(weights, Xt, yt))
            acc = float(np.mean((self.predict_proba(X)[:, 1] >= 0.5) == y))
            self.history_["loss"].append(loss)
            self.history_["acc"].append(acc)
            if self.on_epoch:
                self.on_epoch({"epoch": epoch + 1, "epochs": self.epochs, "loss": loss,
                               "acc": acc, "weight_norm": float(np.linalg.norm(np.asarray(weights)))})
            if self.verbose:
                print(f"  [{self.encoding}] epoch {epoch + 1:2d}/{self.epochs}  "
                      f"loss={loss:.4f}  acc={acc:.3f}")
        self.weights_ = weights
        return self

    def predict_proba(self, X):
        p = _prob(np.asarray(self.circuit_(self.weights_, pnp.array(X, requires_grad=False))))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def save_checkpoint(self, path):
        np.savez(path, weights=np.asarray(self.weights_), encoding=self.encoding,
                 n_qubits=self.n_qubits, n_layers=self.n_layers, reuploads=self.reuploads)

    def load_checkpoint(self, path):
        d = np.load(path, allow_pickle=True)
        self.encoding = str(d["encoding"])
        self.n_qubits = int(d["n_qubits"])
        self.n_layers = int(d["n_layers"])
        self.reuploads = int(d["reuploads"])
        self.classes_ = np.array([0, 1])
        self.circuit_ = self._make_qnode()
        self.weights_ = pnp.array(d["weights"], requires_grad=False)
        return self

    def predict_on_device(self, X, device_name, shots=1024):
        """Run the trained circuit on a DIFFERENT backend (e.g. real hardware).

        Rebuilds the qnode on `device_name` with the trained weights — this is how
        you take a simulator-trained model to IBM Quantum / Braket.
        """
        circ = self._make_qnode(device_name=device_name, shots=shots)
        p = _prob(np.asarray([float(circ(self.weights_, pnp.array(x, requires_grad=False))) for x in X]))
        return np.column_stack([1 - p, p])
