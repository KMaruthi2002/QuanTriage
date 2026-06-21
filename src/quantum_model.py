"""The variational quantum classifier and its noisy-hardware variant.

Architecture (a textbook hybrid quantum-classical model):

    features --> AngleEmbedding --> StronglyEntanglingLayers --> <Z_0>
                 (encode data)      (trainable "weights")        (measure)

The single Pauli-Z expectation value on the first qubit, which lives in
[-1, 1], is mapped to a malignancy probability   p = (1 - <Z>) / 2.
Training nudges the circuit's rotation angles so that probability matches the
labels (binary cross-entropy), exactly like training a tiny neural net -- only
the "neurons" are entangled qubits.

The class is a scikit-learn-compatible estimator so it drops straight into
sklearn tooling (permutation_importance, scoring, etc.).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.base import BaseEstimator, ClassifierMixin


def _build_qnode(n_qubits: int):
    """Return a (device, qnode) pair for the noiseless classifier circuit.

    The qnode supports parameter broadcasting: pass `x` of shape
    (batch, n_qubits) and it returns one expectation value per row.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights, x):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return dev, circuit


def _prob_malignant(expval):
    """Map a Pauli-Z expectation in [-1, 1] to a probability in [0, 1]."""
    return (1.0 - expval) / 2.0


class QuantumClassifier(BaseEstimator, ClassifierMixin):
    """A variational quantum classifier with an sklearn-style API."""

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 4,
        epochs: int = 40,
        stepsize: float = 0.05,
        batch_size: int = 24,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.stepsize = stepsize
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

    # -- internal helpers ---------------------------------------------------
    def _ensure_circuit(self):
        if not hasattr(self, "circuit_"):
            self._dev, self.circuit_ = _build_qnode(self.n_qubits)

    def _proba_pos(self, X):
        """Probability of the positive (malignant) class for each row of X."""
        self._ensure_circuit()
        x = pnp.array(X, requires_grad=False)
        expvals = self.circuit_(self.weights_, x)
        return _prob_malignant(np.asarray(expvals))

    # -- sklearn API --------------------------------------------------------
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self._dev, self.circuit_ = _build_qnode(self.n_qubits)

        rng = np.random.default_rng(self.seed)
        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.n_qubits
        )
        weights = pnp.array(rng.normal(0, 0.1, size=shape), requires_grad=True)

        X_t = pnp.array(X, requires_grad=False)
        y_t = pnp.array(y, requires_grad=False)
        opt = qml.AdamOptimizer(stepsize=self.stepsize)

        def cost(w, xb, yb):
            preds = self.circuit_(w, xb)
            p = _prob_malignant(preds)
            p = pnp.clip(p, 1e-7, 1 - 1e-7)  # keep the log finite
            return -pnp.mean(yb * pnp.log(p) + (1 - yb) * pnp.log(1 - p))

        n = len(X_t)
        self.history_ = {"loss": [], "acc": []}
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                weights = opt.step(lambda w: cost(w, X_t[idx], y_t[idx]), weights)

            self.weights_ = weights
            epoch_loss = float(cost(weights, X_t, y_t))
            epoch_acc = float(np.mean((self._proba_pos(X) >= 0.5) == y))
            self.history_["loss"].append(epoch_loss)
            self.history_["acc"].append(epoch_acc)
            if self.verbose:
                print(
                    f"  epoch {epoch + 1:3d}/{self.epochs}  "
                    f"loss={epoch_loss:.4f}  train_acc={epoch_acc:.3f}"
                )

        self.weights_ = weights
        return self

    def predict_proba(self, X):
        p_pos = self._proba_pos(X)
        return np.column_stack([1 - p_pos, p_pos])

    def predict(self, X, threshold: float = 0.5):
        return (self._proba_pos(X) >= threshold).astype(int)

    def draw(self, X) -> str:
        """Return a clean ASCII drawing of the circuit structure for one sample.

        `decimals=None` / `show_matrices=False` hide the numeric angles so the
        gate *structure* is visible (RX data encoding + Rot/CNOT entangling
        layers) instead of dumping the weight tensor.
        """
        self._ensure_circuit()
        x = pnp.array(X[:1], requires_grad=False)
        return qml.draw(
            self.circuit_, decimals=None, show_matrices=False, max_length=100
        )(self.weights_, x)


def noisy_probabilities(
    model: "QuantumClassifier", X: np.ndarray, noise_level: float
) -> np.ndarray:
    """Evaluate the *trained* model under depolarizing noise (mixed-state sim).

    We reuse the learned weights but run the circuit on `default.mixed`, adding
    a depolarizing channel on every wire after data encoding and after the
    variational layers -- a simple stand-in for the decoherence real quantum
    hardware suffers. Returns malignancy probabilities for each row of X.

    Evaluated sample-by-sample (density-matrix simulation is heavier), so callers
    typically pass a subsample of the test set.
    """
    n_qubits = model.n_qubits
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev)
    def noisy_circuit(weights, x, p):
        qml.AngleEmbedding(x, wires=range(n_qubits))
        for w in range(n_qubits):
            qml.DepolarizingChannel(p, wires=w)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        for w in range(n_qubits):
            qml.DepolarizingChannel(p, wires=w)
        return qml.expval(qml.PauliZ(0))

    weights = model.weights_
    probs = []
    for row in X:
        ev = noisy_circuit(weights, pnp.array(row, requires_grad=False), noise_level)
        probs.append(_prob_malignant(float(ev)))
    return np.array(probs)
