"""A PyTorch hybrid quantum-classical model via ``qml.qnn.TorchLayer``.

Stacks a quantum circuit *inside* a normal PyTorch network:

    features ─▶ Linear ─▶ [Quantum TorchLayer] ─▶ Linear ─▶ logits

The quantum layer is just another differentiable `nn.Module`, so the whole thing
trains end-to-end with a standard PyTorch optimizer and autograd. This is how you
combine a quantum feature map with deep classical layers.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class HybridQuantumNet(nn.Module):
    def __init__(self, in_features, n_qubits=6, n_layers=3, n_classes=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre = nn.Linear(in_features, n_qubits)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        wshapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(qnode, wshapes)
        self.post = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        x = torch.tanh(self.pre(x)) * (np.pi / 2)   # scale to rotation angles
        x = self.qlayer(x)                           # quantum feature map
        return self.post(x)                          # classical head


class HybridQuantumClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-style wrapper that trains the hybrid net with PyTorch."""

    def __init__(self, n_qubits=6, n_layers=3, epochs=30, lr=0.02, batch_size=24,
                 seed=42, verbose=True, on_epoch=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.on_epoch = on_epoch

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.net_ = HybridQuantumNet(X.shape[1], self.n_qubits, self.n_layers, n_classes)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.long)

        self.history_ = {"loss": [], "acc": []}
        n = len(Xt)
        for epoch in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss = ce(self.net_(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            with torch.no_grad():
                logits = self.net_(Xt)
                ep_loss = float(ce(logits, yt))
                acc = float((logits.argmax(1) == yt).float().mean())
            self.history_["loss"].append(ep_loss)
            self.history_["acc"].append(acc)
            if self.on_epoch:
                self.on_epoch({"epoch": epoch + 1, "epochs": self.epochs, "loss": ep_loss, "acc": acc})
            if self.verbose:
                print(f"  [hybrid] epoch {epoch + 1:2d}/{self.epochs}  loss={ep_loss:.4f}  acc={acc:.3f}")
        return self

    def predict_proba(self, X):
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(np.asarray(X), dtype=torch.float32))
            return torch.softmax(logits, 1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
