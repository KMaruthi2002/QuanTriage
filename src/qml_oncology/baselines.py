"""Classical baselines, trained on the identical features and split.

The point of a QML project is an honest comparison. We train two standard,
strong classical models -- logistic regression and an RBF-kernel SVM -- on the
exact same selected, scaled features so the quantum vs. classical numbers are
directly comparable.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def train_baselines(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Return a dict of name -> fitted classical estimator."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=42),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def baseline_pos_proba(model, X: np.ndarray) -> np.ndarray:
    """Probability of the positive (malignant) class for a fitted baseline."""
    return model.predict_proba(X)[:, 1]
