"""Data loading and preprocessing for the QuanTriage breast-cancer classifier.

We deliberately use *feature selection* (not PCA) so the features fed into the
quantum circuit stay human-readable named clinical measurements (e.g.
"worst radius", "worst concave points"). That makes the interpretability step
later actually meaningful to a clinician.

Label convention
----------------
scikit-learn encodes the breast-cancer target as malignant=0, benign=1.
For a cancer screen the clinically important event is *catching a malignancy*,
so we relabel to **malignant = 1 (positive class)**. Then "sensitivity/recall"
means "fraction of real cancers the model caught" -- the metric that matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    """A train/test split with the selected, scaled features and their names."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    # positive class (malignant) prevalence in the test set, handy for reporting
    test_prevalence: float
    # fitted scaler, so callers (e.g. the GUI) can transform raw measurements
    scaler: StandardScaler
    # per-feature stats in ORIGINAL units (from the training set), so a GUI can
    # build sliders in real measurement units: list of {name,min,max,mean,median}
    feature_stats: list[dict]

    def transform_raw(self, raw_values) -> np.ndarray:
        """Scale a vector/array of raw selected-feature values for prediction."""
        arr = np.atleast_2d(np.asarray(raw_values, dtype=float))
        return self.scaler.transform(arr)


def load_data(n_features: int = 6, test_size: float = 0.25, seed: int = 42) -> Dataset:
    """Load breast-cancer data, select the top `n_features`, scale, and split.

    The selector and scaler are fit on the *training* split only to avoid
    leaking test information into preprocessing.
    """
    raw = load_breast_cancer()
    X = raw.data
    # relabel so malignant (sklearn class 0) becomes the positive class (1)
    y = (raw.target == 0).astype(int)
    all_names = list(raw.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # pick the most discriminative named features (fit on train only)
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_train_sel = selector.fit_transform(X_train, y_train)  # selected, original units
    X_test_sel = selector.transform(X_test)
    selected = [all_names[i] for i in selector.get_support(indices=True)]

    # capture per-feature stats in ORIGINAL units (training set) for GUI sliders,
    # including class-conditional means so a GUI can offer example patients
    mal_mask = y_train == 1
    feature_stats = []
    for j, name in enumerate(selected):
        col = X_train_sel[:, j]
        feature_stats.append(
            {
                "name": name,
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "median": float(np.median(col)),
                "mean_malignant": float(col[mal_mask].mean()),
                "mean_benign": float(col[~mal_mask].mean()),
            }
        )

    # standardize (fit on train only); keeps angle-embedding inputs well-scaled
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_sel)
    X_test = scaler.transform(X_test_sel)

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=selected,
        test_prevalence=float(np.mean(y_test)),
        scaler=scaler,
        feature_stats=feature_stats,
    )
