"""A small registry of real cancer datasets, behind one unified interface.

This lets the quantum pipeline run on more than just breast cancer:

  * ``breast_cancer``     , Wisconsin diagnostic, binary (malignant vs benign)
  * ``pan_cancer_rnaseq`` , TCGA pan-cancer gene expression, 5 tumor types
                              (BRCA breast, KIRC kidney, LUAD lung, PRAD prostate,
                               COAD colon), the "multiple cancers" dataset

Add another dataset by writing a loader that returns a ``CancerDataset`` and
registering it in ``DATASETS``.
"""

from __future__ import annotations

import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
CACHE = ROOT / "data_cache"


@dataclass
class CancerDataset:
    key: str
    name: str
    task: str  # "binary" or "multiclass"
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def _prep(X, y, feature_names, n_features, seed):
    """Shared: stratified split, SelectKBest on train, standardize."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    k = min(n_features, X.shape[1])
    selector = SelectKBest(f_classif, k=k).fit(X_train, y_train)
    keep = selector.get_support(indices=True)
    names = [feature_names[i] for i in keep]
    scaler = StandardScaler().fit(selector.transform(X_train))
    X_train = scaler.transform(selector.transform(X_train))
    X_test = scaler.transform(selector.transform(X_test))
    return X_train, X_test, y_train, y_test, names


def load_breast_cancer_ds(n_features: int = 6, seed: int = 42) -> CancerDataset:
    raw = load_breast_cancer()
    y = (raw.target == 0).astype(int)  # malignant -> 1 (positive)
    Xtr, Xte, ytr, yte, names = _prep(raw.data, y, list(raw.feature_names), n_features, seed)
    return CancerDataset(
        key="breast_cancer", name="Breast cancer (Wisconsin diagnostic)", task="binary",
        X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        feature_names=names, class_names=["benign", "malignant"],
    )


def _ensure_pancancer() -> Path:
    """Download + unpack the TCGA pan-cancer RNA-seq dataset on first use."""
    base = CACHE / "TCGA-PANCAN-HiSeq-801x20531"
    if (base / "data.csv").exists():
        return base
    CACHE.mkdir(parents=True, exist_ok=True)
    zpath = CACHE / "pancan.zip"
    if not zpath.exists():
        url = "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip"
        urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(CACHE)
    tar = next(CACHE.glob("*.tar.gz"))
    with tarfile.open(tar) as t:
        t.extractall(CACHE)
    return base


def _multiclass_select(X_train, y_train, n_features):
    """Per-class (one-vs-rest) gene selection so every tumor type gets marker
    genes, otherwise a global F-test starves the smallest class (colon)."""
    classes = np.unique(y_train)
    per = max(1, n_features // len(classes))
    chosen: list[int] = []
    for c in classes:
        F, _ = f_classif(X_train, (y_train == c).astype(int))
        order = np.argsort(np.nan_to_num(F))[::-1]
        added = 0
        for idx in order:
            if idx not in chosen:
                chosen.append(int(idx))
                added += 1
                if added >= per:
                    break
    # top up to n_features using the global ranking
    if len(chosen) < n_features:
        Fg, _ = f_classif(X_train, y_train)
        for idx in np.argsort(np.nan_to_num(Fg))[::-1]:
            if int(idx) not in chosen:
                chosen.append(int(idx))
            if len(chosen) >= n_features:
                break
    return np.array(chosen[:n_features])


def load_pancancer_ds(n_features: int = 8, seed: int = 42) -> CancerDataset:
    import pandas as pd

    base = _ensure_pancancer()
    X = pd.read_csv(base / "data.csv", index_col=0)
    labels = pd.read_csv(base / "labels.csv", index_col=0).iloc[:, 0]
    gene_names = list(X.columns)
    class_names = sorted(labels.unique())
    code = {c: i for i, c in enumerate(class_names)}
    y = labels.map(code).to_numpy()

    Xall = X.to_numpy(dtype=np.float32)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        Xall, y, test_size=0.25, random_state=seed, stratify=y
    )
    keep = _multiclass_select(Xtr_raw, ytr, n_features)
    names = [gene_names[i] for i in keep]
    scaler = StandardScaler().fit(Xtr_raw[:, keep])
    Xtr = scaler.transform(Xtr_raw[:, keep])
    Xte = scaler.transform(Xte_raw[:, keep])
    return CancerDataset(
        key="pan_cancer_rnaseq", name="TCGA pan-cancer RNA-seq (5 tumor types)",
        task="multiclass", X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte,
        feature_names=names, class_names=class_names,
    )


# human-readable tumor-type names for the pan-cancer codes
TCGA_FULL_NAMES = {
    "BRCA": "Breast",
    "KIRC": "Kidney (clear cell)",
    "LUAD": "Lung (adenocarcinoma)",
    "PRAD": "Prostate",
    "COAD": "Colon",
}

DATASETS = {
    "breast_cancer": ("Breast cancer (binary)", load_breast_cancer_ds),
    "pan_cancer_rnaseq": ("Pan-cancer RNA-seq (5 types)", load_pancancer_ds),
}


def load_dataset(key: str, n_features: int, seed: int = 42) -> CancerDataset:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; choose from {list(DATASETS)}")
    return DATASETS[key][1](n_features=n_features, seed=seed)
