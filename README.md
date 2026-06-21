# QuanTriage — A Clinically-Aware Quantum Cancer Triage Classifier

A hybrid **quantum machine learning** project built with [PennyLane](https://pennylane.ai/).
It trains a **variational quantum classifier** to distinguish malignant from benign breast
tumors — and, unlike most QML demos that stop at a single accuracy number, it behaves like a
tool you'd actually want near a clinic:

1. **It knows when it doesn't know.** Low-confidence cases are *referred to a doctor* instead
   of guessed (selective prediction / abstention).
2. **It's tuned to not miss cancer.** The decision threshold is chosen so that missing a
   malignancy is treated as far costlier than a false alarm (cost-sensitive thresholding).
3. **It's stress-tested on noisy "hardware."** The trained model is re-evaluated on a noisy
   quantum simulator that mimics real-device decoherence.
4. **It explains itself.** Permutation importance shows which cell-nucleus measurements drive
   the predictions.

Everything runs on a **simulator** — no quantum hardware required — and a full run finishes in
about **30 seconds** on a laptop.

---

## What is a variational quantum classifier?

```
 features ──▶ AngleEmbedding ──▶ StronglyEntanglingLayers ──▶  ⟨Z₀⟩  ──▶  p(malignant)
              (encode the data    (trainable rotation           (measure     = (1 − ⟨Z⟩)/2
               into qubit angles)  angles = the "weights")        qubit 0)
```

Classical data is *encoded* into a quantum state, an *entangling* circuit of trainable
rotations transforms it, and a single qubit measurement is read out as a probability. Training
adjusts the rotation angles by gradient descent (binary cross-entropy) — the same idea as
training a tiny neural network, but the hidden layer is a set of entangled qubits.

The model is wrapped as a scikit-learn estimator, so it plugs directly into standard tooling
(`permutation_importance`, scoring, etc.) and into an honest side-by-side comparison with
classical baselines.

---

## Quickstart

```bash
cd quantum-ml-pennylane
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/main.py
```

> Use **Python 3.12** — PennyLane does not yet ship wheels for 3.14.

Tune the model from the CLI:

```bash
.venv/bin/python src/main.py --qubits 8 --layers 6 --epochs 60 --cost-ratio 10
```

| flag | meaning | default |
|---|---|---|
| `--qubits` | qubits = number of (named) features used | 6 |
| `--layers` | variational layers (model capacity) | 4 |
| `--epochs` | training epochs | 40 |
| `--cost-ratio` | how many times worse a missed cancer is than a false alarm | 10 |
| `--confidence` | abstain below this confidence | 0.85 |
| `--noise-levels` | depolarizing-noise sweep | 0 0.01 0.03 0.05 0.1 |

### Interactive dashboard (GUI)

A [Streamlit](https://streamlit.io/) dashboard wraps the model in an interactive UI:

```bash
.venv/bin/streamlit run app.py
```

- **Predict a patient** — move sliders for real clinical measurements (or load an example
  malignant/benign patient) and get a live verdict: malignant, benign, or *refer to a doctor*.
- **Model performance** — quantum vs. classical metrics, confusion matrix, ROC, training curve.
- **Trust & robustness** — selective-prediction stats, noise-robustness and feature-importance charts.
- The sidebar's *cost-of-a-missed-cancer* and *confidence* sliders update everything live, so you
  can watch the sensitivity-vs-false-alarm tradeoff move in real time. The quantum circuit trains
  once (cached); interactions after that are instant.

---

## Example results (default settings, seed 42)

**Quantum vs. classical** on the held-out test set:

| model | accuracy | sensitivity | specificity | AUC |
|---|---|---|---|---|
| quantum (thr 0.50) | 0.930 | 0.849 | 0.978 | 0.991 |
| **quantum (cost-tuned, thr 0.27)** | 0.909 | **1.000** | 0.856 | 0.991 |
| logistic regression | 0.965 | 0.925 | 0.989 | 0.997 |
| SVM (RBF) | 0.958 | 0.906 | 0.989 | 0.997 |

After cost-sensitive tuning the quantum model **catches every malignancy in the test set
(0 missed cancers)**, trading a handful of false alarms for safety — the right call for a
screening tool. Classical baselines still win slightly on raw accuracy, which the project
reports honestly: quantum ML is competitive here, not yet superior.

**Selective prediction:** at a 0.85 confidence cutoff the model decides ~59% of cases with
**100% accuracy on the cases it keeps**, and refers the rest to a human.

**Noise robustness:** accuracy degrades gracefully (≈0.92 → 0.74) as depolarizing noise
increases — useful intuition for what real hardware would cost you.

Your exact numbers will vary with the flags and seed.

---

## Results gallery

| | |
|---|---|
| **Cost-tuned to catch every cancer** | **Refers uncertain cases to a doctor** |
| ![confusion matrix](assets/confusion_matrix.png) | ![selective prediction](assets/selective_prediction.png) |
| **Quantum vs. classical (ROC)** | **Survives quantum noise** |
| ![roc](assets/roc_curve.png) | ![noise robustness](assets/noise_robustness.png) |
| **Explains which measurements matter** | **Trains in seconds** |
| ![feature importance](assets/feature_importance.png) | ![training curve](assets/training_curve.png) |

---

## Reading the output

The run prints a full console report and writes artifacts to `results/`:

| file | what it shows |
|---|---|
| `training_curve.png` | loss ↓ and train accuracy ↑ over epochs |
| `roc_curve.png` | ROC for quantum vs. each classical baseline |
| `confusion_matrix.png` | quantum confusion matrix at the cost-tuned threshold |
| `selective_prediction.png` | risk–coverage curve: accuracy rises as the model abstains more |
| `noise_robustness.png` | accuracy & sensitivity vs. quantum-noise strength |
| `feature_importance.png` | which measurements the model relies on |
| `metrics.json` | every number above, machine-readable |

---

## Project layout

```
src/
├── data.py           load breast-cancer data, select top-k NAMED features, scale, split
├── quantum_model.py  the variational circuit, sklearn-style estimator, + noisy variant
├── baselines.py      classical logistic regression + RBF SVM on the same features
├── evaluation.py     metrics, cost threshold, selective prediction, noise, importance, plots
└── main.py           orchestrates the pipeline and prints the report
```

**Data note:** scikit-learn encodes the target as malignant=0/benign=1; we relabel so
**malignant = 1 (positive class)**, which makes "sensitivity/recall" mean "fraction of real
cancers caught." We use feature *selection* (not PCA) so the inputs stay human-readable named
measurements, keeping the interpretability step clinically meaningful.

---

## Extending it

- **Bigger model:** raise `--qubits` / `--layers` (cost scales with simulator size).
- **Different encoding:** swap `AngleEmbedding` for `AmplitudeEmbedding` or a *data re-uploading*
  ansatz for more expressivity.
- **Real quantum hardware:** PennyLane has plugins for IBM Quantum (`pennylane-qiskit`) and
  AWS Braket (`pennylane-braket`) — point the device at real backends instead of `default.qubit`.
- **PyTorch hybrid:** wrap the circuit in `qml.qnn.TorchLayer` to stack it with classical layers.

---

## Disclaimer

This is an educational research project, **not a medical device**. It must not be used for
real diagnosis or clinical decisions.
