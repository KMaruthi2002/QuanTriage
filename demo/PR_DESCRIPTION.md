# Pull request description (ready to paste into the PennyLaneAI/qml PR)

> Copy everything below the line into the GitHub PR body when you open it against
> `PennyLaneAI/qml`. Delete this header.

---

**Title:** Community demo: A clinically-aware quantum classifier (cancer triage that knows when to defer)

### What this PR adds

A new **community demo**, `tutorial_quantum_cancer_triage`, that builds a variational quantum
classifier for breast-cancer diagnosis and goes beyond the usual "encode → measure → accuracy"
pattern by wrapping it in four ideas that make a quantum classifier trustworthy in a real-world
setting:

1. **Selective prediction / abstention** — the model reports a confidence and *defers to a human*
   on low-confidence cases, traced via a risk–coverage curve.
2. **Cost-sensitive thresholding** — the decision threshold is tuned so missed malignancies
   (false negatives) are minimized, the metric that matters clinically.
3. **Robustness under quantum noise** — the trained model is re-evaluated on `default.mixed`
   with a depolarizing channel, sweeping the noise strength.
4. **Interpretability** — permutation importance over the real, named clinical features.

It also includes an **honest benchmark** against a classical logistic-regression baseline, and is
explicit that on this small tabular problem the quantum model is *competitive, not superior* —
the value is in the methodology.

### Files

- `demonstrations/tutorial_quantum_cancer_triage.py`
- `demonstrations/tutorial_quantum_cancer_triage.metadata.json`
- `_static/demonstration_assets/quantum_cancer_triage/thumbnail.png`

### PennyLane features used

`qml.AngleEmbedding`, `qml.StronglyEntanglingLayers`, `qml.AdamOptimizer`, `qml.draw`,
`default.qubit`, and the noisy `default.mixed` device with `qml.DepolarizingChannel`.

### Runtime

Runs end-to-end in ~30 seconds on a laptop (well under the demos CI limit). Uses only
`pennylane`, `scikit-learn`, `numpy`, and `matplotlib`.

### Checklist

- [ ] Demo file name begins with `tutorial_` and matches the metadata file name
- [ ] `.metadata.json` validates and uses allowed `categories` (`QML`, `Community`)
- [ ] Thumbnail added under `_static/demonstration_assets/quantum_cancer_triage/`
- [ ] Demo runs locally without errors
- [ ] Author set to my PennyLane profile username (for credit)

### Notes for reviewers

This is my first contribution — happy to adjust the narrative, categories, references, or the
scope of the four "trust" sections based on your feedback.
