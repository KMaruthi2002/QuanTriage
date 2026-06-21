# PennyLane demo submission

This folder contains a community demo ready to submit to PennyLane, built from the QuanTriage
project in the repo root.

| file | purpose |
|---|---|
| `tutorial_quantum_cancer_triage.py` | the demo, a self-contained, executable sphinx-gallery tutorial (runs in ~30s) |
| `tutorial_quantum_cancer_triage.metadata.json` | demo metadata (title, authors, categories, references) |

The demo is **self-contained** (it does not import from `../src`), as PennyLane demos must be a
single runnable script with execution time under ~10 minutes.

## How to submit (≈15 minutes)

PennyLane demos live in the [PennyLaneAI/qml](https://github.com/PennyLaneAI/qml) repository (also
mirrored at [PennyLaneAI/demos](https://github.com/PennyLaneAI/demos)). The flow:

1. **Fork** the `PennyLaneAI/qml` repo on GitHub and clone your fork.
2. **Copy both files** into the repo's `demonstrations/` directory (keep the `tutorial_` prefix
   and matching names).
3. **Add a thumbnail image** referenced by the metadata's `previewImages` / the `og:image` meta
   tag, under `_static/demonstration_assets/quantum_cancer_triage/thumbnail.png`. You can reuse a
   plot from `../assets/` (e.g. the selective-prediction or ROC figure) as a starting point.
4. **Run it locally** to confirm it executes cleanly:
   ```bash
   python demonstrations/tutorial_quantum_cancer_triage.py
   ```
5. **Open a pull request** against `master`. In the PR description, summarize the demo and note
   it is a community contribution.
6. The Xanadu team reviews demos; expect feedback and a round or two of revisions before merge.

## Get credited on your PennyLane profile

The metadata currently lists the author by **name**. If you create a profile on
[pennylane.ai](https://pennylane.ai/), replace the author entry with your profile handle so the
demo shows up as your contribution:

```json
"authors": [{ "username": "your_pennylane_handle" }]
```

## Notes

- Confirm `categories` against the current allowed list in the qml repo's
  `demonstrations_metadata.md` (we used `["QML", "Community"]`).
- Keep the runtime under the demos CI limit (~10 min), this one is well within it.
- The narrative uses reStructuredText in `#`-comment blocks separated by lines of `#`; that is the
  sphinx-gallery format PennyLane renders into the published page.
