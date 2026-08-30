# XRD-TER: Template Element Replacement Augmentation for XRD Phase Classification

Pipeline for classifying the B-site cation of inorganic crystal structures
from simulated powder XRD patterns, using **Template Element Replacement
(TER)** — a physics-informed data augmentation strategy that generates
virtual structures by substituting isovalent, similar-radius cations into
real structures queried from the Materials Project.

Three crystal families are currently supported: **spinel** (AB2O4),
**garnet** (A3B2C3O12), and **scheelite** (ABO4).

## Repository structure

```
data/
  crystal_families.py   # per-family config: space group, MP query, label
                         # elements, ionic radii, TER replacement rules
  loader.py              # query Materials Project, simulate XRD, apply TER,
                         # cache to disk, load + filter for training
  augmentation.py        # signal-level augmentation (noise, broadening,
                         # peak shift, background, intensity scaling)
  dataset.py             # PyTorch Dataset wrapper (optional GPU preload)

models/
  base.py                 # abstract BaseModel interface
  cnn1d.py                 # 1D CNN baseline
  conformer.py            # CNN + local/global attention hybrid (used in the paper)
  spectral_transformer.py # patch-based Transformer (ViT-style) baseline

training/
  trainer.py               # single-model training loop with early stopping
  ensemble.py              # Deep Ensemble training + predictive entropy

evaluation/
  metrics.py                # accuracy, macro-F1, per-class F1, entropy stats
  cross_validation.py      # 5-fold CV, Experiment A (real only) vs.
                             # Experiment B (real + TER), leakage-safe split

visualization/
  figures.py                # accuracy/F1 comparison, entropy analysis figures
  attention.py               # attention-map overlays for Conformer /
                             # SpectralTransformer
```

`config.py` (root) holds all hyperparameters and paths — set `ACTIVE_FAMILY`
and `MODEL_NAME` there. `main.py` (root) is the CLI entry point, and
`models/__init__.py` is the model factory (`get_model`).

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ (uses PEP 585 generics, e.g. `tuple[np.ndarray, ...]`).

A [Materials Project](https://next-gen.materialsproject.org/api) API key is
required to build a dataset from scratch. **Never hardcode the key.** Set it
as an environment variable and read it in `config.py` or at the call site:

```bash
export MP_API_KEY="your-key-here"
```

`main.py` reads `MP_API_KEY` automatically as a fallback if `--key` isn't
passed, so the key never needs to appear on the command line or in shell
history.

## Usage

```bash
# uses ACTIVE_FAMILY from config.py (default: spinel)
python main.py

# override the family (spinel | garnet | scheelite)
python main.py --family garnet

# first run for a family needs an API key (env var or --key)
python main.py --family scheelite --key YOUR_API_KEY

# build and cache the dataset only, skip training
python main.py --build-only --family garnet
```

Each run trains a 5-fold Deep Ensemble CV (Experiment A: real only vs.
Experiment B: real + TER), and writes `summary.json`, `per_class_f1.json`,
`entropy_records.json`, ensemble checkpoints, and the accuracy/entropy/
attention figures to `output/<family>/results/`.

To call the pipeline programmatically instead of via the CLI:

```python
import os
from data.loader import load_dataset
from evaluation.cross_validation import run_cv
from visualization.figures import plot_accuracy, plot_entropy
from visualization.attention import plot_attention

data = load_dataset(api_key=os.environ.get("MP_API_KEY"))
results = run_cv(data)

plot_accuracy(results, data)
plot_entropy(results, data)
plot_attention(results["last_models"], data)
```

## Key methodological points

- **TER augmentation**: for each real structure, isovalent/similar-radius
  cations are substituted per family-specific `replacement_map` in
  `crystal_families.py`, and the substituted structure's XRD pattern is
  simulated with `pymatgen`'s `XRDCalculator`.
- **Leakage-safe cross-validation**: virtual (TER-generated) structures are
  only included in a training fold if their real "origin" structure is also
  in that fold's training split (`evaluation/cross_validation.py`). This
  fixes an earlier data-leakage bug where virtual structures derived from
  test-set structures inflated reported accuracy.
- **Deep Ensemble** (`training/ensemble.py`) provides predictive entropy as
  an uncertainty estimate, used in `visualization/figures.py` to relate
  uncertainty to B-site ionic radius.

## Adding a new crystal family

Add one block to `CRYSTAL_FAMILIES` in `data/crystal_families.py`: space
group, Materials Project query filters, candidate label elements, ionic
radii, and a TER `replacement_map`. No other file needs to change.

## Citation

<!-- Add citation once accepted / assigned a DOI. -->

## License

<!-- Add a LICENSE file (e.g. MIT) before publishing. -->
