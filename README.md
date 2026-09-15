# XGBoost vs. MLP on tabular data

[![Model checks](https://github.com/sonnenwendnacht/applied_ml_a2_trees_to_nns/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/sonnenwendnacht/applied_ml_a2_trees_to_nns/actions/workflows/checks.yml)

A reproducible comparison of gradient-boosted trees and a multilayer perceptron
for an imbalanced classification problem: predicting term-deposit subscription
from the UCI Bank Marketing dataset.

The repository preserves Junzhe Zong's original applied-machine-learning
coursework and adds a small, tested command-line experiment. The new pipeline
fits preprocessing **inside each cross-validation fold**, excludes post-call
`duration` by default, and records the data, split, code, and environment used
for each result.

## Run it

Use Python 3.12 or newer in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# Offline smoke test: generated mixed numeric/categorical data
python compare.py --rows 800 --output runs/synthetic.json

# Small real-data experiment; download from UCI on first use
python compare.py --download --rows 2000 --output runs/bank-sample.json
```

The download caches only `bank-full.csv` under ignored `data/`. To use an existing
semicolon-separated CSV, replace `--download` with `--csv /path/to/bank-full.csv`.
Use `--rows 0` with real data for the full dataset. Output files are never
overwritten; choose a new name for each run. The raw dataset and trained models
are not committed.

## Experiment design

- A seeded, stratified sample is split into 70% training, 15% validation, and
  15% test rows. Hyperparameters are selected by three-fold cross-validation
  on **training rows only**; neither holdout selects hyperparameters.
- Numeric columns receive median imputation and standardization. Categorical
  columns receive missing-value imputation and one-hot encoding that handles
  unseen categories. These steps are part of the estimator pipeline supplied
  to `GridSearchCV`, not fitted before cross-validation.
- The compact search compares two XGBoost depths and two MLP architectures.
  It is a reproducible baseline, not an exhaustive model search. CPU threads
  are limited to one for both models.
- Selection uses **average precision (AP)**. Accuracy, precision, recall, and
  F1 are also reported at a fixed probability threshold of 0.5. AP is not the
  same calculation as trapezoidal area under a precision-recall curve.
- `duration` is known only after a call finishes, so it is excluded by default.
  `--include-duration` is available only for retrospective comparisons; such
  scores should not be presented as pre-call prediction performance.

See scikit-learn's [guidance on preprocessing and data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
and the dataset's [feature documentation](https://archive.ics.uci.edu/dataset/222/bank+marketing).

## Recorded result

One run on a stratified **2,000-row sample**, seed 42, with `duration` excluded:

| Model | Training CV AP | Test AP | Test F1 at 0.5 | Search + refit, seconds |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.3490 | 0.4101 | 0.2727 | 0.38 |
| MLP | 0.3494 | 0.4116 | 0.3137 | 0.95 |

The test split has only 300 rows, including 35 positive examples. The AP values
are close; this run does **not** establish that either model is generally
better. All seven MLP fits reached the 150-iteration limit, so convergence is
not established. The low recall at the fixed threshold also matters more than
the superficially high accuracy on this imbalanced dataset.

[The machine-readable record](results/bank-sample.json) contains the exact
versions, hashes, selected parameters, validation/test metrics, and warnings.
Timings are one local measurement, not a hardware-independent benchmark.
An [offline smoke-test record](results/synthetic-smoke.json) is included
separately; its scores are not evidence about the real banking task.

## Original coursework and new implementation

| File | Purpose |
| --- | --- |
| [assignment2.ipynb](assignment2.ipynb) | Original notebook, outputs, and larger parameter sweeps, unchanged |
| [historical/](historical/) | Supplemental plots copied unchanged from the PC coursework folder |
| [compare.py](compare.py) | New standalone, fold-local preprocessing and model comparison |
| [tests/test_comparison.py](tests/test_comparison.py) | Six tests of split integrity, preprocessing boundaries, feature policy, and metrics |
| [VALIDATION.md](VALIDATION.md) | What was tested, provenance, and limitations |

The notebook's historical scores include `duration` and use preprocessing fitted
before inner cross-validation. They are **not directly comparable** with the
new smaller experiment. Keeping the notebook unchanged preserves the original
work without silently rewriting its results. Optional notebook dependencies
are listed in `requirements-notebook.txt`; the full original sweeps were not
rerun during this cleanup.

## Limitations and next experiments

This is an educational comparison, not a deployed decision system. It uses
random stratified holdouts rather than chronological or customer-grouped
evaluation, one seed, a small sample, a limited search, and an untuned decision
threshold. Next experiments should evaluate temporal generalization, repeat
seeds, give the MLP a sufficient convergence budget, and choose any threshold
using validation data rather than the test set.

## Data attribution

S. Moro, P. Rita, and P. Cortez, *Bank Marketing*, UCI Machine Learning
Repository, [DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306).
UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
That dataset license is not a license grant for the repository's coursework.
