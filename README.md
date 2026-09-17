# XGBoost vs. MLP on tabular data

[![Model checks](https://github.com/sonnenwendnacht/applied_ml_a2_trees_to_nns/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/sonnenwendnacht/applied_ml_a2_trees_to_nns/actions/workflows/checks.yml)

A reproducible comparison of gradient-boosted trees and a multilayer perceptron
for an imbalanced classification problem: predicting term-deposit subscription
from the UCI Bank Marketing dataset.

The repository preserves Junzhe Zong's original applied-machine-learning
coursework and adds a tested, AI-assisted command-line experiment from September
2026. The maintained pipeline
fits preprocessing **inside each cross-validation fold**, excludes post-call
`duration` by default, and records the data, split, code, and environment used
for each result.

A September 2026 follow-up repeats paired comparisons on a fixed sample and
checks sensitivity to the MLP iteration budget. It publishes all three seeds
at both budgets, including the larger budget's worse held-out AP.

[Run the comparison](#run-it) · [Paired results](#repeated-seed-training-budget-study) ·
[Original coursework](#original-coursework-and-new-implementation) · [Validation](VALIDATION.md)

Main finding on the fixed 5,000-row sample: increasing the MLP iteration cap
within the same CV-selection procedure reduced warnings but lowered mean
held-out average precision. Selected architectures can differ between caps;
this compares training-and-selection procedures, not longer training of one
identical selected model.

## Run it

From the repository root, use Python 3.12 or newer in an isolated environment:

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
Use `--rows 0` with real data for the full dataset; a positive row count must
not exceed the available rows. The runner parses and hashes the same CSV byte
snapshot, so the recorded hash describes the data actually read. Output files
are never overwritten; choose a new name for each run. The raw dataset and
trained models are not committed.

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

## Repeated-seed training-budget study

One fixed **5,000-row sample** (sample seed 42), with `duration` excluded, is used
for run seeds **17, 42, 2026**. Each seed changes the 3,500/750/750 split, CV
shuffle, and model initialization together. Both models share that seed's
splits and CV folds. MLP caps of **150 and 1,500 iterations** were fixed before
these held-out results were inspected; architecture selection remains
training-only CV. This is a budget sensitivity study, not test-set tuning.

| Model / MLP iteration cap | Test AP, mean ± sample SD | Test F1 at 0.5, mean ± sample SD | MLP cap warnings across 21 fits |
| --- | ---: | ---: | ---: |
| XGBoost (identical at both budgets) | 0.4124 ± 0.0429 | 0.2826 ± 0.0787 | — |
| MLP / 150 | 0.3629 ± 0.0337 | 0.3203 ± 0.0516 | 21 |
| MLP / 1,500 | 0.2931 ± 0.0387 | 0.3092 ± 0.0488 | 1 |

The paired test-AP difference (MLP minus XGBoost) is −0.0494 ± 0.0127 at 150
and −0.1192 ± 0.0148 at 1,500. Increasing the cap within the CV-selection procedure
reduced warnings but lowered mean test AP here; selected architectures can
differ. All three final refits at the larger cap stopped
early under the training-loss rule (872, 352, and 1,074 iterations); one inner
CV fit still reached the cap. Neither stopping nor fewer warnings proves a
global optimum. `early_stopping=False`: no internal accuracy-based validation
criterion was introduced.

These are three **overlapping holdouts**, not independent datasets, confidence
intervals, or a significance test. AP and F1 measure different things: the MLP
has lower AP here despite higher mean F1 at the fixed threshold. The prior
2,000-row experiment below is a different sample and cannot isolate the effect
of changing a training budget.

Records: [150 iterations](results/budget-study-2026-09-16/budget-150.json),
[1,500 iterations](results/budget-study-2026-09-16/budget-1500.json),
[exact-repeat checks](results/budget-study-2026-09-16/reproduction.json).
Each complete three-seed/budget run was repeated in a fresh process: all
non-clock fields, including metrics, CV candidates, selected parameters,
refit diagnostics, and prediction hashes, matched exactly on the same machine.

To reproduce, install `requirements-reproducible.txt` instead of the broad
runtime requirements (recorded Python: 3.14.0), cache the official CSV with the
single-run command above, then run:

```bash
python repeated_comparison.py --csv data/bank-full.csv --rows 5000 --sample-seed 42 --seeds 17 42 2026 --mlp-max-iter 150 --output-dir runs/budget-150
python repeated_comparison.py --csv data/bank-full.csv --rows 5000 --sample-seed 42 --seeds 17 42 2026 --mlp-max-iter 1500 --output-dir runs/budget-1500
```

Output directories must be new. Metadata is saved before fitting, each seed is
flushed to `seed-results.jsonl`, and `results.json` is written on completion.
Only aggregate results and hashes are published, not source rows or predictions.
The original single-run CLI still defaults to 150; `--mlp-max-iter` makes the
cap explicit. The repeated runner defaults to 1,500; use the explicit commands
above to reproduce both prespecified budgets.

## Earlier single-seed result (preserved)

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
| [repeated_comparison.py](repeated_comparison.py) | Fixed-sample repeated holdouts, paired summaries, incremental provenance |
| [tests/](tests/) | Tests of split/preprocessing boundaries, metrics, diagnostics, provenance, repeatability, and failure-safe outputs |
| [VALIDATION.md](VALIDATION.md) | What was tested, provenance, and limitations |

The new runners, tests, and budget study are explicitly AI-assisted portfolio
maintenance, not original submission work. The notebook's historical scores
include `duration` and use preprocessing fitted before inner cross-validation.
They are **not directly comparable** with the
new smaller experiment. Keeping the notebook unchanged preserves the original
work without silently rewriting its results. Optional notebook dependencies
are listed in `requirements-notebook.txt`; the full original sweeps were not
rerun during this cleanup.

## Limitations and next experiments

This is an educational comparison, not a deployed decision system. It uses
random stratified holdouts rather than chronological or customer-grouped
evaluation, one fixed subsample, three overlapping repeated holdouts, a limited
search, and an untuned decision threshold. Future experiments could evaluate
temporal generalization and regularization, and choose thresholds using
validation data rather than the test set. The published holdouts have now been
inspected; further model development needs a new evaluation plan, not repeated
tuning against these results.

## Data attribution

S. Moro, P. Rita, and P. Cortez, *Bank Marketing*, UCI Machine Learning
Repository, [DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306).
UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
That dataset license is not a license grant for the repository's coursework.
