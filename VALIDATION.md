# Validation and provenance

## Preserved source

- Original source: Junzhe Zong's `aml/project2` PC folder and existing public
  repository `sonnenwendnacht/applied_ml_a2_trees_to_nns`.
- Base commit: `0457314` ("Final submission: GBDT vs MLP implementation").
- `assignment2.ipynb` is unchanged from that commit. The five PNGs under
  `historical/` are byte-for-byte copies of previously untracked PC figures.
- The command-line experiment, tests, dependency declarations, documentation,
  result records, and CI workflow were added during portfolio preparation.
  They are not represented as part of the original assignment submission.
- Work was prepared in a separate clone; the original PC folder was not edited.

## Checks

```bash
python -m unittest discover -s tests -v
python compare.py --rows 800 --output runs/synthetic.json
python compare.py --download --rows 2000 --output runs/bank-sample.json
```

The six unit tests check:

1. Target and post-call duration are excluded from default predictors.
2. Train/validation/test indices are disjoint, complete, and repeatable.
3. Cross-validation fits a separate scaler on each training fold, not the
   complete input to the search.
4. Missing values and unseen held-out categories do not refit preprocessing.
5. Probability-ranking average precision is distinct from thresholded F1.
6. Invalid target labels fail explicitly.

Local validation passed on Python 3.14.0, scikit-learn 1.7.2, XGBoost 3.2.0,
pandas 2.3.3, and NumPy 2.4.2. Ruff lint and formatting checks also passed.
GitHub Actions installs the declared runtime dependencies on Python 3.12,
runs the tests, and executes a 240-row offline comparison on pushes and pull
requests. Consult the workflow status for the hosted run result.

## Recorded experiments

`results/synthetic-smoke.json` and `results/bank-sample.json` were generated
with the commands above, using the default seed 42. The synthetic example
checks execution, not real-world predictive quality. The real-data run uses
2,000 stratified rows sampled from UCI's 45,211-row `bank-full.csv`, followed by
a 1,400/300/300 train/validation/test split.

Each JSON file records:

- The SHA-256 of `compare.py`, the input sample, and each split's ordered row
  indices; real-data runs also record the raw CSV hash.
- Python and library versions, seed, feature list, split sizes, and class balance.
- Training-CV AP, selected parameters, search-and-refit time, holdout metrics,
  and warning counts.

The source hash ties the result to the script, independently of the later
documentation commit. Version ranges in `requirements.txt` support installation
but are not a frozen environment; changing library versions or hardware may
change metrics or timings. Both recorded runs report seven MLP convergence
warnings, covering six inner-CV fits and the final training-set refit.

## Interpreting the historical notebook

The saved notebook reports test AP of 0.6215 for XGBoost and 0.5896 for the MLP;
it labels these values "AUC-PR" but computes `average_precision_score`.
Those are historical saved outputs, not newly reproduced measurements.

The old experiment differs from the new implementation in several important
ways: it uses the full dataset, retains post-call duration, uses larger search
spaces, and fits preprocessing on the outer training set before the inner CV
search. The latter allows inner-validation rows to influence preprocessing;
it does not by itself establish that the outer test rows were used for fitting.
Do not attribute score differences between the two experiments to a single
methodology change, or compare their fit times as equivalent workloads.

No deployment accuracy, causal effect, statistically significant model ranking,
or full MLP convergence is claimed. The original notebook's full parameter
sweeps were not rerun. The dataset stays local and ignored; only code,
documentation, aggregate metrics, and historical figures are published.
