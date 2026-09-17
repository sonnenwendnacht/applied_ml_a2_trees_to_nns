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

The original six unit tests check:

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
runs the tests, and executes single-run and repeated-seed 240-row offline
comparisons on pushes and pull requests. Consult the workflow status for the
hosted run result.

The September 16 continuation adds tests for positive iteration budgets,
strict JSON parameter sentinels, probability-hash ordering, real-fit stopping
diagnostics, fixed sampling, paired means/sample SD, exact repeated execution,
invalid configurations, source-byte provenance, output exclusivity and
incremental records that survive a later failed seed. All 20 tests pass
on Python 3.12.3 and 3.14.0 with `requirements-reproducible.txt`. Ruff checks
cover the maintained scripts and tests, not the preserved notebook.

### CSV input-integrity refinement

Later September 16 AI-assisted maintenance tightens only the single-run CLI's
input handling. A positive `--rows` request larger than its CSV now fails before
fitting or writing output instead of silently using fewer rows. Parsing and
SHA-256 hashing use one byte snapshot rather than reading the path twice, so a
source change between parsing and hashing cannot misidentify the parsed data.
This does not promise a filesystem-atomic snapshot if another process modifies
the file during the read itself.

Three new tests bring the suite to 23: oversized requests are rejected before
fitting, a simulated post-parse file change cannot change the recorded hash,
and zero/exact/smaller sample sizes remain supported. Both defect regressions
failed against the preceding release; the valid-size controls already passed.
All 23 tests pass on Python 3.12.3 and 3.14.0. Training, preprocessing, selection,
the repeated-run workflow and the existing result records are unchanged.
A before/after Python 3.12 CLI run on the same 2,000-row UCI sample (seed 42,
150-iteration cap) matches every non-clock/non-source-hash report field exactly,
including prediction hashes, selected parameters, metrics and data/split hashes.
The source CSV and published result files were not modified.

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

## September 16 repeated budget study

`results/budget-study-2026-09-16/` contains a new experiment; the two earlier
result files above are unchanged. Commands and rounded scores are in the
README. One stratified 5,000-row sample is chosen once (sample seed 42), and
run seeds 17/42/2026 each create 3,500/750/750 splits. All seeds and both MLP
caps (150/1,500) were specified before looking at this study's held-out scores.
The cap is the only configured treatment difference; training-only CV can
select a different architecture at each cap. Hence this measures the full
budget-plus-CV-selection procedure, not only additional optimization steps
for an identical selected architecture.

Each complete budget run was executed twice in separate processes. Exact
recursive comparison excludes only `created_at_utc` and
`search_and_refit_seconds`: every other field matches. Both budgets share
sample and per-seed split hashes, and all XGBoost non-clock fields are identical
across budgets. The published [reproduction record](results/budget-study-2026-09-16/reproduction.json)
identifies the source reports and scope of these checks. Exact equality is a
same-machine, same-environment observation, not a promise across platforms.

Additive schema-2 single-run records include:

- Full selected estimator parameters. XGBoost's NaN `missing` sentinel is
  explicitly encoded as `{"nonfinite_parameter": "nan"}` in parameter metadata
  only; nonfinite metrics are not silently converted to null.
- Per-candidate mean, population SD and individual fold AP from GridSearchCV;
  repeated-run summaries separately use **sample** SD across the three seeds.
- Final MLP refit iteration count, loss, cap and observed stopping condition.
  Warning totals cover six inner-CV fits plus the refit; they do not give
  individual CV-fit iteration counts.
- SHA-256 of ordered validation/test probabilities encoded as little-endian
  float64 bytes, without publishing individual predictions.

The repeated report adds exact raw CSV/source hashes, fixed-sample and source
row-position hashes, dependencies, paired AP differences and limitations.
Pinned requirements capture the direct numerical stack; they are not a full
OS/transitive-dependency lock. A default-150 rerun of the earlier 2,000-row
experiment before/after the diagnostic changes matched every pre-existing
non-clock metric, selected parameter, sample and split field exactly.

The larger cap reduces 21 iteration-limit warnings to one across three
seven-fit searches but lowers mean test AP from 0.3629 to 0.2931. This is not
claimed as an accuracy improvement. Lower training loss alongside worse
held-out scores is consistent with overfitting; this small comparison does not
establish a unique cause. There is no independence, confidence-interval or
significance claim for the overlapping repeated holdouts.

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
