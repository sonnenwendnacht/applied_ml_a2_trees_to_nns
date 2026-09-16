"""2026-09-16 AI-assisted maintenance: paired, fixed-sample repeated comparison.

The sample is selected once with its own seed. Each run seed changes the split,
training-only CV and model randomness together. Models share that run's rows.
Overlapping repeated holdouts are not independent datasets or significance tests.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from compare import prepare_features, run_comparison, synthetic_data

ROOT = Path(__file__).resolve().parent


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seeds must be integers in [0, 2**32)")
    return seed


def validate_seeds(seeds):
    seeds = list(seeds)
    if not seeds:
        raise ValueError("at least one run seed is required")
    for seed in seeds:
        validate_seed(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("run seeds must be distinct")
    return seeds


def frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def select_fixed_sample(frame, rows, sample_seed):
    """Select only once; positions refer to source row order, not external IDs."""
    validate_seed(sample_seed)
    if type(rows) is not int or rows < 0 or 0 < rows < 200:
        raise ValueError("rows must be 0 or at least 200")
    _, target = prepare_features(frame)
    positions = np.arange(len(frame), dtype=np.int64)
    if rows > len(frame):
        raise ValueError("requested sample exceeds available source rows")
    if 0 < rows < len(frame):
        positions, _ = train_test_split(
            positions, train_size=rows, stratify=target, random_state=sample_seed
        )
    sample = frame.iloc[positions].reset_index(drop=True).copy()
    _, sample_target = prepare_features(sample)
    if len(sample) < 200 or sample_target.value_counts().min() < 20:
        raise ValueError("sample needs at least 200 rows and 20 examples of each class")
    metadata = {
        "sample_seed": sample_seed,
        "source_rows": len(frame),
        "requested_rows": rows,
        "selected_rows": len(sample),
        "sample_sha256": frame_hash(sample),
        "source_row_positions_sha256": hashlib.sha256(
            positions.astype("<i8").tobytes()
        ).hexdigest(),
        "method": "One stratified sample before all run seeds; source order retained when using all rows.",
    }
    return sample, metadata


def mean_and_sample_sd(values):
    values = list(values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("summary values must be nonempty and finite")
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values)
        if len(values) > 1
        else None,
    }


def summarize_runs(runs):
    if not runs:
        raise ValueError("cannot summarize zero runs")
    validate_seeds(row["seed"] for row in runs)
    models_by_run = []
    for row in runs:
        models = {result["model"]: result for result in row["results"]}
        if len(row["results"]) != 2 or set(models) != {"xgboost", "mlp"}:
            raise ValueError("each paired run needs exactly one XGBoost and one MLP")
        models_by_run.append(models)
    model_summary = {}
    for model in ("xgboost", "mlp"):
        model_summary[model] = {
            "training_cv_average_precision": mean_and_sample_sd(
                pair[model]["cv_average_precision"] for pair in models_by_run
            ),
            "total_convergence_warnings": sum(
                pair[model]["convergence_warnings"] for pair in models_by_run
            ),
        }
        for split in ("validation", "test"):
            model_summary[model][split] = {
                metric: mean_and_sample_sd(
                    pair[model][split][metric] for pair in models_by_run
                )
                for metric in ("average_precision", "f1")
            }
    paired = {}
    for split in ("validation", "test"):
        differences = [
            pair["mlp"][split]["average_precision"]
            - pair["xgboost"][split]["average_precision"]
            for pair in models_by_run
        ]
        paired[split] = {
            "direction": "MLP AP minus XGBoost AP on the same run's holdout",
            "by_seed": [
                {"seed": row["seed"], "difference": difference}
                for row, difference in zip(runs, differences)
            ],
            **mean_and_sample_sd(differences),
        }
    return {
        "run_count": len(runs),
        "models": model_summary,
        "paired_average_precision_difference": paired,
        "interpretation": "Descriptive repeated-holdout variability, not a confidence interval or significance test. Splits overlap; one run seed changes split, CV and model randomness together.",
    }


def run_repeated_comparison(
    sample,
    seeds=(17, 42, 2026),
    include_duration=False,
    mlp_max_iter=1500,
    on_result=None,
):
    seeds = validate_seeds(seeds)
    if type(mlp_max_iter) is not int or mlp_max_iter < 1:
        raise ValueError("mlp_max_iter must be a positive integer")
    sample_sha = frame_hash(sample)
    results = []
    for seed in seeds:
        result = run_comparison(
            sample,
            seed=seed,
            include_duration=include_duration,
            mlp_max_iter=mlp_max_iter,
        )
        if result["sample_sha256"] != sample_sha or frame_hash(sample) != sample_sha:
            raise ValueError("the fixed sample changed across paired runs")
        results.append(result)
        if on_result is not None:
            on_result(result)
    return {"runs": results, "summary": summarize_runs(results)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        help="local semicolon-delimited bank-full.csv; default is offline synthetic data",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2000,
        help="fixed sample size, or 0 for all real-data rows",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
    parser.add_argument("--mlp-max-iter", type=int, default=1500)
    parser.add_argument(
        "--include-duration", action="store_true", help="retrospective comparison only"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new output directory; existing leaf directories are refused",
    )
    args = parser.parse_args(argv)
    try:
        seeds = validate_seeds(args.seeds)
        validate_seed(args.sample_seed)
        if type(args.mlp_max_iter) is not int or args.mlp_max_iter < 1:
            raise ValueError("mlp-max-iter must be positive")
        if args.rows < 0 or 0 < args.rows < 200:
            raise ValueError("rows must be 0 or at least 200")
        if args.output_dir.exists():
            raise ValueError("output directory already exists; choose a new directory")
        if args.csv:
            raw = args.csv.read_bytes()
            frame = pd.read_csv(io.BytesIO(raw), sep=";", na_values=["?"])
            dataset = {
                "kind": "bank_csv",
                "filename": args.csv.name,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "full_rows": len(frame),
            }
        else:
            if args.rows == 0:
                raise ValueError("rows 0 requires a real CSV")
            frame = synthetic_data(args.rows, args.sample_seed)
            dataset = {
                "kind": "synthetic_smoke_data",
                "generator": "sklearn.make_classification",
                "generation_seed": args.sample_seed,
            }
        sample, sampling = select_fixed_sample(frame, args.rows, args.sample_seed)
        source_names = ["repeated_comparison.py", "compare.py", "requirements.txt"]
        if (ROOT / "requirements-reproducible.txt").is_file():
            source_names.append("requirements-reproducible.txt")
        metadata = {
            "schema_version": 1,
            "maintenance_date": "2026-09-16",
            "attribution": "New AI-assisted robustness evaluation; not original coursework results.",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "sampling": sampling,
            "run_seeds": seeds,
            "mlp_max_iter": args.mlp_max_iter,
            "include_duration": args.include_duration,
            "decision_threshold": 0.5,
            "experiment_design": "One fixed sample; paired models on identical per-seed 70/15/15 splits; 3-fold training-only CV selects hyperparameters by AP; no test-informed budget, threshold or architecture selection.",
            "limitations": [
                "Random stratified evaluation is not chronological or customer-grouped generalization.",
                "Repeated holdouts overlap and are not independent datasets; no significance claim.",
                "Higher MLP iteration budgets can increase overfitting; absence of cap warnings is not proof of global convergence.",
                "Selection and fixed 0.5 threshold match the maintained baseline; no threshold tuning.",
                "Exact reproducibility is scoped to unchanged source, data, configuration and software/hardware; timestamps and fit times naturally vary.",
            ],
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in source_names
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "numpy",
                        "pandas",
                        "scikit-learn",
                        "xgboost",
                        "threadpoolctl",
                    )
                },
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
        with (args.output_dir / "seed-results.jsonl").open(
            "x", encoding="utf-8"
        ) as handle:

            def save_progress(result):
                handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                values = {
                    row["model"]: row["test"]["average_precision"]
                    for row in result["results"]
                }
                print(
                    f"seed={result['seed']} test_AP: XGBoost={values['xgboost']:.6f} MLP={values['mlp']:.6f}",
                    flush=True,
                )

            report = run_repeated_comparison(
                sample, seeds, args.include_duration, args.mlp_max_iter, save_progress
            )
        with (args.output_dir / "results.json").open("x", encoding="utf-8") as handle:
            json.dump({**metadata, **report}, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
