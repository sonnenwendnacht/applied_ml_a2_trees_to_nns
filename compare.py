"""Small, reproducible XGBoost versus MLP comparison on mixed tabular data."""

import argparse
import hashlib
import io
import json
import math
import platform
import time
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def synthetic_data(rows=800, seed=42):
    values, target = make_classification(
        n_samples=rows,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.88, 0.12],
        random_state=seed,
    )
    frame = pd.DataFrame(values, columns=[f"numeric_{i}" for i in range(6)])
    frame["contact"] = np.where(frame["numeric_0"] > 0, "telephone", "cellular")
    frame.loc[frame.index % 19 == 0, "contact"] = np.nan
    frame.loc[frame.index % 23 == 0, "numeric_1"] = np.nan
    # Deliberately outcome-dependent: the default feature policy must remove it.
    frame["duration"] = target * 100 + 1
    frame["y"] = np.where(target == 1, "yes", "no")
    return frame


def download_bank_data(destination):
    """Cache only the named CSV from the official archive; never extract paths."""
    destination = Path(destination)
    if destination.exists():
        return destination
    with urllib.request.urlopen(DATA_URL, timeout=45) as response:
        archive = response.read()
    with (
        zipfile.ZipFile(io.BytesIO(archive)) as outer,
        zipfile.ZipFile(io.BytesIO(outer.read("bank.zip"))) as inner,
    ):
        content = inner.read("bank-full.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(content)
    return destination


def prepare_features(frame, include_duration=False):
    if "y" not in frame:
        raise ValueError("CSV needs a 'y' target column containing 'yes' or 'no'")
    target = frame["y"].map({"yes": 1, "no": 0})
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("target must contain both classes and only 'yes'/'no' values")
    excluded = ["y"] + (["duration"] if not include_duration else [])
    features = frame.drop(columns=excluded, errors="ignore").replace("unknown", np.nan)
    if features.shape[1] == 0:
        raise ValueError("no predictor columns remain")
    return features, target.astype(int)


def build_pipeline(estimator):
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant", fill_value="unknown", keep_empty_features=True
                ),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ]
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def split_data(features, target, seed):
    train_val, test = train_test_split(
        np.arange(len(target)),
        test_size=0.15,
        stratify=target,
        random_state=seed,
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        stratify=target.iloc[train_val],
        random_state=seed,
    )
    return {"train": train, "validation": validation, "test": test}


def classification_metrics(target, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(target, probabilities)),
    }


def parameter_metadata(value):
    """Encode estimator parameters, including XGBoost's missing-value sentinel.

    Only parameter metadata uses this representation. Nonfinite metrics must
    still fail strict JSON serialization rather than silently becoming null.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_parameter": str(value)}
    if isinstance(value, dict):
        return {key: parameter_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [parameter_metadata(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported estimator parameter type: {type(value).__name__}")


def probability_sha256(probabilities):
    values = np.asarray(probabilities, dtype="<f8")
    if not np.isfinite(values).all():
        raise ValueError("predicted probabilities must be finite")
    return hashlib.sha256(values.tobytes()).hexdigest()


def run_comparison(frame, seed=42, include_duration=False, *, mlp_max_iter=150):
    if (
        isinstance(mlp_max_iter, bool)
        or not isinstance(mlp_max_iter, int)
        or mlp_max_iter < 1
    ):
        raise ValueError("mlp_max_iter must be a positive integer")
    features, target = prepare_features(frame, include_duration)
    if len(frame) < 200 or target.value_counts().min() < 20:
        raise ValueError("use at least 200 rows and at least 20 examples of each class")
    indices = split_data(features, target, seed)
    train = indices["train"]
    models = [
        (
            "xgboost",
            XGBClassifier(
                n_estimators=80,
                learning_rate=0.1,
                n_jobs=1,
                tree_method="hist",
                eval_metric="logloss",
                random_state=seed,
            ),
            {"model__max_depth": [3, 5]},
        ),
        (
            "mlp",
            MLPClassifier(
                max_iter=mlp_max_iter, early_stopping=False, random_state=seed
            ),
            {"model__hidden_layer_sizes": [(32,), (64, 32)]},
        ),
    ]
    reports = []
    with threadpool_limits(limits=1):
        for name, estimator, grid in models:
            search = GridSearchCV(
                build_pipeline(estimator),
                grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
                scoring="average_precision",
                n_jobs=1,
                error_score="raise",
                refit=True,
            )
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                search.fit(features.iloc[train], target.iloc[train])
            elapsed = time.perf_counter() - start
            fitted_model = search.best_estimator_.named_steps["model"]
            result = {
                "model": name,
                "best_params": search.best_params_,
                "cv_average_precision": float(search.best_score_),
                "search_and_refit_seconds": elapsed,
                "convergence_warnings": sum(
                    issubclass(w.category, ConvergenceWarning) for w in caught
                ),
                "other_warnings": sorted(
                    {
                        str(w.message)
                        for w in caught
                        if not issubclass(w.category, ConvergenceWarning)
                    }
                ),
                "selected_estimator_params": parameter_metadata(
                    fitted_model.get_params()
                ),
                "cv_candidates": [
                    {
                        "params": parameter_metadata(params),
                        "mean_average_precision": float(
                            search.cv_results_["mean_test_score"][candidate]
                        ),
                        "std_average_precision": float(
                            search.cv_results_["std_test_score"][candidate]
                        ),
                        "fold_average_precision": [
                            float(
                                search.cv_results_[f"split{fold}_test_score"][candidate]
                            )
                            for fold in range(3)
                        ],
                    }
                    for candidate, params in enumerate(search.cv_results_["params"])
                ],
                "probability_sha256": {},
            }
            if name == "mlp":
                result["refit_diagnostics"] = {
                    "n_iter": int(fitted_model.n_iter_),
                    "max_iter": mlp_max_iter,
                    "training_loss": float(fitted_model.loss_),
                    "stop_observation": (
                        "iteration_limit_reached"
                        if fitted_model.n_iter_ >= mlp_max_iter
                        else "stopped_before_limit_under_training_loss_rule"
                    ),
                    "note": "Final training-set refit only; stopping is not proof of global convergence. Warning count covers all six CV fits and the refit.",
                }
            for split in ("validation", "test"):
                rows = indices[split]
                probabilities = search.predict_proba(features.iloc[rows])[:, 1]
                result[split] = classification_metrics(target.iloc[rows], probabilities)
                result["probability_sha256"][split] = probability_sha256(probabilities)
            reports.append(result)
    frame_hash = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "rows": len(frame),
        "sample_sha256": frame_hash,
        "features": features.columns.tolist(),
        "include_duration": include_duration,
        "split_sizes": {name: len(rows) for name, rows in indices.items()},
        "split_sha256": {
            name: hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
            for name, rows in indices.items()
        },
        "positive_fraction": {
            name: float(target.iloc[rows].mean()) for name, rows in indices.items()
        },
        "decision_threshold": 0.5,
        "training_config": {
            "mlp_max_iter": mlp_max_iter,
            "mlp_early_stopping": False,
            "cv_folds": 3,
            "cv_shuffle": True,
            "cv_seed": seed,
            "worker_threads": 1,
        },
        "model_selection": "3-fold stratified CV on training rows, preprocessing fitted inside each fold; average precision scoring",
        "results": reports,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--csv", type=Path, help="local bank-full.csv (semicolon-separated)"
    )
    source.add_argument(
        "--download",
        action="store_true",
        help="download official UCI bank-full.csv to ignored data/",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=800,
        help="stratified sample size, or 0 for all real-data rows",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=150,
        help="positive MLP iteration cap (default: 150)",
    )
    parser.add_argument(
        "--include-duration",
        action="store_true",
        help="retrospective comparison only: include post-call duration",
    )
    parser.add_argument("--output", type=Path, help="write a new JSON result file")
    args = parser.parse_args(argv)
    if args.rows < 0 or 0 < args.rows < 200:
        parser.error("--rows must be 0 or at least 200")
    if args.mlp_max_iter < 1:
        parser.error("--mlp-max-iter must be positive")
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    path = args.csv
    try:
        if args.download:
            path = download_bank_data(
                Path(__file__).resolve().parent / "data" / "bank-full.csv"
            )
        if path:
            frame = pd.read_csv(path, sep=";", na_values=["?"])
            _, target = prepare_features(frame, args.include_duration)
            provenance = {
                "kind": "bank_csv",
                "filename": path.name,
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "full_rows": len(frame),
            }
            if 0 < args.rows < len(frame):
                selected, _ = train_test_split(
                    np.arange(len(frame)),
                    train_size=args.rows,
                    stratify=target,
                    random_state=args.seed,
                )
                frame = frame.iloc[selected].reset_index(drop=True)
        else:
            if args.rows == 0:
                raise ValueError("--rows 0 requires real data via --csv or --download")
            frame = synthetic_data(args.rows, args.seed)
            provenance = {
                "kind": "synthetic_smoke_data",
                "generator": "sklearn.make_classification",
            }
        report = run_comparison(
            frame, args.seed, args.include_duration, mlp_max_iter=args.mlp_max_iter
        )
        report["dataset"] = provenance
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(
        f"Data: {provenance['kind']}; rows={len(frame)}; duration included={args.include_duration}"
    )
    print(f"{'Model':<12} {'CV AP':>9} {'Test AP':>9} {'Test F1':>9} {'Fit (s)':>9}")
    for row in report["results"]:
        print(
            f"{row['model']:<12} {row['cv_average_precision']:>9.4f} {row['test']['average_precision']:>9.4f} {row['test']['f1']:>9.4f} {row['search_and_refit_seconds']:>9.2f}"
        )
        if row["convergence_warnings"]:
            print(
                f"  Note: {row['convergence_warnings']} fits reached their iteration limit; convergence is not established."
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
