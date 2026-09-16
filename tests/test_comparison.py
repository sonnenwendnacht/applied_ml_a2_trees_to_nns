import json
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from compare import (
    build_pipeline,
    classification_metrics,
    parameter_metadata,
    prepare_features,
    probability_sha256,
    run_comparison,
    split_data,
    synthetic_data,
)


class ComparisonTests(unittest.TestCase):
    def test_default_feature_policy_excludes_target_and_duration(self):
        frame = synthetic_data(240)
        features, target = prepare_features(frame)
        self.assertNotIn("duration", features)
        self.assertNotIn("y", features)
        self.assertEqual(set(target), {0, 1})
        self.assertIn("duration", prepare_features(frame, include_duration=True)[0])

    def test_holdouts_are_disjoint_complete_and_reproducible(self):
        features, target = prepare_features(synthetic_data(240))
        splits = split_data(features, target, 42)
        again = split_data(features, target, 42)
        combined = np.concatenate(list(splits.values()))
        self.assertEqual(len(set(combined)), len(target))
        self.assertEqual(set(combined), set(range(len(target))))
        for name, indices in splits.items():
            np.testing.assert_array_equal(indices, again[name])
            self.assertEqual(target.iloc[indices].nunique(), 2)

    def test_preprocessing_is_refit_inside_each_cross_validation_fold(self):
        frame = pd.DataFrame(
            {"amount": [0.0, 1, 2, 3, 4, 5, 6, 1000], "kind": list("aabbaabb")}
        )
        target = pd.Series([0, 1] * 4)
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        folds = list(cv.split(frame, target))
        results = cross_validate(
            build_pipeline(DummyClassifier()),
            frame,
            target,
            cv=folds,
            return_estimator=True,
        )
        for estimator, (train, _) in zip(results["estimator"], folds):
            scaler = (
                estimator.named_steps["preprocess"]
                .named_transformers_["numeric"]
                .named_steps["scale"]
            )
            self.assertAlmostEqual(scaler.mean_[0], frame.iloc[train]["amount"].mean())
            self.assertNotEqual(scaler.mean_[0], frame["amount"].mean())

    def test_unseen_categories_and_missing_values_do_not_refit_preprocessing(self):
        train = pd.DataFrame(
            {"amount": [1.0, 2.0, np.nan, 4.0], "kind": ["a", "a", "b", np.nan]}
        )
        pipeline = build_pipeline(DummyClassifier()).fit(train, [0, 1, 0, 1])
        transform = pipeline.named_steps["preprocess"]
        mean = (
            transform.named_transformers_["numeric"].named_steps["scale"].mean_.copy()
        )
        heldout = pd.DataFrame(
            {"amount": [10000.0, np.nan], "kind": ["unseen", np.nan]}
        )
        self.assertTrue(np.isfinite(transform.transform(heldout)).all())
        np.testing.assert_array_equal(
            mean, transform.named_transformers_["numeric"].named_steps["scale"].mean_
        )

    def test_average_precision_and_thresholded_metrics_are_distinct(self):
        result = classification_metrics(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])
        )
        self.assertEqual(result["average_precision"], 1.0)
        self.assertEqual(result["f1"], 0.0)

    def test_invalid_target_is_rejected(self):
        frame = synthetic_data(240)
        frame.loc[0, "y"] = "unexpected"
        with self.assertRaises(ValueError):
            prepare_features(frame)

    def test_invalid_iteration_budgets_are_rejected(self):
        for budget in (0, -1, True, 1.5, "150"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                run_comparison(synthetic_data(240), mlp_max_iter=budget)

    def test_parameter_sentinels_are_explicit_not_nonstandard_json(self):
        encoded = parameter_metadata(
            {"missing": np.float64(np.nan), "layers": (32, 16)}
        )
        self.assertEqual(encoded["missing"], {"nonfinite_parameter": "nan"})
        self.assertEqual(encoded["layers"], [32, 16])
        json.dumps(encoded, allow_nan=False)
        with self.assertRaises(TypeError):
            parameter_metadata(object())

    def test_probability_hash_checks_values_order_and_normalizes_byte_order(self):
        values = np.array([0.2, 0.8], dtype=">f8")
        self.assertEqual(
            probability_sha256(values), probability_sha256(values.astype("<f8"))
        )
        self.assertNotEqual(
            probability_sha256(values), probability_sha256(values[::-1])
        )
        with self.assertRaises(ValueError):
            probability_sha256([float("nan")])

    def test_iteration_diagnostics_and_strict_json_on_real_fit(self):
        report = run_comparison(synthetic_data(240), mlp_max_iter=1)
        self.assertEqual(report["training_config"]["mlp_max_iter"], 1)
        self.assertEqual(report["schema_version"], 2)
        xgb, mlp = report["results"]
        self.assertEqual(
            xgb["selected_estimator_params"]["missing"], {"nonfinite_parameter": "nan"}
        )
        self.assertEqual(mlp["refit_diagnostics"]["n_iter"], 1)
        self.assertEqual(
            mlp["refit_diagnostics"]["stop_observation"], "iteration_limit_reached"
        )
        self.assertEqual(mlp["convergence_warnings"], 7)
        for model in report["results"]:
            self.assertEqual(len(model["cv_candidates"]), 2)
            self.assertEqual(
                len(model["cv_candidates"][0]["fold_average_precision"]), 3
            )
            self.assertEqual(set(model["probability_sha256"]), {"validation", "test"})
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
