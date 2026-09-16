"""Tests for the AI-assisted paired repeated-holdout maintenance workflow."""

import contextlib
import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from compare import synthetic_data
from repeated_comparison import (
    frame_hash,
    main,
    mean_and_sample_sd,
    run_repeated_comparison,
    select_fixed_sample,
    summarize_runs,
    validate_seeds,
)


def fake_report(frame, seed, include_duration=False, mlp_max_iter=150):
    models = []
    for name, score in (("xgboost", 0.2), ("mlp", 0.3)):
        models.append(
            {
                "model": name,
                "cv_average_precision": score,
                "convergence_warnings": int(name == "mlp"),
                "validation": {"average_precision": score, "f1": score / 2},
                "test": {"average_precision": score, "f1": score / 2},
            }
        )
    return {
        "seed": seed,
        "sample_sha256": frame_hash(frame),
        "include_duration": include_duration,
        "mlp_max_iter": mlp_max_iter,
        "results": models,
    }


def without_clock_fields(value):
    value = copy.deepcopy(value)
    for report in value["runs"]:
        report.pop("created_at_utc", None)
        for model in report["results"]:
            model.pop("search_and_refit_seconds", None)
            for candidate in model.get("cv_summary", []):
                if isinstance(candidate, dict):
                    candidate.pop("mean_fit_time", None)
                    candidate.pop("std_fit_time", None)
                    candidate.pop("mean_score_time", None)
                    candidate.pop("std_score_time", None)
    return value


class RepeatedComparisonTests(unittest.TestCase):
    def test_fixed_sampling_is_reproducible_and_has_source_row_provenance(self):
        frame = synthetic_data(1000, 42)
        sample, metadata = select_fixed_sample(frame, 400, 31)
        again, repeat_metadata = select_fixed_sample(frame, 400, 31)
        other, other_metadata = select_fixed_sample(frame, 400, 32)
        pd.testing.assert_frame_equal(sample, again)
        self.assertEqual(metadata, repeat_metadata)
        self.assertEqual(metadata["selected_rows"], 400)
        self.assertEqual(metadata["source_rows"], 1000)
        self.assertEqual(metadata["sample_sha256"], frame_hash(sample))
        self.assertNotEqual(
            metadata["source_row_positions_sha256"],
            other_metadata["source_row_positions_sha256"],
        )
        self.assertNotEqual(frame_hash(sample), frame_hash(other))
        pd.testing.assert_frame_equal(frame, synthetic_data(1000, 42))

    def test_all_rows_preserve_order_even_when_sample_seed_changes(self):
        frame = synthetic_data(240)
        first, first_meta = select_fixed_sample(frame, 0, 31)
        second, second_meta = select_fixed_sample(frame, 240, 32)
        pd.testing.assert_frame_equal(first, frame)
        pd.testing.assert_frame_equal(second, frame)
        self.assertEqual(
            first_meta["source_row_positions_sha256"],
            second_meta["source_row_positions_sha256"],
        )

    def test_all_run_seeds_receive_identical_sample_and_budget(self):
        frame, _ = select_fixed_sample(synthetic_data(1000), 400, 31)
        seen = []
        progress = []

        def record(sample, **kwargs):
            seen.append((frame_hash(sample), kwargs))
            return fake_report(sample, **kwargs)

        with patch("repeated_comparison.run_comparison", side_effect=record):
            result = run_repeated_comparison(
                frame, [7, 8, 9], True, 1500, progress.append
            )
        self.assertEqual([item[0] for item in seen], [frame_hash(frame)] * 3)
        self.assertEqual([item[1]["seed"] for item in seen], [7, 8, 9])
        self.assertTrue(all(item[1]["mlp_max_iter"] == 1500 for item in seen))
        self.assertTrue(all(item[1]["include_duration"] is True for item in seen))
        self.assertEqual(progress, result["runs"])

    def test_summary_is_paired_by_seed_and_uses_sample_sd(self):
        runs = []
        frame = synthetic_data(240)
        for seed, xgb, mlp in ((7, 0.2, 0.5), (8, 0.4, 0.6), (9, 0.6, 0.7)):
            report = fake_report(frame, seed)
            for entry, value in zip(report["results"], (xgb, mlp)):
                for split in ("validation", "test"):
                    entry[split]["average_precision"] = value
            runs.append(report)
        result = summarize_runs(runs)
        paired = result["paired_average_precision_difference"]["test"]
        self.assertAlmostEqual(paired["mean"], 0.2)
        self.assertAlmostEqual(paired["sample_standard_deviation"], 0.1)
        self.assertEqual([row["seed"] for row in paired["by_seed"]], [7, 8, 9])
        self.assertAlmostEqual(
            result["models"]["xgboost"]["test"]["average_precision"]["mean"], 0.4
        )
        self.assertAlmostEqual(
            result["models"]["xgboost"]["test"]["average_precision"][
                "sample_standard_deviation"
            ],
            0.2,
        )
        self.assertIsNone(
            summarize_runs(runs[:1])["paired_average_precision_difference"]["test"][
                "sample_standard_deviation"
            ]
        )

    def test_same_seed_real_model_runs_reproduce_without_timing_fields(self):
        frame = synthetic_data(240, 42)
        first = run_repeated_comparison(frame, [7, 8], mlp_max_iter=2)
        second = run_repeated_comparison(frame, [7, 8], mlp_max_iter=2)
        self.assertEqual(without_clock_fields(first), without_clock_fields(second))
        self.assertEqual(
            first["runs"][0]["sample_sha256"], first["runs"][1]["sample_sha256"]
        )
        self.assertNotEqual(
            first["runs"][0]["split_sha256"], first["runs"][1]["split_sha256"]
        )

    def test_invalid_seeds_samples_budgets_and_summaries(self):
        for seeds in ([], [7, 7], [-1], [2**32], [True], [1.5]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                validate_seeds(seeds)
        frame = synthetic_data(240)
        for size in (-1, 1, 199, 241, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                select_fixed_sample(frame, size, 42)
        with self.assertRaises(ValueError):
            run_repeated_comparison(frame, [7], mlp_max_iter=0)
        with self.assertRaises(ValueError):
            mean_and_sample_sd([math.nan])
        with self.assertRaises(ValueError):
            summarize_runs([])
        broken = fake_report(frame, 7)
        broken["results"].append(broken["results"][0])
        with self.assertRaises(ValueError):
            summarize_runs([broken])

    def test_runner_detects_sample_mutation(self):
        frame = synthetic_data(240)

        def mutate(sample, **kwargs):
            report = fake_report(sample, **kwargs)
            sample.loc[0, "numeric_0"] = 99999
            return report

        with (
            patch("repeated_comparison.run_comparison", side_effect=mutate),
            self.assertRaisesRegex(ValueError, "sample changed"),
        ):
            run_repeated_comparison(frame, [7], mlp_max_iter=2)

    def test_cli_saves_incrementally_and_refuses_existing_outputs(self):
        with tempfile.TemporaryDirectory(prefix="tabular-repeat-test-") as temporary:
            output = Path(temporary) / "parent" / "new"
            args = [
                "--rows",
                "240",
                "--seeds",
                "7",
                "8",
                "--mlp-max-iter",
                "2",
                "--output-dir",
                str(output),
            ]
            with (
                patch("repeated_comparison.run_comparison", side_effect=fake_report),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(args), 0)
            original = (output / "results.json").read_bytes()
            report = json.loads(original)
            self.assertEqual(report["run_seeds"], [7, 8])
            self.assertEqual(report["sampling"]["sample_seed"], 42)
            self.assertEqual(report["mlp_max_iter"], 2)
            self.assertEqual(
                len((output / "seed-results.jsonl").read_text().splitlines()), 2
            )
            self.assertNotIn(temporary, original.decode())
            self.assertIn("repeated_comparison.py", report["source_sha256"])
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(args)
            self.assertEqual((output / "results.json").read_bytes(), original)

    def test_partial_progress_survives_a_later_failed_fit(self):
        with tempfile.TemporaryDirectory(prefix="tabular-repeat-test-") as temporary:
            output = Path(temporary) / "new"

            def fail_second(sample, **kwargs):
                if kwargs["seed"] == 8:
                    raise ValueError("controlled second fit failure")
                return fake_report(sample, **kwargs)

            with (
                patch("repeated_comparison.run_comparison", side_effect=fail_second),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(
                    ["--rows", "240", "--seeds", "7", "8", "--output-dir", str(output)]
                )
            progress = (output / "seed-results.jsonl").read_text().splitlines()
            self.assertEqual(len(progress), 1)
            self.assertEqual(json.loads(progress[0])["seed"], 7)
            self.assertTrue((output / "metadata.json").is_file())
            self.assertFalse((output / "results.json").exists())

    def test_csv_provenance_hashes_exact_bytes_without_bundling_records(self):
        with tempfile.TemporaryDirectory(prefix="tabular-csv-test-") as temporary:
            directory = Path(temporary)
            csv_path = directory / "public-synthetic.csv"
            output = directory / "new-output"
            frame = synthetic_data(1000, 42)
            marker = "RAW_SYNTHETIC_RECORD_NOT_FOR_OUTPUT"
            frame["public_marker"] = [f"{marker}_{row}" for row in range(len(frame))]
            # BOM and CRLF distinguish original bytes from a normalized re-export.
            raw = frame.to_csv(index=False, sep=";", lineterminator="\r\n").encode(
                "utf-8-sig"
            )
            csv_path.write_bytes(raw)
            parsed = pd.read_csv(io.BytesIO(raw), sep=";", na_values=["?"])
            expected_sample, expected_sampling = select_fixed_sample(parsed, 400, 31)
            observed_samples = []

            def record(sample, **kwargs):
                observed_samples.append(sample.copy())
                return fake_report(sample, **kwargs)

            with (
                patch("repeated_comparison.run_comparison", side_effect=record),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "--csv",
                            str(csv_path),
                            "--rows",
                            "400",
                            "--sample-seed",
                            "31",
                            "--seeds",
                            "7",
                            "8",
                            "--mlp-max-iter",
                            "2",
                            "--output-dir",
                            str(output),
                        ]
                    ),
                    0,
                )
            expected_dataset = {
                "kind": "bank_csv",
                "filename": csv_path.name,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "full_rows": 1000,
            }
            report = json.loads((output / "results.json").read_bytes())
            metadata = json.loads((output / "metadata.json").read_bytes())
            self.assertEqual(report["dataset"], expected_dataset)
            self.assertEqual(metadata["dataset"], expected_dataset)
            self.assertEqual(report["sampling"], expected_sampling)
            self.assertEqual(report["sampling"]["source_rows"], 1000)
            self.assertEqual(report["sampling"]["selected_rows"], 400)
            self.assertEqual(
                report["sampling"]["sample_sha256"], frame_hash(expected_sample)
            )
            self.assertEqual(len(observed_samples), 2)
            for sample in observed_samples:
                pd.testing.assert_frame_equal(sample, expected_sample)
            self.assertEqual(csv_path.read_bytes(), raw)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["metadata.json", "results.json", "seed-results.jsonl"],
            )
            for path in output.iterdir():
                payload = path.read_bytes()
                self.assertNotIn(raw, payload)
                self.assertNotIn(marker.encode(), payload)


if __name__ == "__main__":
    unittest.main()
