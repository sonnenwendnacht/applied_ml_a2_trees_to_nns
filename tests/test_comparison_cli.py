"""Regression coverage for the maintained single-run CSV entry point."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from compare import main, synthetic_data


class ComparisonCliTests(unittest.TestCase):
    def test_requested_sample_cannot_silently_exceed_csv_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            output = Path(temporary) / "result.json"
            source.write_text(synthetic_data(240).to_csv(sep=";", index=False))
            with (
                patch("compare.run_comparison", return_value={"results": []}) as run,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as errors,
                self.assertRaises(SystemExit) as raised,
            ):
                main(["--csv", str(source), "--rows", "241", "--output", str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("exceeds", errors.getvalue())
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_csv_hash_describes_the_bytes_parsed_even_if_source_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            output = Path(temporary) / "result.json"
            raw = synthetic_data(240).to_csv(sep=";", index=False).encode("utf-8")
            replacement = synthetic_data(300).to_csv(sep=";", index=False).encode()
            source.write_bytes(raw)
            expected = pd.read_csv(io.BytesIO(raw), sep=";", na_values=["?"])
            read_csv = pd.read_csv

            def change_source_after_parsing(*args, **kwargs):
                frame = read_csv(*args, **kwargs)
                source.write_bytes(replacement)
                return frame

            with (
                patch("compare.pd.read_csv", side_effect=change_source_after_parsing),
                patch("compare.run_comparison", return_value={"results": []}) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        ["--csv", str(source), "--rows", "0", "--output", str(output)]
                    ),
                    0,
                )
            pd.testing.assert_frame_equal(run.call_args.args[0], expected)
            report = json.loads(output.read_text())
            self.assertEqual(
                report["dataset"]["file_sha256"], hashlib.sha256(raw).hexdigest()
            )
            self.assertEqual(report["dataset"]["full_rows"], len(expected))
            self.assertEqual(source.read_bytes(), replacement)

    def test_zero_exact_and_smaller_sample_sizes_remain_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            source.write_text(synthetic_data(300).to_csv(sep=";", index=False))
            for rows, expected in ((0, 300), (300, 300), (240, 240)):
                with (
                    self.subTest(rows=rows),
                    patch(
                        "compare.run_comparison", return_value={"results": []}
                    ) as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(
                        main(["--csv", str(source), "--rows", str(rows)]), 0
                    )
                    self.assertEqual(len(run.call_args.args[0]), expected)


if __name__ == "__main__":
    unittest.main()
