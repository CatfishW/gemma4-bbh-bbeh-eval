"""Result aggregation must never silently compare partial/mismatched runs."""
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rl_craft.core import Example
from rl_craft.data import file_hash
from scripts.craft_pilot import report


class PilotReportTests(unittest.TestCase):
    def fixture(self, root):
        rows = [Example("suite/seen/25", "suite/seen", "q", "a", 25),
                Example("suite/held/25", "suite/held", "q2", "b", 25)]
        (root / "validation.jsonl").write_text("".join(json.dumps(asdict(r))+"\n" for r in rows))
        (root / "study.json").write_text(json.dumps({"seen_tasks": ["suite/seen"], "heldout_tasks": ["suite/held"],
            "validation_sha256": file_hash(root / "validation.jsonl")}))
        for arm in ("base", "full", "no_crossfit"):
            for mode in ("sample", "always-continue"):
                folder = root / f"eval-{arm}-{mode}"
                folder.mkdir()
                records = [{"key": r.key, "task": r.task, "index": r.index, "correct": arm == "full"} for r in rows]
                (folder / "summary.json").write_text(json.dumps({"n": 2}))
                (folder / "predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            if arm != "base":
                folder = root / arm / "checkpoint-00001"
                folder.mkdir(parents=True)
                (folder.parent / "latest").write_text(folder.name)
                metrics = [{"sampled_tokens": 10, "generation_prefill_tokens": 20,
                            "wall_seconds": 1, "peak_allocated_bytes": 100,
                            "mean_stop_probability": .5}]
                (folder / "metrics.json").write_text(json.dumps(metrics))

    def test_paired_counts_and_heldout_denominators(self):
        with tempfile.TemporaryDirectory() as d, patch("sys.stdout", new_callable=io.StringIO):
            root = Path(d); self.fixture(root)
            report(SimpleNamespace(output=root))
            result = json.loads((root / "comparison.json").read_text())
            comparison = result["paired_comparisons"]["full-sample vs base-sample"]
            self.assertEqual((comparison["wins"], comparison["losses"]), (2, 0))
            self.assertAlmostEqual(comparison["mcnemar_p"], .5)
            self.assertEqual(result["evaluation"]["full-sample"]["heldout"], {"n": 1, "correct": 1})

    def test_partial_or_duplicate_predictions_fail(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as d:
                root = Path(d); self.fixture(root)
                path = root / "eval-full-sample/predictions.jsonl"
                first = path.read_text().splitlines()[0] + "\n"
                path.write_text(first * (2 if duplicate else 1))
                with self.assertRaisesRegex(ValueError, "incomplete/mismatched"):
                    report(SimpleNamespace(output=root))
                self.assertFalse((root / "comparison.json").exists())

    def test_changed_validation_identity_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            path = root / "validation.jsonl"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "differs from declared"):
                report(SimpleNamespace(output=root))


if __name__ == "__main__":
    unittest.main()
