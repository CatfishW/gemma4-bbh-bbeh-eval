import json
from pathlib import Path
import tempfile
import unittest

from rl.persistence import truncate_jsonl_at_iteration


class ResumePersistenceTests(unittest.TestCase):
    def test_truncates_rows_at_and_after_resumed_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                "".join(json.dumps({"iteration": index}) + "\n" for index in range(7))
            )

            removed = truncate_jsonl_at_iteration(path, 5)

            self.assertEqual(removed, 2)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["iteration"] for row in rows], [0, 1, 2, 3, 4])

    def test_missing_file_is_a_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            self.assertEqual(truncate_jsonl_at_iteration(path, 3), 0)
            self.assertFalse(path.exists())

    def test_rejects_rows_without_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text('{"reward": 1}\n')
            with self.assertRaisesRegex(ValueError, "missing iteration"):
                truncate_jsonl_at_iteration(path, 1)


if __name__ == "__main__":
    unittest.main()
