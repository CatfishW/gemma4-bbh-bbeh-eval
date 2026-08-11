import unittest

from scripts.compare_official_thinking_evals import exact_mcnemar_p, paired_comparison


def row(task: str, correct: bool, tokens: int) -> dict:
    return {
        "task": task,
        "correct": correct,
        "completion_tokens": tokens,
        "thinking_tokens": tokens - 2,
        "truncated": False,
        "parse_error": None,
    }


class OfficialThinkingComparisonTests(unittest.TestCase):
    def test_exact_mcnemar_handles_no_and_one_sided_discordance(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p(3, 0), 0.25)
        self.assertEqual(exact_mcnemar_p(2, 2), 1.0)

    def test_paired_comparison_reports_wins_losses_and_token_reduction(self):
        keys = ["a", "b", "c", "d"]
        baseline = {
            "a": row("t1", False, 10),
            "b": row("t1", True, 10),
            "c": row("t2", False, 10),
            "d": row("t2", True, 10),
        }
        challenger = {
            "a": row("t1", True, 8),
            "b": row("t1", True, 8),
            "c": row("t2", False, 8),
            "d": row("t2", False, 8),
        }
        result = paired_comparison(
            baseline,
            challenger,
            keys,
            replicates=100,
            seed=7,
        )
        self.assertEqual(result["paired_wins"], 1)
        self.assertEqual(result["paired_losses"], 1)
        self.assertEqual(result["accuracy_point_difference"], 0.0)
        self.assertAlmostEqual(result["completion_token_reduction"], 0.2)


if __name__ == "__main__":
    unittest.main()
