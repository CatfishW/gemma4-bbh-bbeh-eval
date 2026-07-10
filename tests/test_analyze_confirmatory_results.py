import unittest

from scripts.analyze_confirmatory_results import exact_mcnemar_p, holm_adjust


class ConfirmatoryStatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_handles_symmetry_and_no_discordance(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(3, 1), exact_mcnemar_p(1, 3))
        self.assertLess(exact_mcnemar_p(10, 0), 0.01)

    def test_holm_adjustment_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.04)
        self.assertAlmostEqual(adjusted["c"], 0.5)


if __name__ == "__main__":
    unittest.main()
