import unittest

from scripts.build_cross_model_report import average_ranks, pearson


class CrossModelReportTests(unittest.TestCase):
    def test_average_ranks_handle_ties(self) -> None:
        self.assertEqual(average_ranks({"a": 1.0, "b": 1.0, "c": 0.0}), {
            "a": 1.5,
            "b": 1.5,
            "c": 3.0,
        })

    def test_pearson_identical_ranks_is_one(self) -> None:
        self.assertAlmostEqual(pearson([1, 2, 3], [1, 2, 3]), 1.0)


if __name__ == "__main__":
    unittest.main()
