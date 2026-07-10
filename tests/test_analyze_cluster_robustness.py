import unittest

from scripts.analyze_cluster_robustness import (
    cluster_bootstrap,
    cluster_sign_flip_p,
    leave_one_task_out,
)


class ClusterRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [
            {
                "benchmark": "toy",
                "task": "a",
                "total": 10,
                "correct_difference": 2,
                "accuracy_difference": 0.2,
            },
            {
                "benchmark": "toy",
                "task": "b",
                "total": 20,
                "correct_difference": 4,
                "accuracy_difference": 0.2,
            },
        ]

    def test_bootstrap_is_deterministic_and_preserves_constant_effect(self):
        result = cluster_bootstrap(self.tasks, replicates=100, seed=7)
        self.assertEqual(result["pooled_micro_difference_95"], [0.2, 0.2])
        self.assertEqual(result["macro_task_difference_95"], [0.2, 0.2])

    def test_sign_flip_probability_is_bounded(self):
        value = cluster_sign_flip_p(self.tasks, replicates=1000, seed=7)
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_leave_one_task_out_reports_both_extremes(self):
        result = leave_one_task_out(self.tasks)
        self.assertAlmostEqual(result["minimum"]["pooled_accuracy_difference"], 0.2)
        self.assertAlmostEqual(result["maximum"]["pooled_accuracy_difference"], 0.2)


if __name__ == "__main__":
    unittest.main()
