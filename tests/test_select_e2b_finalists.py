import unittest

from scripts.select_e2b_finalists import (
    beta_superiority_probability,
    fit_task_policy,
)


class FinalistSelectionTests(unittest.TestCase):
    def test_beta_superiority_probability_is_symmetric(self) -> None:
        self.assertEqual(beta_superiority_probability(0, 0), 0.5)
        self.assertGreater(beta_superiority_probability(3, 0), 0.9)
        self.assertLess(beta_superiority_probability(0, 3), 0.1)
        self.assertAlmostEqual(
            beta_superiority_probability(2, 1),
            1.0 - beta_superiority_probability(1, 2),
        )

    def test_router_requires_both_net_wins_and_posterior_gate(self) -> None:
        keys = [("bbh", "task", index) for index in range(4)]
        rows_by_arm = {
            "direct_answer": {
                key: {"correct": value, "usage": {"completion_tokens": 1}}
                for key, value in zip(keys, [False, False, True, True])
            },
            "challenger": {
                key: {"correct": value, "usage": {"completion_tokens": 2}}
                for key, value in zip(keys, [True, True, True, True])
            },
        }
        policy, _ = fit_task_policy(
            rows_by_arm,
            ["direct_answer", "challenger"],
            "direct_answer",
            keys,
            minimum_net_wins=2,
            posterior_threshold=0.8,
            manifest_index={"direct_answer": 0, "challenger": 1},
        )
        self.assertEqual(policy[("bbh", "task")], "challenger")

        conservative_policy, _ = fit_task_policy(
            rows_by_arm,
            ["direct_answer", "challenger"],
            "direct_answer",
            keys,
            minimum_net_wins=3,
            posterior_threshold=0.8,
            manifest_index={"direct_answer": 0, "challenger": 1},
        )
        self.assertEqual(conservative_policy[("bbh", "task")], "direct_answer")


if __name__ == "__main__":
    unittest.main()
