import math
import random
import unittest

from rl.posterior import (
    DifficultyTracker,
    allocate_rollouts,
    allocation_entropy,
    beta_mean,
    expected_bernoulli_variance,
)


class ExpectedBernoulliVarianceTests(unittest.TestCase):
    def test_matches_monte_carlo(self) -> None:
        rng = random.Random(7)
        alpha, beta = 3.0, 5.0
        draws = [rng.betavariate(alpha, beta) for _ in range(200_000)]
        empirical = sum(p * (1.0 - p) for p in draws) / len(draws)
        closed_form = expected_bernoulli_variance(alpha, beta)
        self.assertAlmostEqual(closed_form, empirical, places=3)

    def test_symmetric_and_bounded(self) -> None:
        self.assertAlmostEqual(
            expected_bernoulli_variance(2.0, 6.0), expected_bernoulli_variance(6.0, 2.0)
        )
        self.assertLess(expected_bernoulli_variance(100.0, 1.0), 0.02)
        self.assertGreater(expected_bernoulli_variance(1.0, 1.0), 0.15)


class DifficultyTrackerTests(unittest.TestCase):
    def build(self) -> DifficultyTracker:
        return DifficultyTracker(
            prompt_tasks={
                "bbh/task_a/0": "bbh/task_a",
                "bbh/task_a/1": "bbh/task_a",
                "bbh/task_b/0": "bbh/task_b",
            },
            discount=0.9,
            prior_strength=4.0,
        )

    def test_fresh_prompt_uses_hierarchical_prior(self) -> None:
        tracker = self.build()
        tracker.update({"bbh/task_a/0": [(1, 100)] * 8})
        snapshot = tracker.snapshot()
        sibling = snapshot.prompts["bbh/task_a/1"]
        stranger = snapshot.prompts["bbh/task_b/0"]
        self.assertGreater(sibling.baseline, stranger.baseline)
        self.assertGreater(sibling.baseline, 0.6)
        self.assertAlmostEqual(stranger.baseline, 0.5, places=6)

    def test_discount_decays_stale_evidence(self) -> None:
        tracker = self.build()
        tracker.update({"bbh/task_a/0": [(1, 50)] * 10})
        strong = tracker.snapshot().prompts["bbh/task_a/0"].baseline
        for _ in range(10):
            tracker.update({})
        toward_task_prior = tracker.snapshot().prompts["bbh/task_a/0"].baseline
        for _ in range(60):
            tracker.update({})
        toward_global_prior = tracker.snapshot().prompts["bbh/task_a/0"].baseline
        self.assertGreater(strong, 0.8)
        self.assertLess(toward_task_prior, strong)
        self.assertLess(toward_global_prior, toward_task_prior)
        self.assertLess(abs(toward_global_prior - 0.5), 0.1)

    def test_snapshot_is_predictable(self) -> None:
        tracker = self.build()
        tracker.update({"bbh/task_a/0": [(1, 60), (0, 80)]})
        snapshot = tracker.snapshot()
        before = snapshot.prompts["bbh/task_a/0"].baseline
        tracker.update({"bbh/task_a/0": [(1, 60)] * 5})
        self.assertEqual(snapshot.prompts["bbh/task_a/0"].baseline, before)

    def test_serialization_roundtrip(self) -> None:
        tracker = self.build()
        tracker.update({"bbh/task_a/0": [(1, 64)], "bbh/task_b/0": [(0, 32)]})
        clone = DifficultyTracker.from_json(tracker.to_json())
        original = tracker.snapshot()
        restored = clone.snapshot()
        for prompt in original.prompts:
            self.assertAlmostEqual(
                original.prompts[prompt].baseline, restored.prompts[prompt].baseline
            )
            self.assertAlmostEqual(
                original.prompts[prompt].allocation_score,
                restored.prompts[prompt].allocation_score,
            )

    def test_length_statistics(self) -> None:
        tracker = self.build()
        tracker.update({"bbh/task_a/0": [(1, 100), (0, 300)]})
        snapshot = tracker.snapshot()
        state = snapshot.prompts["bbh/task_a/0"]
        self.assertAlmostEqual(state.mean_length, 200.0, places=1)
        self.assertAlmostEqual(state.mean_correct_length, 100.0, places=1)
        sibling = snapshot.prompts["bbh/task_a/1"]
        self.assertAlmostEqual(sibling.mean_length, 200.0, places=1)


class AllocationTests(unittest.TestCase):
    def test_budget_conserved_and_caps_respected(self) -> None:
        scores = {f"p{i}": 0.1 + 0.05 * i for i in range(10)}
        lru = sorted(scores)
        allocation = allocate_rollouts(
            scores, budget=32, n_max=6, floor_fraction=0.25, least_recently_sampled=lru
        )
        self.assertEqual(sum(allocation.values()), 32)
        self.assertTrue(all(count <= 6 for count in allocation.values()))

    def test_higher_score_gets_no_fewer_rollouts(self) -> None:
        scores = {"low": 0.05, "mid": 0.25, "high": 0.5}
        allocation = allocate_rollouts(
            scores,
            budget=12,
            n_max=8,
            floor_fraction=0.0,
            least_recently_sampled=sorted(scores),
        )
        self.assertGreaterEqual(allocation.get("high", 0), allocation.get("mid", 0))
        self.assertGreaterEqual(allocation.get("mid", 0), allocation.get("low", 0))

    def test_floor_goes_to_least_recently_sampled(self) -> None:
        scores = {"stale": 0.0, "fresh": 1.0}
        allocation = allocate_rollouts(
            scores,
            budget=4,
            n_max=4,
            floor_fraction=0.25,
            least_recently_sampled=["stale", "fresh"],
        )
        self.assertGreaterEqual(allocation.get("stale", 0), 1)
        self.assertEqual(sum(allocation.values()), 4)

    def test_zero_scores_fall_back_to_uniform_fill(self) -> None:
        scores = {"a": 0.0, "b": 0.0}
        allocation = allocate_rollouts(
            scores, budget=6, n_max=4, floor_fraction=0.0, least_recently_sampled=["a", "b"]
        )
        self.assertEqual(sum(allocation.values()), 6)

    def test_budget_beyond_capacity_saturates(self) -> None:
        scores = {"a": 0.3, "b": 0.2}
        allocation = allocate_rollouts(
            scores, budget=100, n_max=4, floor_fraction=0.0, least_recently_sampled=["a", "b"]
        )
        self.assertEqual(allocation, {"a": 4, "b": 4})

    def test_entropy(self) -> None:
        self.assertAlmostEqual(allocation_entropy({"a": 2, "b": 2}), math.log(2))
        self.assertEqual(allocation_entropy({"a": 4}), 0.0)
        self.assertEqual(allocation_entropy({}), 0.0)


class BetaMeanTests(unittest.TestCase):
    def test_beta_mean(self) -> None:
        self.assertAlmostEqual(beta_mean(2.0, 2.0), 0.5)
        self.assertAlmostEqual(beta_mean(9.0, 1.0), 0.9)


if __name__ == "__main__":
    unittest.main()
