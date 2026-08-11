import unittest

from rl.rewards import LengthDualController, LengthShapingConfig, correctness_reward


class CorrectnessRewardTests(unittest.TestCase):
    def test_uses_repository_scorer_normalization(self) -> None:
        self.assertEqual(correctness_reward("The final answer is: (A)", "(A)"), 1)
        self.assertEqual(correctness_reward("A", "(A)"), 1)
        self.assertEqual(correctness_reward("The final answer is: 42.0", "42"), 1)
        self.assertEqual(correctness_reward("I think it is B", "(A)"), 0)
        self.assertEqual(correctness_reward("yes", "Yes"), 1)


class LengthDualControllerTests(unittest.TestCase):
    def enabled_config(self) -> LengthShapingConfig:
        return LengthShapingConfig(
            enabled=True,
            target_length=100.0,
            max_length=200.0,
            initial_multiplier=0.2,
            max_multiplier=0.5,
            step_size=0.1,
        )

    def test_disabled_controller_is_inert(self) -> None:
        controller = LengthDualController(LengthShapingConfig(enabled=False))
        self.assertEqual(controller.shaped_reward(1, 500), 1.0)
        self.assertEqual(controller.shaped_baseline(0.4, 500), 0.4)
        controller.update(500.0)
        self.assertEqual(controller.multiplier, 0.0)

    def test_wrong_rollouts_never_penalized(self) -> None:
        controller = LengthDualController(self.enabled_config())
        self.assertEqual(controller.shaped_reward(0, 10_000), 0.0)

    def test_penalty_scales_with_length_and_caps(self) -> None:
        controller = LengthDualController(self.enabled_config())
        short = controller.shaped_reward(1, 50)
        long = controller.shaped_reward(1, 200)
        longest = controller.shaped_reward(1, 5_000)
        self.assertGreater(short, long)
        self.assertEqual(long, longest)
        self.assertAlmostEqual(short, 1.0 - 0.2 * 0.25)

    def test_dual_ascent_moves_toward_constraint(self) -> None:
        controller = LengthDualController(self.enabled_config())
        controller.update(mean_correct_length=200.0)
        self.assertAlmostEqual(controller.multiplier, 0.3)
        controller.update(mean_correct_length=50.0)
        self.assertAlmostEqual(controller.multiplier, 0.25)

    def test_dual_projection_bounds(self) -> None:
        controller = LengthDualController(self.enabled_config())
        for _ in range(20):
            controller.update(mean_correct_length=1_000.0)
        self.assertEqual(controller.multiplier, 0.5)
        for _ in range(50):
            controller.update(mean_correct_length=1.0)
        self.assertEqual(controller.multiplier, 0.0)

    def test_baseline_shaping_is_consistent(self) -> None:
        controller = LengthDualController(self.enabled_config())
        baseline = controller.shaped_baseline(0.5, mean_correct_length=100.0)
        self.assertAlmostEqual(baseline, 0.5 * (1.0 - 0.2 * 0.5))


if __name__ == "__main__":
    unittest.main()
