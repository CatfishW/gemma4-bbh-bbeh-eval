import unittest

from scripts.analyze_policy_fallback_sensitivity import build_fallback_replay


class PolicyFallbackSensitivityTests(unittest.TestCase):
    def test_replay_uses_baseline_only_for_fallback_rows(self) -> None:
        fallback_key = ("bbh", "a", 50)
        specialized_key = ("usr", "b", 51)
        baseline = {
            fallback_key: {"prediction": "baseline-fallback"},
            specialized_key: {"prediction": "baseline-specialized"},
        }
        policy = {
            fallback_key: {
                "prediction": "policy-fallback",
                "prompt_strategy": "direct_answer",
            },
            specialized_key: {
                "prediction": "policy-specialized",
                "prompt_strategy": "concise_cot",
            },
        }
        replay, fallback_keys, specialized_keys = build_fallback_replay(
            baseline, policy, "direct_answer"
        )
        self.assertEqual(replay[fallback_key]["prediction"], "baseline-fallback")
        self.assertEqual(replay[specialized_key]["prediction"], "policy-specialized")
        self.assertEqual(fallback_keys, [fallback_key])
        self.assertEqual(specialized_keys, [specialized_key])


if __name__ == "__main__":
    unittest.main()
