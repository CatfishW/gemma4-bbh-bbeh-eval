import unittest

from scripts.audit_inference_budgets import arm_budget_metrics


class InferenceBudgetAuditTests(unittest.TestCase):
    def test_cap_binding_and_conditional_accuracy_are_separate(self):
        config = {
            "max_tokens": 10,
            "selection_max_tokens": 5,
            "prompt_strategy": "toy",
            "response_selection": "self_rank",
            "self_consistency_k": 1,
        }
        rows = [
            {
                "correct": False,
                "error": None,
                "prediction": "unfinished",
                "usage": {"completion_tokens": 15},
                "generations": [{"usage": {"completion_tokens": 10}}],
                "selection": {"usage": {"completion_tokens": 5}},
            },
            {
                "correct": True,
                "error": None,
                "prediction": "The final answer is: A",
                "usage": {"completion_tokens": 4},
                "generations": [{"usage": {"completion_tokens": 4}}],
                "selection": {"usage": {"completion_tokens": 2}},
            },
        ]
        result = arm_budget_metrics("toy-model", "toy-arm", config, rows)
        self.assertEqual(result["generation_cap_bindings"], 1)
        self.assertEqual(result["selection_cap_bindings"], 1)
        self.assertEqual(result["examples_with_any_cap_binding"], 1)
        self.assertEqual(result["accuracy_with_cap_binding"], 0.0)
        self.assertEqual(result["accuracy_without_cap_binding"], 1.0)
        self.assertEqual(result["final_answer_marker_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
