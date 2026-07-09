from argparse import Namespace
import unittest

from eval_benchmarks import (
    Example,
    PROMPT_STRATEGIES,
    build_prompt,
    prompt_run_metadata,
    resolve_prompt_strategy,
)


class PromptPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = Example(
            benchmark="bbh",
            task="date_understanding",
            index=25,
            input="Question text",
            target="(A)",
        )

    def test_task_policy_overrides_default_strategy(self) -> None:
        args = Namespace(
            prompt_strategy="direct_answer",
            prompt_policy_map={"bbh/date_understanding": "selective_verify"},
            prompt_policy_default="direct_answer",
        )
        self.assertEqual(resolve_prompt_strategy(args, self.example), "selective_verify")

    def test_policy_falls_back_to_direct_answer(self) -> None:
        args = Namespace(
            prompt_strategy="direct_answer",
            prompt_policy_map={},
            prompt_policy_default="direct_answer",
        )
        self.assertEqual(resolve_prompt_strategy(args, self.example), "direct_answer")

    def test_policy_metadata_records_selected_strategies(self) -> None:
        payload = {
            "name": "test_policy",
            "description": "test",
            "default_strategy": "direct_answer",
            "task_strategies": {"bbh/date_understanding": "selective_verify"},
        }
        args = Namespace(
            prompt_strategy="direct_answer",
            prompt_policy="policy.json",
            prompt_policy_payload=payload,
            prompt_policy_map={"bbh/date_understanding": "selective_verify"},
            prompt_policy_default="direct_answer",
        )
        metadata = prompt_run_metadata(args)
        self.assertEqual(metadata["prompt_strategy"], "policy:test_policy")
        self.assertEqual(
            metadata["prompt_policy_selected_strategies"],
            ["direct_answer", "selective_verify"],
        )

    def test_research_prompts_preserve_input_and_request_final_answer(self) -> None:
        names = {
            "selective_verify",
            "compare_then_commit",
            "fast_slow_gate",
            "constraint_guard",
            "negation_label_guard",
            "draft_verify",
        }
        self.assertTrue(names <= set(PROMPT_STRATEGIES))
        for name in names:
            prompt = build_prompt("Question text", name)
            self.assertIn("Question text", prompt)
            self.assertIn("only", prompt.lower())
            self.assertIn("answer", prompt.lower())


if __name__ == "__main__":
    unittest.main()
