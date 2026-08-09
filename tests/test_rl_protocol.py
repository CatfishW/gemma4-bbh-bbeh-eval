import unittest

from eval_benchmarks import Example
from rl.protocol import (
    build_protocol_split,
    holdout_tasks,
    prompt_id,
    split_name,
)


def make_examples(benchmark: str, task: str, count: int) -> list[Example]:
    return [
        Example(benchmark=benchmark, task=task, index=index, input=f"q{index}", target="(A)")
        for index in range(count)
    ]


class SplitNameTests(unittest.TestCase):
    def test_boundaries(self) -> None:
        example = lambda index: Example("bbh", "t", index, "q", "a")  # noqa: E731
        self.assertEqual(split_name(example(0)), "calibration")
        self.assertEqual(split_name(example(24)), "calibration")
        self.assertEqual(split_name(example(25)), "validation")
        self.assertEqual(split_name(example(49)), "validation")
        self.assertEqual(split_name(example(50)), "test")


class ProtocolSplitTests(unittest.TestCase):
    def test_counts_match_protocol(self) -> None:
        examples = make_examples("bbh", "alpha", 100)
        split = build_protocol_split(examples, holdout_stride=0)
        self.assertEqual(len(split.train), 25)
        self.assertEqual(len(split.validation), 25)
        self.assertEqual(len(split.test), 50)

    def test_short_tasks_never_reach_test(self) -> None:
        examples = make_examples("bbh", "alpha", 40)
        split = build_protocol_split(examples, holdout_stride=0)
        self.assertEqual(len(split.train), 25)
        self.assertEqual(len(split.validation), 15)
        self.assertEqual(len(split.test), 0)

    def test_holdout_tasks_excluded_from_train_only(self) -> None:
        examples = (
            make_examples("bbh", "alpha", 60)
            + make_examples("bbh", "beta", 60)
            + make_examples("bbh", "gamma", 60)
            + make_examples("bbh", "delta", 60)
            + make_examples("bbh", "epsilon", 60)
        )
        split = build_protocol_split(examples, holdout_stride=4)
        # alphabetical: alpha, beta, delta, epsilon, gamma -> positions 0 and 4 held out
        self.assertEqual(split.holdout_tasks, ("bbh/alpha", "bbh/gamma"))
        train_tasks = {f"{e.benchmark}/{e.task}" for e in split.train}
        self.assertNotIn("bbh/alpha", train_tasks)
        self.assertNotIn("bbh/gamma", train_tasks)
        self.assertIn("bbh/beta", train_tasks)
        validation_tasks = {f"{e.benchmark}/{e.task}" for e in split.validation}
        test_tasks = {f"{e.benchmark}/{e.task}" for e in split.test}
        self.assertIn("bbh/alpha", validation_tasks)
        self.assertIn("bbh/alpha", test_tasks)

    def test_holdout_is_per_benchmark_and_deterministic(self) -> None:
        examples = make_examples("bbh", "alpha", 60) + make_examples("bbeh", "omega", 60)
        held = holdout_tasks(examples, stride=4)
        self.assertEqual(held, ("bbeh/omega", "bbh/alpha"))
        self.assertEqual(held, holdout_tasks(list(reversed(examples)), stride=4))


class PromptIdTests(unittest.TestCase):
    def test_prompt_id_format(self) -> None:
        example = Example("usr", "unpuzzles/original", 3, "q", "a")
        self.assertEqual(prompt_id(example), "usr/unpuzzles/original/3")


if __name__ == "__main__":
    unittest.main()
