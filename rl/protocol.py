"""Frozen data protocol for weight-level RL.

Reuses the confirmatory-study split exactly:

- calibration: 0 <= index < 25   (RL training pool)
- validation:  25 <= index < 50  (model selection only)
- test:        index >= 50       (frozen; final claims only)

Tasks with fewer than 50 examples contribute rows only to calibration and
validation, never to test — identical to `experiments/e2b_confirmatory_protocol.json`.

To measure unseen-task generalization, every fourth task per benchmark in
alphabetical order is excluded from the RL training pool (its calibration rows
are never trained on) while remaining part of validation and test scoring.
"""
from __future__ import annotations

from dataclasses import dataclass

from eval_benchmarks import Example, example_task_key

CALIBRATION_END = 25
VALIDATION_END = 50
HOLDOUT_TASK_STRIDE = 4


@dataclass(frozen=True)
class ProtocolSplit:
    train: list[Example]
    validation: list[Example]
    test: list[Example]
    holdout_tasks: tuple[str, ...]


def split_name(example: Example) -> str:
    if example.index < CALIBRATION_END:
        return "calibration"
    if example.index < VALIDATION_END:
        return "validation"
    return "test"


def holdout_tasks(examples: list[Example], stride: int = HOLDOUT_TASK_STRIDE) -> tuple[str, ...]:
    """Every `stride`-th task per benchmark, alphabetical, held out from training."""
    by_benchmark: dict[str, set[str]] = {}
    for example in examples:
        by_benchmark.setdefault(example.benchmark, set()).add(example.task)
    held: list[str] = []
    for benchmark in sorted(by_benchmark):
        tasks = sorted(by_benchmark[benchmark])
        held.extend(
            f"{benchmark}/{task}" for position, task in enumerate(tasks) if position % stride == 0
        )
    return tuple(held)


def build_protocol_split(
    examples: list[Example],
    holdout_stride: int = HOLDOUT_TASK_STRIDE,
) -> ProtocolSplit:
    held = set(holdout_tasks(examples, stride=holdout_stride)) if holdout_stride else set()
    train: list[Example] = []
    validation: list[Example] = []
    test: list[Example] = []
    for example in examples:
        name = split_name(example)
        if name == "calibration":
            if example_task_key(example) not in held:
                train.append(example)
        elif name == "validation":
            validation.append(example)
        else:
            test.append(example)
    return ProtocolSplit(
        train=train,
        validation=validation,
        test=test,
        holdout_tasks=tuple(sorted(held)),
    )


def prompt_id(example: Example) -> str:
    return f"{example.benchmark}/{example.task}/{example.index}"
