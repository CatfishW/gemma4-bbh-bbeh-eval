"""Rewards for verifiable-reasoning RL.

Correctness is scored by the unchanged repository scorer so the training signal
is byte-identical to the frozen evaluation protocol. Deployment efficiency is a
constraint, not a reward hack: a success-conditioned length penalty whose
multiplier follows projected dual ascent toward a target mean length of correct
completions.
"""
from __future__ import annotations

from dataclasses import dataclass

from eval_benchmarks import evaluate_correctness


def correctness_reward(completion: str, target: str) -> int:
    return 1 if evaluate_correctness(completion, target) else 0


@dataclass
class LengthShapingConfig:
    enabled: bool = False
    target_length: float = 160.0
    max_length: float = 320.0
    initial_multiplier: float = 0.0
    max_multiplier: float = 0.5
    step_size: float = 0.05


class LengthDualController:
    """Projected dual ascent on E[len | correct] <= target_length.

    The penalty applies only to correct rollouts: penalizing wrong rollouts for
    length rewards giving up early, while penalizing only successes moves mass
    toward the shortest working solutions.
    """

    def __init__(self, config: LengthShapingConfig) -> None:
        self.config = config
        self.multiplier = config.initial_multiplier if config.enabled else 0.0

    def shaped_reward(self, reward: int, completion_length: int) -> float:
        if not reward:
            return 0.0
        if not self.config.enabled or self.multiplier <= 0.0:
            return float(reward)
        fraction = min(1.0, completion_length / self.config.max_length)
        return float(reward) * (1.0 - self.multiplier * fraction)

    def shaped_baseline(self, baseline: float, mean_correct_length: float) -> float:
        """Predictable baseline for the shaped reward (frozen statistics only)."""
        if not self.config.enabled or self.multiplier <= 0.0:
            return baseline
        fraction = min(1.0, mean_correct_length / self.config.max_length)
        return baseline * (1.0 - self.multiplier * fraction)

    def update(self, mean_correct_length: float | None) -> None:
        if not self.config.enabled or mean_correct_length is None:
            return
        gradient = mean_correct_length / self.config.target_length - 1.0
        self.multiplier = min(
            max(self.multiplier + self.config.step_size * gradient, 0.0),
            self.config.max_multiplier,
        )

    def state(self) -> dict:
        return {"multiplier": self.multiplier}

    def load_state(self, payload: dict) -> None:
        self.multiplier = float(payload.get("multiplier", self.multiplier))
