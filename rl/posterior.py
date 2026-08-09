"""Discounted hierarchical Beta posteriors and variance-optimal rollout allocation.

The tracker maintains, per training prompt, a discounted Beta posterior over the
current-policy success probability, partially pooled through a task-level prior.
A frozen `PosteriorSnapshot` taken before each iteration supplies

- the predictable baseline  b_i = E[p_i]
- the allocation score      s_i = sqrt(E[p_i (1 - p_i)])
- length statistics for cost estimates and shaped-reward baselines.

Predictability (the snapshot never sees the current iteration's outcomes) is
what keeps the policy-gradient increments conditionally unbiased under
difficulty-adaptive allocation; see paper/volt/method.md, Proposition 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path


@dataclass
class PromptState:
    task: str
    wins: float = 0.0
    losses: float = 0.0
    length_sum: float = 0.0
    length_count: float = 0.0
    correct_length_sum: float = 0.0
    correct_length_count: float = 0.0
    last_sampled_iteration: int = -1


@dataclass(frozen=True)
class PromptPosterior:
    baseline: float
    allocation_score: float
    mean_length: float
    mean_correct_length: float
    observation_mass: float


@dataclass(frozen=True)
class PosteriorSnapshot:
    iteration: int
    prompts: dict[str, PromptPosterior]
    task_means: dict[str, float]

    def baseline(self, prompt: str) -> float:
        return self.prompts[prompt].baseline

    def allocation_score(self, prompt: str) -> float:
        return self.prompts[prompt].allocation_score


def beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)

def expected_bernoulli_variance(alpha: float, beta: float) -> float:
    """E[p(1-p)] under Beta(alpha, beta): alpha*beta / ((a+b) * (a+b+1))."""
    total = alpha + beta
    return (alpha * beta) / (total * total + total)


class DifficultyTracker:
    def __init__(
        self,
        prompt_tasks: dict[str, str],
        discount: float = 0.92,
        prior_strength: float = 4.0,
        default_length: float = 160.0,
    ) -> None:
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if prior_strength <= 0.0:
            raise ValueError("prior_strength must be positive")
        self.discount = discount
        self.prior_strength = prior_strength
        self.default_length = default_length
        self.iteration = 0
        self.states: dict[str, PromptState] = {
            prompt: PromptState(task=task) for prompt, task in prompt_tasks.items()
        }

    def task_posterior_means(self) -> dict[str, float]:
        """Task-level pooled success means with a Beta(1, 1) smoother."""
        wins: dict[str, float] = {}
        losses: dict[str, float] = {}
        for state in self.states.values():
            wins[state.task] = wins.get(state.task, 0.0) + state.wins
            losses[state.task] = losses.get(state.task, 0.0) + state.losses
        return {
            task: (wins[task] + 1.0) / (wins[task] + losses[task] + 2.0)
            for task in wins
        }

    def _prompt_alpha_beta(self, state: PromptState, task_mean: float) -> tuple[float, float]:
        alpha = state.wins + self.prior_strength * task_mean
        beta = state.losses + self.prior_strength * (1.0 - task_mean)
        return alpha, beta

    def _task_length_means(self, correct_only: bool) -> dict[str, float]:
        total: dict[str, float] = {}
        count: dict[str, float] = {}
        for state in self.states.values():
            if correct_only:
                total[state.task] = total.get(state.task, 0.0) + state.correct_length_sum
                count[state.task] = count.get(state.task, 0.0) + state.correct_length_count
            else:
                total[state.task] = total.get(state.task, 0.0) + state.length_sum
                count[state.task] = count.get(state.task, 0.0) + state.length_count
        return {
            task: (total[task] / count[task]) if count.get(task, 0.0) > 0 else self.default_length
            for task in total
        }

    def snapshot(self) -> PosteriorSnapshot:
        task_means = self.task_posterior_means()
        task_lengths = self._task_length_means(correct_only=False)
        task_correct_lengths = self._task_length_means(correct_only=True)
        prompts: dict[str, PromptPosterior] = {}
        for prompt, state in self.states.items():
            alpha, beta = self._prompt_alpha_beta(state, task_means[state.task])
            mean_length = (
                state.length_sum / state.length_count
                if state.length_count > 0
                else task_lengths[state.task]
            )
            mean_correct_length = (
                state.correct_length_sum / state.correct_length_count
                if state.correct_length_count > 0
                else task_correct_lengths[state.task]
            )
            prompts[prompt] = PromptPosterior(
                baseline=beta_mean(alpha, beta),
                allocation_score=math.sqrt(expected_bernoulli_variance(alpha, beta)),
                mean_length=mean_length,
                mean_correct_length=mean_correct_length,
                observation_mass=state.wins + state.losses,
            )
        return PosteriorSnapshot(iteration=self.iteration, prompts=prompts, task_means=task_means)

    def update(self, outcomes: dict[str, list[tuple[int, int]]]) -> None:
        """Advance one iteration.

        outcomes maps prompt id -> list of (reward, completion_length) pairs
        observed in the iteration that just finished. All states decay by the
        discount; sampled states then absorb the fresh evidence.
        """
        length_decay = 0.9
        for prompt, state in self.states.items():
            state.wins *= self.discount
            state.losses *= self.discount
            rollouts = outcomes.get(prompt)
            if not rollouts:
                continue
            state.last_sampled_iteration = self.iteration
            state.length_sum = state.length_sum * length_decay + sum(
                float(length) for _, length in rollouts
            )
            state.length_count = state.length_count * length_decay + len(rollouts)
            correct = [(reward, length) for reward, length in rollouts if reward]
            if correct:
                state.correct_length_sum = state.correct_length_sum * length_decay + sum(
                    float(length) for _, length in correct
                )
                state.correct_length_count = (
                    state.correct_length_count * length_decay + len(correct)
                )
            state.wins += sum(1.0 for reward, _ in rollouts if reward)
            state.losses += sum(1.0 for reward, _ in rollouts if not reward)
        self.iteration += 1

    def least_recently_sampled(self) -> list[str]:
        return sorted(
            self.states,
            key=lambda prompt: (self.states[prompt].last_sampled_iteration, prompt),
        )

    def to_json(self) -> dict:
        return {
            "iteration": self.iteration,
            "discount": self.discount,
            "prior_strength": self.prior_strength,
            "default_length": self.default_length,
            "states": {
                prompt: {
                    "task": state.task,
                    "wins": state.wins,
                    "losses": state.losses,
                    "length_sum": state.length_sum,
                    "length_count": state.length_count,
                    "correct_length_sum": state.correct_length_sum,
                    "correct_length_count": state.correct_length_count,
                    "last_sampled_iteration": state.last_sampled_iteration,
                }
                for prompt, state in self.states.items()
            },
        }

    @classmethod
    def from_json(cls, payload: dict) -> "DifficultyTracker":
        tracker = cls(
            prompt_tasks={
                prompt: raw["task"] for prompt, raw in payload["states"].items()
            },
            discount=payload["discount"],
            prior_strength=payload["prior_strength"],
            default_length=payload["default_length"],
        )
        tracker.iteration = payload["iteration"]
        for prompt, raw in payload["states"].items():
            state = tracker.states[prompt]
            state.wins = raw["wins"]
            state.losses = raw["losses"]
            state.length_sum = raw["length_sum"]
            state.length_count = raw["length_count"]
            state.correct_length_sum = raw["correct_length_sum"]
            state.correct_length_count = raw["correct_length_count"]
            state.last_sampled_iteration = raw["last_sampled_iteration"]
        return tracker

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "DifficultyTracker":
        return cls.from_json(json.loads(path.read_text()))


def allocate_rollouts(
    scores: dict[str, float],
    budget: int,
    n_max: int,
    floor_fraction: float,
    least_recently_sampled: list[str],
) -> dict[str, int]:
    """Water-fill `budget` rollouts proportionally to allocation scores.

    A reserved floor (floor_fraction of the budget, one rollout each) goes to the
    least-recently-sampled prompts so every posterior keeps receiving evidence;
    the remainder is proportional to scores subject to the per-prompt cap n_max.
    """
    if budget <= 0:
        return {}
    if n_max <= 0:
        raise ValueError("n_max must be positive")
    allocation = {prompt: 0 for prompt in scores}

    floor_budget = min(int(round(floor_fraction * budget)), budget, len(scores))
    for prompt in least_recently_sampled[:floor_budget]:
        allocation[prompt] += 1
    remaining = budget - floor_budget

    while remaining > 0:
        active = {
            prompt: score
            for prompt, score in scores.items()
            if allocation[prompt] < n_max and score > 0.0
        }
        if not active:
            for prompt in sorted(allocation):
                if remaining <= 0:
                    break
                headroom = n_max - allocation[prompt]
                if headroom > 0:
                    grant = min(headroom, remaining)
                    allocation[prompt] += grant
                    remaining -= grant
            break
        total_score = sum(active.values())
        quotas = {
            prompt: remaining * score / total_score for prompt, score in active.items()
        }
        granted = 0
        fractional: list[tuple[float, str]] = []
        for prompt, quota in quotas.items():
            headroom = n_max - allocation[prompt]
            base = min(int(quota), headroom)
            allocation[prompt] += base
            granted += base
            if base < headroom:
                fractional.append((quota - int(quota), prompt))
        remaining -= granted
        if remaining <= 0:
            break
        fractional.sort(key=lambda pair: (-pair[0], pair[1]))
        progressed = False
        for _, prompt in fractional:
            if remaining <= 0:
                break
            if allocation[prompt] < n_max:
                allocation[prompt] += 1
                remaining -= 1
                progressed = True
        if not progressed and granted == 0:
            break

    return {prompt: count for prompt, count in allocation.items() if count > 0}


def allocation_entropy(allocation: dict[str, int]) -> float:
    total = sum(allocation.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in allocation.values():
        if count > 0:
            share = count / total
            entropy -= share * math.log(share)
    return entropy
