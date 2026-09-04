"""CRAFT: counterfactual stopping, cross-fitted credit, and predictable sampling.

No dependency on torch, model weights, benchmark labels or test data at import.
The gradient estimator and its assumptions are derived in docs/CRAFT_RL.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Mapping


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seed_for(seed: int, *parts: object) -> int:
    return int(digest([seed, *parts])[:15], 16) % (2**31 - 1)


@dataclass(frozen=True)
class Config:
    iterations: int = 48
    roots_per_step: int = 8
    samples_per_arm: int = 2
    prefix_tokens: int = 32
    continue_tokens: int = 96
    answer_tokens: int = 32
    max_context: int = 4096
    temperature: float = 0.8
    gate_temperature: float = 1.0
    cost_scale: float = 160.0
    cost_weight: float = 0.2
    prefill_price: float = 0.0
    target_accuracy: float = 0.4
    dual_lr: float = 0.05
    dual_max: float = 8.0
    exploration: float = 0.2
    ema_decay: float = 0.9
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    loss_normalizer: float = 1.0
    rank: int = 32
    alpha: int = 64
    seed: int = 20260904
    max_sampled_tokens: int = 200000
    estimator: str = "paired"  # paired counterfactual or ordinary sampled-action PG
    suffix_baseline: str = "crossfit"  # crossfit or historical-prompt baseline
    adaptive_allocation: bool = True
    quality_dual: bool = True

    def __post_init__(self) -> None:
        positive = ("iterations", "roots_per_step", "samples_per_arm", "prefix_tokens",
                    "continue_tokens", "answer_tokens", "max_context", "rank", "alpha", "max_sampled_tokens")
        for name in positive:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("invalid seed")
        for name in ("adaptive_allocation", "quality_dual"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("temperature", "gate_temperature", "cost_scale", "learning_rate",
                     "max_grad_norm", "loss_normalizer"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.exploration <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("invalid exploration or EMA")
        if not 0 <= self.target_accuracy <= 1:
            raise ValueError("target accuracy must be a probability")
        if min(self.cost_weight, self.prefill_price, self.dual_lr, self.dual_max) < 0:
            raise ValueError("cost/dual settings must be nonnegative")
        if self.estimator not in {"paired", "sampled"} or self.suffix_baseline not in {"crossfit", "history"}:
            raise ValueError("unknown estimator/baseline")
        if self.estimator == "sampled" and self.samples_per_arm != 1:
            raise ValueError("sampled-action control requires samples_per_arm=1")
        if self.max_context > 131072 or self.samples_per_arm > 16 or self.roots_per_step > 256:
            raise ValueError("reference implementation resource limit")

    @property
    def worst_tree_tokens(self) -> int:
        if self.estimator == "sampled":
            return self.prefix_tokens + self.continue_tokens + self.answer_tokens
        return self.prefix_tokens + self.samples_per_arm * (self.continue_tokens + 2 * self.answer_tokens)


@dataclass(frozen=True)
class Example:
    key: str
    task: str
    question: str
    target: str
    index: int


@dataclass(frozen=True)
class Segment:
    context: tuple[int, ...]
    tokens: tuple[int, ...]
    terminated: bool

    def __post_init__(self) -> None:
        if not self.context or not self.tokens:
            raise ValueError("segments must contain context and sampled actions")
        if any(type(t) is not int or t < 0 for t in (*self.context, *self.tokens)):
            raise ValueError("tokens must be nonnegative integers")


@dataclass(frozen=True)
class Branch:
    segments: tuple[Segment, ...]
    reward: float
    cost: float
    prediction: str

    def __post_init__(self) -> None:
        if not self.segments or self.reward not in (0.0, 1.0):
            raise ValueError("binary outcome and nonempty trajectory required")
        if not math.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("invalid cost")


@dataclass(frozen=True)
class Fork:
    key: str
    task: str
    prefix: Segment
    gate_context: tuple[int, ...]
    probabilities: tuple[float, float]  # stop, continue; behavior snapshot
    arms: tuple[tuple[Branch, ...], tuple[Branch, ...]]
    importance: float
    historical_baseline: float
    multiplier: float
    selected_action: int | None = None


@dataclass(frozen=True)
class Credit:
    prefix: float
    gate: tuple[float, float]
    suffixes: tuple[tuple[float, ...], tuple[float, ...]]
    utility: float
    accuracy: float
    expected_cost: float
    leverage: float


def credit(fork: Fork, cfg: Config) -> Credit:
    """Detached coefficients for an exact on-policy, two-arm score-function estimator.

    Gate coefficients multiply log probabilities. In paired mode these are p*Q,
    NOT Q alone. Prefix receives the mixture return once, never once per leaf.
    A suffix baseline excludes the entire trajectory receiving the advantage.
    """
    p = fork.probabilities
    if len(p) != 2 or any(not math.isfinite(x) or x <= 0 for x in p) or abs(sum(p) - 1) > 1e-6:
        raise ValueError("strictly positive normalized gate probabilities required")
    if not math.isfinite(fork.importance) or fork.importance <= 0 or fork.multiplier < 0:
        raise ValueError("invalid importance or dual multiplier")
    utilities = tuple(tuple((1 + fork.multiplier) * b.reward - cfg.cost_weight * b.cost / cfg.cost_scale
                            for b in arm) for arm in fork.arms)
    if fork.selected_action is not None:
        a = fork.selected_action
        if a not in (0, 1) or len(fork.arms[a]) != 1 or fork.arms[1-a]:
            raise ValueError("sampled estimator needs exactly one selected trajectory")
        u = utilities[a][0]
        adv = u - fork.historical_baseline
        g = (adv, 0.0) if a == 0 else (0.0, adv)
        s = ((adv,), ()) if a == 0 else ((), (adv,))
        b = fork.arms[a][0]
        return Credit(adv, g, s, u, b.reward, b.cost, adv * adv)
    if any(len(arm) != cfg.samples_per_arm for arm in fork.arms):
        raise ValueError("both counterfactual arms require the declared independent samples")
    means = tuple(sum(us) / len(us) for us in utilities)
    mixture = sum(p[a] * means[a] for a in (0, 1))
    # Center the gate by its mixture: identical gradient, smaller coefficients.
    gate = tuple(p[a] * (means[a] - mixture) for a in (0, 1))
    all_u = [u for arm in utilities for u in arm]
    suffixes = []
    for a, us in enumerate(utilities):
        coeffs = []
        for u in us:
            baseline = ((sum(all_u) - u) / (len(all_u) - 1)
                        if cfg.suffix_baseline == "crossfit" else fork.historical_baseline)
            coeffs.append(p[a] * (u - baseline) / len(us))
        suffixes.append(tuple(coeffs))
    accuracy = sum(p[a] * sum(b.reward for b in fork.arms[a]) / len(fork.arms[a]) for a in (0, 1))
    cost = sum(p[a] * sum(b.cost for b in fork.arms[a]) / len(fork.arms[a]) for a in (0, 1))
    within = sum(p[a] * sum((u-means[a])**2 for u in utilities[a]) / len(utilities[a]) for a in (0, 1))
    leverage = p[0] * p[1] * (means[1] - means[0])**2 + within
    return Credit(mixture - fork.historical_baseline, gate, tuple(suffixes),
                  mixture, accuracy, cost, leverage)


class Scheduler:
    """Macro-task objective, with-replacement proposals and exact p/q correction.

    All draws and importance weights are fixed before fresh outcomes are observed.
    Adaptive leverage is a heuristic proposal, not a proven optimal allocator.
    """
    def __init__(self, examples: list[Example], cfg: Config, targets: dict[str, float] | None = None):
        if not examples or len({e.key for e in examples}) != len(examples):
            raise ValueError("empty/duplicate training identities")
        self.cfg = cfg
        self.tasks = {e.key: e.task for e in examples}
        counts = {task: sum(e.task == task for e in examples) for task in set(self.tasks.values())}
        self.target = {e.key: 1 / (len(counts) * counts[e.task]) for e in examples}
        self.rows = {e.key: {"return": 0.0, "leverage": 1.0, "cost": 1.0, "visits": 0} for e in examples}
        self.dual = {task: 0.0 for task in counts}
        self.targets = dict(targets) if targets is not None else {t: cfg.target_accuracy for t in counts}
        if set(self.targets) != set(counts) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.targets.values()):
            raise ValueError("task targets must cover exactly the training tasks with probabilities")

    def proposal(self) -> dict[str, float]:
        if not self.cfg.adaptive_allocation:
            return dict(self.target)
        raw = {k: self.target[k] * math.sqrt(max(v["leverage"], 1e-8) / max(v["cost"], 1e-8))
               for k, v in self.rows.items()}
        total = sum(raw.values())
        eps = self.cfg.exploration
        return {k: eps*self.target[k] + (1-eps)*v/total for k, v in raw.items()}

    def draw(self, rng: random.Random, n: int) -> list[tuple[str, float]]:
        if type(n) is not int or n < 1:
            raise ValueError("positive draw count required")
        q = self.proposal()
        keys = sorted(q)
        return [(k, self.target[k]/q[k]) for k in rng.choices(keys, [q[k] for k in keys], k=n)]

    def update(self, forks: list[Fork], credits: list[Credit]) -> None:
        if not forks or len(forks) != len(credits):
            raise ValueError("invalid update batch")
        grouped: dict[str, list[Credit]] = {}
        task_error = {t: 0.0 for t in self.dual}
        for f, c in zip(forks, credits):
            grouped.setdefault(f.key, []).append(c)
            # Horvitz-Thompson task-conditional constraint gradient; no random
            # per-task sample-count denominator and no current-reward routing.
            task_error[f.task] += f.importance * (self.targets[f.task]-c.accuracy) * len(self.dual) / len(forks)
        d = self.cfg.ema_decay
        for key, cs in grouped.items():
            row = self.rows[key]
            for field, attr in (("return", "utility"), ("leverage", "leverage"), ("cost", "expected_cost")):
                row[field] = d*row[field] + (1-d)*sum(getattr(c, attr) for c in cs)/len(cs)
            row["visits"] += len(cs)
        if self.cfg.quality_dual:
            for task, error in task_error.items():
                self.dual[task] = min(self.cfg.dual_max, max(0.0, self.dual[task] + self.cfg.dual_lr*error))

    def state_dict(self) -> dict:
        return json.loads(json.dumps({"rows": self.rows, "dual": self.dual, "targets": self.targets}, allow_nan=False))

    def load_state_dict(self, state: Mapping) -> None:
        if set(state) != {"rows", "dual", "targets"} or state["targets"] != self.targets or set(state["rows"]) != set(self.rows) or set(state["dual"]) != set(self.dual):
            raise ValueError("scheduler identities changed")
        candidate = json.loads(json.dumps(state, allow_nan=False))
        for row in candidate["rows"].values():
            if set(row) != {"return", "leverage", "cost", "visits"} or any(not isinstance(v, (float, int)) for v in row.values()):
                raise ValueError("invalid scheduler state")
            if row["cost"] <= 0 or min(row["leverage"], row["visits"]) < 0:
                raise ValueError("invalid scheduler moments")
        if any(not 0 <= v <= self.cfg.dual_max for v in candidate["dual"].values()):
            raise ValueError("invalid dual state")
        self.rows, self.dual = candidate["rows"], candidate["dual"]
