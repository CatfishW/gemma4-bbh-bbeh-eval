"""Single-update on-policy trainer; all rollouts finish before any weight change."""
from __future__ import annotations

from dataclasses import asdict
import math
import random
import time
from typing import Callable, Protocol

from .core import Branch, Config, Credit, Example, Fork, Scheduler, Segment, credit, seed_for


class Backend(Protocol):
    model: object
    def prompt(self, stage: str, question: str, notes: str = "") -> tuple[int, ...]: ...
    def sample(self, context: tuple[int, ...], cap: int, temperature: float, seed: int) -> Segment: ...
    def decode(self, tokens: tuple[int, ...], final: bool = False) -> str: ...
    def log_prob(self, segment: Segment, temperature: float): ...
    def gate_log_probs(self, context: tuple[int, ...], temperature: float): ...


def branch_cost(prefix: Segment, gate_context: tuple[int, ...], segments: tuple[Segment, ...], cfg: Config) -> float:
    # Generated-token proxy plus optional prefill-equivalent price, NOT wall latency.
    generated = len(prefix.tokens) + 1 + sum(len(s.tokens) for s in segments)
    prefill = len(prefix.context) + len(gate_context) + sum(len(s.context) for s in segments)
    return generated + cfg.prefill_price * prefill


def collect(backend: Backend, example: Example, cfg: Config, scheduler: Scheduler,
            importance: float, iteration: int, ordinal: int,
            scorer: Callable[[str, str], bool]) -> Fork:
    import torch
    def sample(stage, context, cap, *parts):
        return backend.sample(context, cap, cfg.temperature,
                              seed_for(cfg.seed, iteration, ordinal, example.key, stage, *parts))
    with torch.no_grad():
        prefix = sample("prefix", backend.prompt("notes", example.question), cfg.prefix_tokens)
        notes = backend.decode(prefix.tokens)
        gate_context = backend.prompt("gate", example.question, notes)
        probabilities = tuple(float(x) for x in backend.gate_log_probs(gate_context, cfg.gate_temperature).exp())
        action = None
        if cfg.estimator == "sampled":
            rng = random.Random(seed_for(cfg.seed, iteration, ordinal, example.key, "action"))
            action = int(rng.random() >= probabilities[0])
        arms = [[], []]
        for a in (0, 1):
            if action is not None and a != action:
                continue
            for j in range(cfg.samples_per_arm):
                segments = []
                current = notes
                if a == 1:
                    extension = sample("continue", backend.prompt("continue", example.question, notes), cfg.continue_tokens, j)
                    segments.append(extension)
                    current = notes + "\n" + backend.decode(extension.tokens)
                answer = sample("answer", backend.prompt("answer", example.question, current), cfg.answer_tokens, a, j)
                segments.append(answer)
                prediction = backend.decode(answer.tokens, final=True)
                # Score only complete final outputs. Budget-ended thought blocks are
                # allowed, but a truncated answer never earns a positive reward.
                reward = float(bool(scorer(prediction, example.target)) and answer.terminated)
                arms[a].append(Branch(tuple(segments), reward, branch_cost(prefix, gate_context, tuple(segments), cfg), prediction))
    return Fork(example.key, example.task, prefix, gate_context, probabilities,
                (tuple(arms[0]), tuple(arms[1])), importance,
                scheduler.rows[example.key]["return"],
                scheduler.dual[example.task] if cfg.quality_dual else 0.0, action)


def apply_credit(backend: Backend, fork: Fork, c: Credit, normalizer: float) -> float:
    """Backward one segment at a time, avoiding retained whole-tree model graphs."""
    scale = fork.importance / normalizer
    total = 0.0
    def backward(log_probability, coefficient):
        nonlocal total
        if not math.isfinite(coefficient):
            raise FloatingPointError("nonfinite credit")
        loss = -log_probability * (scale * coefficient)
        if not loss.isfinite().all():
            raise FloatingPointError("nonfinite policy loss")
        total += float(loss.detach())
        loss.backward()
    if c.prefix:
        backward(backend.log_prob(fork.prefix, backend.config.temperature), c.prefix)
    if any(c.gate):
        logs = backend.gate_log_probs(fork.gate_context, backend.config.gate_temperature)
        # Keep the coefficients detached: differentiating p*Q here in addition
        # to log p would double count the gate derivative.
        import torch
        coefficients = torch.tensor(c.gate, dtype=logs.dtype, device=logs.device)
        backward((logs * coefficients).sum(), 1.0)
    for a, arm in enumerate(fork.arms):
        for branch, coefficient in zip(arm, c.suffixes[a]):
            if coefficient:
                for segment in branch.segments:
                    backward(backend.log_prob(segment, backend.config.temperature), coefficient)
    return total


def lora_parameters(model) -> list:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any("lora_" not in name for name, _ in named):
        raise ValueError("only explicitly named LoRA parameters may be optimized")
    return [p for _, p in named]


class Trainer:
    def __init__(self, backend: Backend, examples: list[Example], cfg: Config,
                 scorer: Callable[[str, str], bool], targets: dict[str, float] | None = None):
        import torch
        if any(e.index < 0 or e.index >= 25 for e in examples):
            raise ValueError("RL training accepts calibration indices 0..24 only")
        self.backend, self.cfg, self.scorer = backend, cfg, scorer
        self.backend.config = cfg
        backend.model.eval()  # dropout is disabled during BOTH collection and scoring
        self.parameters = lora_parameters(backend.model)
        self.optimizer = torch.optim.AdamW(self.parameters, lr=cfg.learning_rate, weight_decay=0.0)
        self.examples = {e.key: e for e in examples}
        self.scheduler = Scheduler(examples, cfg, targets)
        self.rng = random.Random(cfg.seed)
        self.iteration, self.sampled_tokens = 0, 0

    def step(self) -> tuple[dict, list[Fork]]:
        import torch
        remaining = self.cfg.max_sampled_tokens - self.sampled_tokens
        roots = min(self.cfg.roots_per_step, remaining // self.cfg.worst_tree_tokens)
        if roots == 0:
            raise StopIteration("insufficient token budget for another complete rollout tree")
        started = time.perf_counter()
        snapshot = self.scheduler.state_dict()
        draws = self.scheduler.draw(self.rng, roots)
        forks = [collect(self.backend, self.examples[k], self.cfg, self.scheduler, weight,
                         self.iteration, j, self.scorer) for j, (k, weight) in enumerate(draws)]
        if snapshot != self.scheduler.state_dict():
            raise RuntimeError("current outcomes changed the allocation snapshot")
        credits = [credit(f, self.cfg) for f in forks]
        sampled = sum(len(f.prefix.tokens) + sum(len(s.tokens) for arm in f.arms for b in arm for s in b.segments) for f in forks)
        if sampled > roots * self.cfg.worst_tree_tokens:
            raise RuntimeError("backend exceeded declared generation caps")
        collected_at = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        loss = sum(apply_credit(self.backend, f, c, roots*self.cfg.loss_normalizer) for f, c in zip(forks, credits))
        for p in self.parameters:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise FloatingPointError("nonfinite LoRA gradients; optimizer not stepped")
        norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.cfg.max_grad_norm, error_if_nonfinite=True))
        self.optimizer.step()
        if any(not torch.isfinite(p).all() for p in self.parameters):
            raise FloatingPointError("nonfinite adapter update; abort and reload prior checkpoint")
        # Only after the update may new statistics affect future allocation/credit.
        self.scheduler.update(forks, credits)
        self.sampled_tokens += sampled
        self.iteration += 1
        weighted = lambda attr: sum(f.importance*getattr(c, attr) for f, c in zip(forks, credits))/roots
        prefill = sum(len(f.prefix.context)+len(f.gate_context)+sum(len(s.context) for arm in f.arms for b in arm for s in b.segments) for f in forks)
        metrics = {"iteration": self.iteration, "roots": roots, "sampled_tokens": sampled,
                   "cumulative_sampled_tokens": self.sampled_tokens, "generation_prefill_tokens": prefill,
                   "surrogate_loss": loss, "gradient_norm_before_clip": norm,
                   "ht_train_accuracy_estimate": weighted("accuracy"),
                   "ht_expected_deployment_cost_proxy": weighted("expected_cost"),
                   "ht_train_utility_estimate": weighted("utility"),
                   "mean_stop_probability": sum(f.probabilities[0] for f in forks)/roots,
                   "max_importance": max(f.importance for f in forks),
                   "dual": dict(self.scheduler.dual),
                   "collection_seconds": collected_at-started,
                   "update_seconds": time.perf_counter()-collected_at,
                   "wall_seconds": time.perf_counter()-started,
                   "final_answer_truncations": sum(not b.segments[-1].terminated for f in forks for arm in f.arms for b in arm)}
        return metrics, forks

    def state_dict(self) -> dict:
        import torch
        return {"iteration": self.iteration, "sampled_tokens": self.sampled_tokens,
                "scheduler": self.scheduler.state_dict(), "optimizer": self.optimizer.state_dict(),
                "rng": self.rng.getstate(), "torch_rng": torch.get_rng_state(),
                "config": asdict(self.cfg)}

    def load_state_dict(self, state: dict) -> None:
        import torch
        if state["config"] != asdict(self.cfg) or not 0 <= state["iteration"] <= self.cfg.iterations:
            raise ValueError("resume config/iteration mismatch")
        if not 0 <= state["sampled_tokens"] <= self.cfg.max_sampled_tokens:
            raise ValueError("resume budget mismatch")
        self.scheduler.load_state_dict(state["scheduler"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        self.iteration, self.sampled_tokens = state["iteration"], state["sampled_tokens"]


def predict(backend: Backend, question: str, cfg: Config, seed: int = 0,
            mode: str = "sample") -> dict:
    """Deployment path; receives no target, reward, scheduler, or reference model.

    `sample` is the trained stochastic gate. `greedy`, `always-stop`, and
    `always-continue` are explicitly different deployment controls.
    """
    import torch
    if mode not in {"sample", "greedy", "always-stop", "always-continue"}:
        raise ValueError("invalid deployment gate")
    started = time.perf_counter()
    with torch.no_grad():
        prefix = backend.sample(backend.prompt("notes", question), cfg.prefix_tokens, cfg.temperature,
                                seed_for(seed, "prefix", question))
        notes = backend.decode(prefix.tokens)
        ctx = backend.prompt("gate", question, notes)
        p = tuple(float(x) for x in backend.gate_log_probs(ctx, cfg.gate_temperature).exp())
        if mode == "sample":
            action = int(random.Random(seed_for(seed, "gate", question)).random() >= p[0])
        elif mode == "greedy":
            action = int(p[1] > p[0])
        else:
            action = int(mode == "always-continue")
        segments = []
        if action:
            extra = backend.sample(backend.prompt("continue", question, notes), cfg.continue_tokens,
                                   cfg.temperature, seed_for(seed, "continue", question))
            segments.append(extra)
            notes += "\n" + backend.decode(extra.tokens)
        answer = backend.sample(backend.prompt("answer", question, notes), cfg.answer_tokens,
                                cfg.temperature, seed_for(seed, "answer", question))
        segments.append(answer)
        result = backend.decode(answer.tokens, final=True)
    return {"prediction": result, "action": "continue" if action else "stop",
            "probabilities": p, "mode": mode, "cost_proxy": branch_cost(prefix, ctx, tuple(segments), cfg),
            "generated_tokens": len(prefix.tokens) + sum(len(s.tokens) for s in segments),
            "prefill_tokens": len(prefix.context)+len(ctx)+sum(len(s.context) for s in segments),
            "answer_terminated": answer.terminated, "elapsed_seconds": time.perf_counter()-started,
            "segments": [asdict(prefix), *[asdict(s) for s in segments]]}
