"""RL trainer: VOLT (variance-optimal allocation + predictable baselines) and
GRPO / Dr. GRPO baselines through one code path.

One optimizer step per iteration over freshly sampled rollouts (strictly
on-policy REINFORCE with the mode's baseline), token-level loss with a constant
length normalizer, LoRA-only updates, resumable state.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import random
import statistics

import torch

from eval_benchmarks import Example
from rl.configs import TrainConfig
from rl.modeling import completion_log_probs
from rl.posterior import (
    DifficultyTracker,
    PosteriorSnapshot,
    allocate_rollouts,
    allocation_entropy,
)
from rl.protocol import prompt_id
from rl.rewards import LengthDualController, correctness_reward
from rl.rollout import (
    RolloutRequest,
    RolloutResult,
    chat_prompt_token_ids,
    generate_rollouts,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ScoredRollout:
    result: RolloutResult
    reward: int
    shaped_reward: float
    advantage: float


class RLTrainer:
    def __init__(
        self,
        config: TrainConfig,
        model,
        tokenizer,
        train_examples: list[Example],
        val_examples: list[Example],
        device: str = "cuda:0",
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.examples: dict[str, Example] = {}
        self.prompt_ids: list[str] = []
        self.prompt_token_cache: dict[str, list[int]] = {}
        dropped = 0
        for example in train_examples:
            pid = prompt_id(example)
            token_ids = chat_prompt_token_ids(tokenizer, example, config.prompt_strategy)
            if len(token_ids) > config.max_prompt_tokens:
                dropped += 1
                continue
            self.examples[pid] = example
            self.prompt_ids.append(pid)
            self.prompt_token_cache[pid] = token_ids
        LOGGER.info(
            "training pool: %d prompts (%d dropped over %d tokens)",
            len(self.prompt_ids),
            dropped,
            config.max_prompt_tokens,
        )

        self.tracker = DifficultyTracker(
            prompt_tasks={
                pid: f"{self.examples[pid].benchmark}/{self.examples[pid].task}"
                for pid in self.prompt_ids
            },
            discount=config.discount,
            prior_strength=config.prior_strength,
        )
        self.dual = LengthDualController(config.length_shaping)
        self.iteration = 0
        self.cumulative_generated_tokens = 0
        self.rng = random.Random(config.seed)
        self.grpo_cursor: list[str] = []

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )
        self.val_probe_examples = self._fixed_val_probe(val_examples)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.samples_path = self.output_dir / "samples.jsonl"

    # ----- selection ------------------------------------------------------

    def _fixed_val_probe(self, val_examples: list[Example]) -> list[Example]:
        usable = [
            example
            for example in val_examples
            if len(chat_prompt_token_ids(self.tokenizer, example, self.config.prompt_strategy))
            <= self.config.max_prompt_tokens
        ]
        probe_rng = random.Random(self.config.seed + 1)
        probe_rng.shuffle(usable)
        return usable[: self.config.val_probe_size]

    def _volt_allocation(self, snapshot: PosteriorSnapshot) -> dict[str, int]:
        if self.config.uniform_allocation:
            per_prompt = max(1, self.config.rollout_budget // len(self.prompt_ids))
            chosen = self.tracker.least_recently_sampled()
            allocation: dict[str, int] = {}
            budget = self.config.rollout_budget
            for pid in chosen:
                if budget <= 0:
                    break
                take = min(per_prompt, budget)
                allocation[pid] = take
                budget -= take
            return allocation
        scores = {pid: snapshot.allocation_score(pid) for pid in self.prompt_ids}
        return allocate_rollouts(
            scores=scores,
            budget=self.config.rollout_budget,
            n_max=self.config.n_max,
            floor_fraction=self.config.floor_fraction,
            least_recently_sampled=self.tracker.least_recently_sampled(),
        )

    def _grpo_allocation(self, groups: int | None = None) -> dict[str, int]:
        if groups is None:
            groups = max(1, self.config.rollout_budget // self.config.group_size)
        allocation: dict[str, int] = {}
        while len(allocation) < groups:
            if not self.grpo_cursor:
                self.grpo_cursor = list(self.prompt_ids)
                self.rng.shuffle(self.grpo_cursor)
            pid = self.grpo_cursor.pop()
            if pid not in allocation:
                allocation[pid] = self.config.group_size
        return allocation

    def _generate_for_allocation(self, allocation: dict[str, int]) -> list[RolloutResult]:
        requests: list[RolloutRequest] = []
        for pid, count in sorted(allocation.items()):
            for _ in range(count):
                requests.append(
                    RolloutRequest(
                        prompt_id=pid,
                        example=self.examples[pid],
                        prompt_token_ids=self.prompt_token_cache[pid],
                    )
                )
        rollouts = generate_rollouts(
            self.model,
            self.tokenizer,
            requests,
            iteration=self.iteration,
            seed=self.config.seed,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
            batch_size=self.config.generation_batch_size,
            max_batch_tokens=self.config.generation_max_batch_tokens,
            device=self.device,
        )
        self.cumulative_generated_tokens += sum(len(r.completion_token_ids) for r in rollouts)
        torch.cuda.empty_cache()
        return rollouts

    def _collect_rollouts(
        self, snapshot: PosteriorSnapshot
    ) -> tuple[dict[str, int], list[RolloutResult], list[RolloutResult]]:
        """Returns (allocation, rollouts used for training, all rollouts generated).

        For DAPO-style dynamic sampling (mode grpo_ds), degenerate groups are
        discarded and replacement groups sampled in waves; discarded tokens
        still count toward the generation budget, which is exactly the
        inefficiency the method is known for.
        """
        if self.config.mode == "volt":
            allocation = self._volt_allocation(snapshot)
            rollouts = self._generate_for_allocation(allocation)
            return allocation, rollouts, rollouts
        if self.config.mode in ("grpo", "drgrpo"):
            allocation = self._grpo_allocation()
            rollouts = self._generate_for_allocation(allocation)
            return allocation, rollouts, rollouts

        if self.config.mode != "grpo_ds":
            raise ValueError(f"unknown mode {self.config.mode}")
        target_groups = max(1, self.config.rollout_budget // self.config.group_size)
        max_extra_waves = 3
        kept: list[RolloutResult] = []
        kept_allocation: dict[str, int] = {}
        everything: list[RolloutResult] = []
        for _ in range(1 + max_extra_waves):
            missing = target_groups - len(kept_allocation)
            if missing <= 0:
                break
            wave_allocation = self._grpo_allocation(groups=missing)
            wave_rollouts = self._generate_for_allocation(wave_allocation)
            everything.extend(wave_rollouts)
            by_prompt: dict[str, list[RolloutResult]] = {}
            for rollout in wave_rollouts:
                by_prompt.setdefault(rollout.prompt_id, []).append(rollout)
            for pid, group in by_prompt.items():
                rewards = {
                    correctness_reward(r.completion_text, r.example.target) for r in group
                }
                if len(rewards) > 1:
                    kept.extend(group)
                    kept_allocation[pid] = len(group)
        return kept_allocation, kept, everything

    # ----- advantages -----------------------------------------------------

    def _score_rollouts(
        self, rollouts: list[RolloutResult], snapshot: PosteriorSnapshot
    ) -> list[ScoredRollout]:
        scored: list[ScoredRollout] = []
        if self.config.mode == "volt":
            for rollout in rollouts:
                reward = correctness_reward(rollout.completion_text, rollout.example.target)
                shaped = self.dual.shaped_reward(reward, len(rollout.completion_token_ids))
                posterior = snapshot.prompts[rollout.prompt_id]
                baseline = self.dual.shaped_baseline(
                    posterior.baseline, posterior.mean_correct_length
                )
                scored.append(
                    ScoredRollout(
                        result=rollout,
                        reward=reward,
                        shaped_reward=shaped,
                        advantage=shaped - baseline,
                    )
                )
            return scored

        by_prompt: dict[str, list[RolloutResult]] = {}
        for rollout in rollouts:
            by_prompt.setdefault(rollout.prompt_id, []).append(rollout)
        for _, group in sorted(by_prompt.items()):
            rewards = [
                correctness_reward(rollout.completion_text, rollout.example.target)
                for rollout in group
            ]
            mean = statistics.fmean(rewards)
            std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            for rollout, reward in zip(group, rewards):
                advantage = reward - mean
                if self.config.mode in ("grpo", "grpo_ds"):
                    advantage = advantage / (std + 1e-4) if std > 0 else 0.0
                scored.append(
                    ScoredRollout(
                        result=rollout,
                        reward=reward,
                        shaped_reward=float(reward),
                        advantage=advantage,
                    )
                )
        return scored

    # ----- optimization ---------------------------------------------------

    def _train_microbatches(
        self, scored: list[ScoredRollout], microbatch_token_cap: int
    ) -> list[list[ScoredRollout]]:
        informative = [s for s in scored if abs(s.advantage) > 1e-6 and s.result.completion_token_ids]
        informative.sort(
            key=lambda s: len(s.result.prompt_token_ids) + len(s.result.completion_token_ids)
        )
        batches: list[list[ScoredRollout]] = []
        current: list[ScoredRollout] = []
        current_longest = 0
        for item in informative:
            length = len(item.result.prompt_token_ids) + len(item.result.completion_token_ids)
            longest = max(current_longest, length)
            if current and longest * (len(current) + 1) > microbatch_token_cap:
                batches.append(current)
                current = []
                longest = length
            current.append(item)
            current_longest = longest
        if current:
            batches.append(current)
        return batches

    def _optimize(self, scored: list[ScoredRollout]) -> dict:
        """One optimizer step over the iteration's rollouts.

        On CUDA OOM (shared GPU, fluctuating co-tenant usage) the whole step is
        redone from zeroed gradients with a halved microbatch token cap, so no
        partial backward can double-count.
        """
        cap = self.config.train_tokens_per_microbatch
        attempts = 0
        while True:
            try:
                return self._optimize_once(scored, cap)
            except torch.OutOfMemoryError:
                attempts += 1
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                cap = max(512, cap // 2)
                if attempts > 3:
                    raise
                LOGGER.warning(
                    "optimize OOM: retrying iteration %d with microbatch cap %d",
                    self.iteration,
                    cap,
                )

    def _optimize_once(self, scored: list[ScoredRollout], microbatch_token_cap: int) -> dict:
        self.model.train()
        total_rollouts = len(scored)
        normalizer = float(total_rollouts * self.config.loss_length_normalizer)
        loss_sum = 0.0
        token_count = 0
        microbatches = self._train_microbatches(scored, microbatch_token_cap)
        self.optimizer.zero_grad(set_to_none=True)
        for microbatch in microbatches:
            pad_id = self.tokenizer.pad_token_id
            longest = max(
                len(s.result.prompt_token_ids) + len(s.result.completion_token_ids)
                for s in microbatch
            )
            input_ids = torch.full((len(microbatch), longest), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(microbatch), longest), dtype=torch.long)
            completion_mask = torch.zeros((len(microbatch), longest), dtype=torch.long)
            advantages = torch.zeros(len(microbatch), dtype=torch.float32)
            for row, item in enumerate(microbatch):
                prompt = item.result.prompt_token_ids
                completion = item.result.completion_token_ids
                sequence = prompt + completion
                input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
                attention_mask[row, : len(sequence)] = 1
                completion_mask[row, len(prompt) : len(sequence)] = 1
                advantages[row] = item.advantage
                token_count += len(completion)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            completion_mask = completion_mask.to(self.device)
            advantages = advantages.to(self.device)

            log_probs = completion_log_probs(
                self.model, input_ids, attention_mask, completion_mask
            )
            loss = -(advantages * log_probs).sum() / normalizer
            loss.backward()
            loss_sum += loss.item()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.max_grad_norm,
        )
        learning_rate = self._learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        return {
            "peak_memory_gb": round(peak_memory_gb, 2),
            "loss": loss_sum,
            "grad_norm": float(gradient_norm),
            "lr": learning_rate,
            "trained_rollouts": sum(len(batch) for batch in microbatches),
            "trained_tokens": token_count,
        }

    def _learning_rate(self) -> float:
        warmup = max(1, self.config.warmup_iterations)
        scale = min(1.0, (self.iteration + 1) / warmup)
        return self.config.learning_rate * scale

    # ----- evaluation probe -------------------------------------------------

    def _val_probe(self) -> dict:
        requests = [
            RolloutRequest(
                prompt_id=prompt_id(example),
                example=example,
                prompt_token_ids=chat_prompt_token_ids(
                    self.tokenizer, example, self.config.prompt_strategy
                ),
            )
            for example in self.val_probe_examples
        ]
        results = generate_rollouts(
            self.model,
            self.tokenizer,
            requests,
            iteration=self.iteration,
            seed=self.config.seed + 7,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=self.config.val_probe_max_new_tokens,
            batch_size=self.config.generation_batch_size,
            max_batch_tokens=self.config.generation_max_batch_tokens,
            device=self.device,
        )
        correct = sum(
            correctness_reward(result.completion_text, result.example.target)
            for result in results
        )
        lengths = [len(result.completion_token_ids) for result in results]
        return {
            "val_probe_correct": correct,
            "val_probe_total": len(results),
            "val_probe_accuracy": correct / len(results) if results else 0.0,
            "val_probe_mean_completion_tokens": statistics.fmean(lengths) if lengths else 0.0,
        }

    # ----- persistence ------------------------------------------------------

    def save_checkpoint(self, tag: str) -> None:
        checkpoint_dir = self.output_dir / f"checkpoint-{tag}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir / "adapter")
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        self.tracker.save(checkpoint_dir / "posterior.json")
        state = {
            "iteration": self.iteration,
            "cumulative_generated_tokens": self.cumulative_generated_tokens,
            "dual": self.dual.state(),
            "grpo_cursor": self.grpo_cursor,
            "rng_state": self.rng.getstate(),
        }
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(state, default=list) + "\n"
        )
        (self.output_dir / "latest_checkpoint.txt").write_text(f"checkpoint-{tag}\n")

    def resume_if_available(self) -> bool:
        pointer = self.output_dir / "latest_checkpoint.txt"
        if not pointer.exists():
            return False
        checkpoint_dir = self.output_dir / pointer.read_text().strip()
        state = json.loads((checkpoint_dir / "trainer_state.json").read_text())
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_weights = load_file(
            checkpoint_dir / "adapter" / "adapter_model.safetensors", device="cpu"
        )
        set_peft_model_state_dict(self.model, adapter_weights)
        self.optimizer.load_state_dict(
            torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu")
        )
        self.tracker = DifficultyTracker.load(checkpoint_dir / "posterior.json")
        self.dual.load_state(state["dual"])
        self.iteration = state["iteration"]
        self.cumulative_generated_tokens = state["cumulative_generated_tokens"]
        self.grpo_cursor = list(state.get("grpo_cursor", []))
        rng_state = state.get("rng_state")
        if rng_state:
            self.rng.setstate(
                (rng_state[0], tuple(rng_state[1]), rng_state[2])
            )
        LOGGER.info("resumed from %s at iteration %d", checkpoint_dir.name, self.iteration)
        return True

    def _log_metrics(self, payload: dict) -> None:
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self.metrics_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _log_samples(self, scored: list[ScoredRollout]) -> None:
        keep = self.config.log_samples_per_iteration
        if keep <= 0:
            return
        with self.samples_path.open("a") as handle:
            for item in scored[:keep]:
                handle.write(
                    json.dumps(
                        {
                            "iteration": self.iteration,
                            "prompt_id": item.result.prompt_id,
                            "reward": item.reward,
                            "advantage": round(item.advantage, 4),
                            "completion": item.result.completion_text[:2000],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # ----- main loop --------------------------------------------------------

    def train(self) -> None:
        while self.iteration < self.config.iterations:
            snapshot = self.tracker.snapshot()
            allocation, rollouts, everything = self._collect_rollouts(snapshot)
            generated_tokens = sum(len(r.completion_token_ids) for r in everything)

            scored = self._score_rollouts(rollouts, snapshot)
            optimization = self._optimize(scored)

            outcomes: dict[str, list[tuple[int, int]]] = {}
            correct_lengths: list[int] = []
            for rollout in everything:
                reward = correctness_reward(rollout.completion_text, rollout.example.target)
                outcomes.setdefault(rollout.prompt_id, []).append(
                    (reward, len(rollout.completion_token_ids))
                )
                if reward:
                    correct_lengths.append(len(rollout.completion_token_ids))
            self.tracker.update(outcomes)
            self.dual.update(statistics.fmean(correct_lengths) if correct_lengths else None)

            rewards = [item.reward for item in scored]
            metrics = {
                "iteration": self.iteration,
                "mode": self.config.mode,
                "rollouts": len(scored),
                "distinct_prompts": len(allocation),
                "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
                "nonzero_advantage_fraction": (
                    sum(1 for item in scored if abs(item.advantage) > 1e-6) / len(scored)
                    if scored
                    else 0.0
                ),
                "allocation_entropy": allocation_entropy(allocation),
                "mean_completion_tokens": (
                    statistics.fmean(len(item.result.completion_token_ids) for item in scored)
                    if scored
                    else 0.0
                ),
                "generated_tokens": generated_tokens,
                "cumulative_generated_tokens": self.cumulative_generated_tokens,
                "length_multiplier": self.dual.multiplier,
                **optimization,
            }
            if (
                self.config.val_probe_every > 0
                and (self.iteration + 1) % self.config.val_probe_every == 0
            ):
                metrics.update(self._val_probe())
            self._log_metrics(metrics)
            self._log_samples(scored)
            LOGGER.info(
                "iter %d/%d reward %.3f nonzero-adv %.2f entropy %.2f tokens %d (cum %.2fM)",
                self.iteration + 1,
                self.config.iterations,
                metrics["mean_reward"],
                metrics["nonzero_advantage_fraction"],
                metrics["allocation_entropy"],
                generated_tokens,
                self.cumulative_generated_tokens / 1e6,
            )

            self.iteration += 1
            if self.iteration % self.config.save_every == 0 or self.iteration == self.config.iterations:
                self.save_checkpoint(f"{self.iteration:04d}")
