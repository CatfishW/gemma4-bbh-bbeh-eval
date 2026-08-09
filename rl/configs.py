"""Run configuration for RL training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from rl.rewards import LengthShapingConfig


@dataclass
class TrainConfig:
    # Data
    datasets_root: str = "/data/benwulab/gemma4-eval/datasets"
    benchmarks: str = "bbh,bbeh,usr"
    holdout_task_stride: int = 4

    # Model
    model_path: str = "/data/models/gemma-4-E2B-it"
    attn_implementation: str = "sdpa"
    load_in_4bit: bool = False
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0

    # Algorithm: "volt", "grpo", or "drgrpo"
    mode: str = "volt"
    iterations: int = 60
    rollout_budget: int = 512          # rollouts per iteration (all modes)
    group_size: int = 8                # grpo/drgrpo group size
    n_max: int = 8                     # volt per-prompt cap
    floor_fraction: float = 0.15       # volt exploration floor
    discount: float = 0.92             # volt posterior discount
    prior_strength: float = 4.0        # volt hierarchical prior mass
    uniform_allocation: bool = False   # ablation: predictable baseline, uniform n_i

    # Generation
    prompt_strategy: str = "concise_cot"
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 320
    max_prompt_tokens: int = 3072
    generation_batch_size: int = 16
    generation_max_batch_tokens: int = 16384

    # Optimization
    learning_rate: float = 2e-5
    warmup_iterations: int = 5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    train_tokens_per_microbatch: int = 4096
    loss_length_normalizer: int = 320  # constant normalizer (Dr. GRPO style)

    # Length shaping
    length_shaping: LengthShapingConfig = field(default_factory=LengthShapingConfig)

    # Bookkeeping
    output_dir: str = "runs/rl-run"
    seed: int = 20260709
    save_every: int = 5
    val_probe_every: int = 5
    val_probe_size: int = 300
    val_probe_max_new_tokens: int = 256
    log_samples_per_iteration: int = 2

    def to_json(self) -> dict:
        payload = asdict(self)
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "TrainConfig":
        shaping_payload = payload.pop("length_shaping", None) or {}
        config = cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__ and k != "length_shaping"})
        config.length_shaping = LengthShapingConfig(**shaping_payload)
        return config

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        return cls.from_json(json.loads(path.read_text()))
