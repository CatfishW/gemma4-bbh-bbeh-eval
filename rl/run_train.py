#!/usr/bin/env python3
"""CLI entry point for RL training (VOLT and baselines)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_benchmarks import load_bbh, load_bbeh, load_unpuzzles_simple_reasoning  # noqa: E402
from rl.configs import TrainConfig  # noqa: E402
from rl.protocol import build_protocol_split  # noqa: E402


def load_examples(datasets_root: Path, benchmarks: str):
    examples = []
    wanted = {name.strip() for name in benchmarks.split(",") if name.strip()}
    if "bbh" in wanted:
        examples.extend(load_bbh(datasets_root, None))
    if "bbeh" in wanted:
        examples.extend(load_bbeh(datasets_root, None))
    if "usr" in wanted:
        examples.extend(load_unpuzzles_simple_reasoning(datasets_root, None))
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="JSON file of TrainConfig overrides")
    parser.add_argument("--mode", choices=["volt", "grpo", "drgrpo", "grpo_ds"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--rollout-budget", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--uniform-allocation", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    payload = {}
    if args.config is not None:
        payload = json.loads(args.config.read_text())
    config = TrainConfig.from_json(payload)
    if args.mode:
        config.mode = args.mode
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    if args.iterations:
        config.iterations = args.iterations
    if args.rollout_budget:
        config.rollout_budget = args.rollout_budget
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.uniform_allocation:
        config.uniform_allocation = True
    if args.load_in_4bit:
        config.load_in_4bit = True

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "train.log"),
        ],
    )

    import torch

    torch.manual_seed(config.seed)

    from rl.modeling import (
        attach_lora,
        load_policy_model,
        load_tokenizer,
        self_check_log_probs,
    )
    from rl.trainer import RLTrainer

    examples = load_examples(Path(config.datasets_root), config.benchmarks)
    split = build_protocol_split(examples, holdout_stride=config.holdout_task_stride)
    logging.info(
        "protocol split: train %d, validation %d, test %d, holdout tasks %d",
        len(split.train),
        len(split.validation),
        len(split.test),
        len(split.holdout_tasks),
    )
    (output_dir / "holdout_tasks.json").write_text(
        json.dumps(sorted(split.holdout_tasks), indent=2) + "\n"
    )
    config.save(output_dir / "train_config.json")

    tokenizer = load_tokenizer(config.model_path)
    model = load_policy_model(
        config.model_path,
        attn_implementation=config.attn_implementation,
        load_in_4bit=config.load_in_4bit,
        device=args.device,
    )
    model = attach_lora(model, config.lora_rank, config.lora_alpha, config.lora_dropout)
    self_check_log_probs(model, tokenizer, args.device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        flagged = sum(
            1
            for _, module in model.named_modules()
            if getattr(module, "gradient_checkpointing", False)
        )
        logging.info("gradient checkpointing active on %d modules", flagged)
        if flagged == 0:
            raise RuntimeError("gradient checkpointing did not engage on any module")

    trainer = RLTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_examples=split.train,
        val_examples=split.validation,
        device=args.device,
    )
    if args.resume:
        trainer.resume_if_available()
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
