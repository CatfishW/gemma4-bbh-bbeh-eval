#!/usr/bin/env python3
"""Local batched evaluation of a (possibly LoRA-adapted) policy on the frozen splits.

Mirrors the deployed evaluation protocol: single user message, no system
message, greedy decoding, the frozen per-arm max-token limits, and the
unchanged repository scorer. Writes predictions.jsonl and summary.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_benchmarks import (  # noqa: E402
    evaluate_correctness,
    load_bbh,
    load_bbeh,
    load_unpuzzles_simple_reasoning,
)
from rl.protocol import build_protocol_split, prompt_id  # noqa: E402
from rl.rollout import RolloutRequest, chat_prompt_token_ids, generate_rollouts  # noqa: E402

STRATEGY_MAX_TOKENS = {"direct_answer": 64, "concise_cot": 256}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--model-path", default="/data/models/gemma-4-E2B-it")
    parser.add_argument("--adapter", type=Path, help="LoRA adapter directory (optional)")
    parser.add_argument("--benchmarks", default="bbh,bbeh,usr")
    parser.add_argument("--split", choices=["validation", "test", "train"], default="validation")
    parser.add_argument("--prompt-strategy", default="concise_cot")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="cap examples for smoke runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    examples = []
    wanted = {name.strip() for name in args.benchmarks.split(",") if name.strip()}
    if "bbh" in wanted:
        examples.extend(load_bbh(args.datasets_root, None))
    if "bbeh" in wanted:
        examples.extend(load_bbeh(args.datasets_root, None))
    if "usr" in wanted:
        examples.extend(load_unpuzzles_simple_reasoning(args.datasets_root, None))
    split = build_protocol_split(examples)
    rows = {"validation": split.validation, "test": split.test, "train": split.train}[args.split]
    if args.limit:
        rows = rows[: args.limit]

    max_new_tokens = args.max_new_tokens or STRATEGY_MAX_TOKENS.get(args.prompt_strategy, 256)

    from rl.modeling import load_policy_model, load_tokenizer

    tokenizer = load_tokenizer(args.model_path)
    model = load_policy_model(
        args.model_path,
        load_in_4bit=args.load_in_4bit,
        device=args.device,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        logging.info("loaded adapter %s", args.adapter)
    model.eval()

    requests: list[RolloutRequest] = []
    skipped_rows: list[dict] = []
    for example in rows:
        token_ids = chat_prompt_token_ids(tokenizer, example, args.prompt_strategy)
        if len(token_ids) > args.max_prompt_tokens:
            # Scored as incorrect so accuracy denominators match the frozen
            # split exactly; applied identically to every evaluated policy.
            skipped_rows.append(
                {
                    "benchmark": example.benchmark,
                    "task": example.task,
                    "index": example.index,
                    "prompt_strategy": args.prompt_strategy,
                    "target": example.target,
                    "prediction": "",
                    "correct": False,
                    "completion_tokens": 0,
                    "skipped_long_prompt": True,
                }
            )
            continue
        requests.append(
            RolloutRequest(
                prompt_id=prompt_id(example), example=example, prompt_token_ids=token_ids
            )
        )
    logging.info(
        "evaluating %d examples (%d over prompt cap scored incorrect)",
        len(requests),
        len(skipped_rows),
    )

    results = generate_rollouts(
        model,
        tokenizer,
        requests,
        iteration=0,
        seed=20260709,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )

    records = list(skipped_rows)
    for result in results:
        correct = evaluate_correctness(result.completion_text, result.example.target)
        records.append(
            {
                "benchmark": result.example.benchmark,
                "task": result.example.task,
                "index": result.example.index,
                "prompt_strategy": args.prompt_strategy,
                "target": result.example.target,
                "prediction": result.completion_text,
                "correct": bool(correct),
                "completion_tokens": len(result.completion_token_ids),
            }
        )
    records.sort(key=lambda row: (row["benchmark"], row["task"], row["index"]))
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def accuracy(subset: list[dict]) -> dict:
        correct = sum(1 for row in subset if row["correct"])
        return {
            "examples": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else 0.0,
            "mean_completion_tokens": (
                statistics.fmean(row["completion_tokens"] for row in subset) if subset else 0.0
            ),
        }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": args.model_path,
        "adapter": str(args.adapter) if args.adapter else None,
        "split": args.split,
        "prompt_strategy": args.prompt_strategy,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "skipped_over_prompt_cap": len(skipped_rows),
        "overall": accuracy(records),
        "by_benchmark": {
            benchmark: accuracy([row for row in records if row["benchmark"] == benchmark])
            for benchmark in sorted({row["benchmark"] for row in records})
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info(
        "%s %s: %d/%d = %.2f%%",
        args.split,
        args.prompt_strategy,
        summary["overall"]["correct"],
        summary["overall"]["examples"],
        100.0 * summary["overall"]["accuracy"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
