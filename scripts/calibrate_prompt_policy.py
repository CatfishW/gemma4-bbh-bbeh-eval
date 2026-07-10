#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks import (
    Example,
    PROMPT_STRATEGIES,
    load_bbeh,
    load_bbh,
    load_unpuzzles_simple_reasoning,
    summarize,
)


ExampleKey = tuple[str, str, int]
TaskKey = tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a task-conditioned prompt policy from a fixed calibration prefix and score "
            "it on the remaining examples."
        )
    )
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--strategies", required=True, help="Comma-separated strategy directory names")
    parser.add_argument("--baseline", default="direct_answer")
    parser.add_argument("--calibration-size", type=int, default=25)
    parser.add_argument("--min-reward-gain", type=int, default=1)
    parser.add_argument("--policy-name", default="reward_routed_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path)
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Fit and write policy.json without requiring held-out predictions in the run root.",
    )
    args = parser.parse_args()
    args.strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if args.baseline not in args.strategies:
        raise SystemExit("--baseline must be included in --strategies")
    unknown = sorted(set(args.strategies) - set(PROMPT_STRATEGIES))
    if unknown:
        raise SystemExit(f"unknown prompt strategies: {', '.join(unknown)}")
    if args.calibration_size < 1:
        raise SystemExit("--calibration-size must be >= 1")
    if args.min_reward_gain < 0:
        raise SystemExit("--min-reward-gain must be >= 0")
    return args


def prediction_path(root: Path, strategy: str) -> Path | None:
    candidates = [
        root / strategy / "predictions.jsonl",
        root / strategy / "predictions.jsonl.gz",
    ]
    return next((path for path in candidates if path.exists()), None)


def read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def example_key(row: dict) -> ExampleKey:
    return str(row["benchmark"]), str(row["task"]), int(row["index"])


def task_key(key: ExampleKey) -> TaskKey:
    return key[0], key[1]


def task_name(key: TaskKey) -> str:
    return f"{key[0]}/{key[1]}"


def load_strategy_rows(roots: list[Path], strategy: str) -> tuple[dict[ExampleKey, dict], list[str]]:
    rows: dict[ExampleKey, dict] = {}
    sources = []
    for root in roots:
        path = prediction_path(root, strategy)
        if path is None:
            continue
        sources.append(str(path))
        for row in read_jsonl(path):
            key = example_key(row)
            if key in rows:
                raise SystemExit(f"duplicate prediction for {strategy}: {key}")
            rows[key] = row
    if not rows:
        raise SystemExit(f"no predictions found for strategy {strategy}")
    return rows, sources


def score_keys(rows: dict[ExampleKey, dict], keys: Iterable[ExampleKey]) -> tuple[int, int]:
    selected = list(keys)
    return sum(bool(rows[key]["correct"]) for key in selected), len(selected)


def load_examples(root: Path) -> dict[ExampleKey, Example]:
    examples = [
        *load_bbh(root, None),
        *load_bbeh(root, None),
        *load_unpuzzles_simple_reasoning(root, None),
    ]
    return {(item.benchmark, item.task, item.index): item for item in examples}


def markdown_text(value: str, limit: int = 1600) -> str:
    value = value.strip()
    if len(value) > limit:
        value = value[: limit - 3].rstrip() + "..."
    return value.replace("```", "` ` `")


def write_examples(
    path: Path,
    heldout_keys: list[ExampleKey],
    baseline: str,
    policy: dict[TaskKey, str],
    rows_by_strategy: dict[str, dict[ExampleKey, dict]],
    examples: dict[ExampleKey, Example] | None,
) -> None:
    wins = []
    losses = []
    for key in heldout_keys:
        selected = policy[task_key(key)]
        baseline_row = rows_by_strategy[baseline][key]
        selected_row = rows_by_strategy[selected][key]
        if not baseline_row["correct"] and selected_row["correct"]:
            wins.append(key)
        elif baseline_row["correct"] and not selected_row["correct"]:
            losses.append(key)

    chosen = [("Policy win", key) for key in wins[:8]]
    chosen.extend(("Policy loss", key) for key in losses[:4])
    lines = [
        "# Reward-Routed Prompt Policy Examples",
        "",
        "These examples come only from the held-out suffix of each task. The policy arm was "
        "selected from the fixed calibration prefix.",
        "",
    ]
    for label, key in chosen:
        selected = policy[task_key(key)]
        direct_row = rows_by_strategy[baseline][key]
        selected_row = rows_by_strategy[selected][key]
        lines.extend(
            [
                f"## {label}: {key[0]}/{key[1]} #{key[2]}",
                "",
                f"- Selected strategy: `{selected}`",
                f"- Ground truth: `{selected_row['target']}`",
                f"- Direct answer: `{markdown_text(str(direct_row['prediction']), 500)}`",
                f"- Policy answer: `{markdown_text(str(selected_row['prediction']), 500)}`",
                "",
            ]
        )
        if examples is not None and key in examples:
            lines.extend(
                [
                    "Question:",
                    "",
                    "```text",
                    markdown_text(examples[key].input),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_strategy = {}
    source_files = {}
    for strategy in args.strategies:
        rows_by_strategy[strategy], source_files[strategy] = load_strategy_rows(
            args.runs_root, strategy
        )

    all_keys = set(rows_by_strategy[args.baseline])
    for strategy, rows in rows_by_strategy.items():
        if set(rows) != all_keys:
            missing = len(all_keys - set(rows))
            extra = len(set(rows) - all_keys)
            raise SystemExit(
                f"prediction coverage mismatch for {strategy}: missing={missing} extra={extra}"
            )

    calibration_keys = sorted(key for key in all_keys if key[2] < args.calibration_size)
    heldout_keys = sorted(all_keys - set(calibration_keys))
    tasks = sorted({task_key(key) for key in all_keys})
    calibration_by_task: dict[TaskKey, list[ExampleKey]] = defaultdict(list)
    for key in calibration_keys:
        calibration_by_task[task_key(key)].append(key)

    policy: dict[TaskKey, str] = {}
    task_rows = []
    for task in tasks:
        keys = calibration_by_task[task]
        rewards = {
            strategy: score_keys(rows_by_strategy[strategy], keys)[0]
            for strategy in args.strategies
        }
        best_reward = max(rewards.values())
        best = next(strategy for strategy in args.strategies if rewards[strategy] == best_reward)
        if rewards[best] < rewards[args.baseline] + args.min_reward_gain:
            best = args.baseline
        policy[task] = best
        total = len(keys)
        task_rows.append(
            {
                "task": task_name(task),
                "selected_strategy": best,
                "calibration_examples": total,
                "rewards": rewards,
                "posterior_means": {
                    strategy: (reward + 1) / (total + 2)
                    for strategy, reward in rewards.items()
                },
            }
        )

    policy_payload = {
        "name": args.policy_name,
        "description": (
            "Beta-Bernoulli contextual bandit over prompt strategies. Each benchmark task is a "
            "context; exact-match correctness is the reward; direct_answer wins ties."
        ),
        "default_strategy": args.baseline,
        "calibration_size_per_task": args.calibration_size,
        "min_reward_gain": args.min_reward_gain,
        "selection_rule": (
            "Select the highest posterior-mean arm on the fixed calibration prefix; require the "
            "configured reward gain over direct_answer and preserve strategy order for ties."
        ),
        "strategy_order": args.strategies,
        "task_strategies": {task_name(task): strategy for task, strategy in sorted(policy.items())},
        "system_messages_sent": 0,
    }
    (args.output_dir / "policy.json").write_text(
        json.dumps(policy_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    calibration_summary = {
        "source_roots": [str(root) for root in args.runs_root],
        "source_files": source_files,
        "calibration_size_per_task": args.calibration_size,
        "calibration_examples": len(calibration_keys),
        "strategies": args.strategies,
        "baseline": args.baseline,
        "selected_strategy_counts": dict(Counter(policy.values())),
        "tasks": task_rows,
    }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(calibration_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.calibration_only:
        print(json.dumps(calibration_summary, indent=2, ensure_ascii=False))
        return 0
    if not heldout_keys:
        raise SystemExit(
            "no held-out predictions remain; use --calibration-only to fit a policy from a "
            "calibration-only sweep"
        )

    selected_rows = []
    for key in heldout_keys:
        selected_strategy = policy[task_key(key)]
        row = dict(rows_by_strategy[selected_strategy][key])
        row["prompt_strategy"] = selected_strategy
        row["policy_name"] = args.policy_name
        selected_rows.append(row)
    with gzip.open(args.output_dir / "heldout_predictions.jsonl.gz", "wt", encoding="utf-8") as out:
        for row in selected_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    baseline_correct, heldout_total = score_keys(
        rows_by_strategy[args.baseline], heldout_keys
    )
    policy_correct = sum(bool(row["correct"]) for row in selected_rows)
    heldout_summary = {
        "policy_name": args.policy_name,
        "evaluation_split": f"index >= {args.calibration_size} within each task",
        "calibration_examples_excluded": len(calibration_keys),
        "heldout_examples": heldout_total,
        "baseline": {
            "strategy": args.baseline,
            "correct": baseline_correct,
            "total": heldout_total,
            "accuracy": baseline_correct / heldout_total,
        },
        "policy": {
            "correct": policy_correct,
            "total": heldout_total,
            "accuracy": policy_correct / heldout_total,
        },
        "absolute_correct_gain": policy_correct - baseline_correct,
        "absolute_accuracy_gain": (policy_correct - baseline_correct) / heldout_total,
        **summarize(selected_rows),
    }
    (args.output_dir / "heldout_summary.json").write_text(
        json.dumps(heldout_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    example_lookup = load_examples(args.datasets_root) if args.datasets_root else None
    write_examples(
        args.output_dir / "examples.md",
        heldout_keys,
        args.baseline,
        policy,
        rows_by_strategy,
        example_lookup,
    )
    print(json.dumps(heldout_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
