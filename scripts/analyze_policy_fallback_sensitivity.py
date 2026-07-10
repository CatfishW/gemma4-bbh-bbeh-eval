#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_confirmatory_results import (
    ExampleKey,
    completion_tokens,
    exact_mcnemar_p,
    load_arm_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay policy fallback rows from the lower-cap direct baseline."
    )
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="direct_answer")
    parser.add_argument("--policy", default="cbrr_policy")
    parser.add_argument("--fallback-strategy", default="direct_answer")
    return parser.parse_args()


def build_fallback_replay(
    baseline: dict[ExampleKey, dict],
    policy: dict[ExampleKey, dict],
    fallback_strategy: str,
) -> tuple[dict[ExampleKey, dict], list[ExampleKey], list[ExampleKey]]:
    if set(baseline) != set(policy):
        raise ValueError("baseline and policy coverage differ")
    fallback_keys = sorted(
        key
        for key, row in policy.items()
        if str(row.get("prompt_strategy")) == fallback_strategy
    )
    fallback_key_set = set(fallback_keys)
    specialized_keys = sorted(set(policy) - fallback_key_set)
    replay = {
        key: baseline[key] if key in fallback_key_set else policy[key]
        for key in policy
    }
    return replay, fallback_keys, specialized_keys


def correct(rows: dict[ExampleKey, dict], keys: list[ExampleKey]) -> int:
    return sum(bool(rows[key]["correct"]) for key in keys)


def mean_completion_tokens(
    rows: dict[ExampleKey, dict], keys: list[ExampleKey]
) -> float:
    return statistics.mean(completion_tokens(rows[key]) for key in keys)


def paired_comparison(
    baseline: dict[ExampleKey, dict],
    challenger: dict[ExampleKey, dict],
    keys: list[ExampleKey],
) -> dict:
    wins = sum(
        not baseline[key]["correct"] and challenger[key]["correct"] for key in keys
    )
    losses = sum(
        baseline[key]["correct"] and not challenger[key]["correct"] for key in keys
    )
    return {
        "paired_wins": wins,
        "paired_losses": losses,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(wins, losses),
    }


def write_report(path: Path, analysis: dict) -> None:
    replay = analysis["fallback_replay"]
    lines = [
        "# E2B Direct-Fallback Replay Sensitivity",
        "",
        "This is a post-hoc sensitivity analysis, not a preregistered primary test. "
        "It replaces CBRR rows assigned to `direct_answer` with the corresponding "
        "64-token-cap baseline output while retaining frozen CBRR outputs for "
        "specialized prompt assignments.",
        "",
        f"- Direct baseline: {analysis['direct']['correct']}/{analysis['examples']} "
        f"({100 * analysis['direct']['accuracy']:.2f}%).",
        f"- Registered CBRR: {analysis['cbrr']['correct']}/{analysis['examples']} "
        f"({100 * analysis['cbrr']['accuracy']:.2f}%).",
        f"- Fallback replay: {replay['correct']}/{analysis['examples']} "
        f"({100 * replay['accuracy']:.2f}%), "
        f"{replay['accuracy_point_difference_vs_direct']:+.2f} points versus direct.",
        f"- Correct-answer change from registered CBRR: "
        f"{replay['correct_difference_vs_cbrr']:+d}.",
        f"- Mean completion tokens after replay: "
        f"{replay['mean_completion_tokens']:.2f}.",
        "",
        "This isolates the direct-fallback cap only. It does not make every specialized "
        "arm token-matched; the registered CBRR token and cap audit remains the governing "
        "cost report.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    baseline = load_arm_rows(args.test_root / args.baseline)
    policy = load_arm_rows(args.test_root / args.policy)
    replay, fallback_keys, specialized_keys = build_fallback_replay(
        baseline, policy, args.fallback_strategy
    )
    keys = sorted(baseline)
    direct_correct = correct(baseline, keys)
    policy_correct = correct(policy, keys)
    replay_correct = correct(replay, keys)
    by_benchmark = {}
    for benchmark in sorted({key[0] for key in keys}):
        benchmark_keys = [key for key in keys if key[0] == benchmark]
        benchmark_direct = correct(baseline, benchmark_keys)
        benchmark_replay = correct(replay, benchmark_keys)
        by_benchmark[benchmark] = {
            "examples": len(benchmark_keys),
            "direct_correct": benchmark_direct,
            "fallback_replay_correct": benchmark_replay,
            "accuracy_point_difference_vs_direct": 100
            * (benchmark_replay - benchmark_direct)
            / len(benchmark_keys),
        }
    comparison = paired_comparison(baseline, replay, keys)
    analysis = {
        "analysis_status": "post_hoc_sensitivity",
        "inference_rerun": False,
        "prediction_mutation": False,
        "examples": len(keys),
        "fallback_strategy": args.fallback_strategy,
        "fallback_examples": len(fallback_keys),
        "specialized_examples": len(specialized_keys),
        "direct": {
            "correct": direct_correct,
            "accuracy": direct_correct / len(keys),
            "mean_completion_tokens": mean_completion_tokens(baseline, keys),
        },
        "cbrr": {
            "correct": policy_correct,
            "accuracy": policy_correct / len(keys),
            "mean_completion_tokens": mean_completion_tokens(policy, keys),
        },
        "fallback_subset": {
            "examples": len(fallback_keys),
            "direct_correct": correct(baseline, fallback_keys),
            "cbrr_correct": correct(policy, fallback_keys),
        },
        "specialized_subset": {
            "examples": len(specialized_keys),
            "direct_correct": correct(baseline, specialized_keys),
            "cbrr_correct": correct(policy, specialized_keys),
        },
        "fallback_replay": {
            "correct": replay_correct,
            "accuracy": replay_correct / len(keys),
            "correct_difference_vs_cbrr": replay_correct - policy_correct,
            "accuracy_point_difference_vs_direct": 100
            * (replay_correct - direct_correct)
            / len(keys),
            "mean_completion_tokens": mean_completion_tokens(replay, keys),
            **comparison,
        },
        "by_benchmark": by_benchmark,
        "system_messages_sent": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fallback_replay_sensitivity.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    write_report(args.output_dir / "fallback_replay_sensitivity.md", analysis)
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
