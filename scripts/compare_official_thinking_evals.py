#!/usr/bin/env python3
"""Compare completed native-thinking BBEH predictions on matched frozen rows."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics


def load_predictions(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("prompt_id")
        if not isinstance(key, str):
            raise ValueError(f"missing prompt_id at {path}:{line_number}")
        if key in rows:
            raise ValueError(f"duplicate prompt_id in {path}: {key}")
        rows[key] = row
    return rows


def metrics(rows: dict[str, dict], keys: list[str]) -> dict:
    correct = sum(bool(rows[key]["correct"]) for key in keys)
    return {
        "examples": len(keys),
        "correct": correct,
        "accuracy": correct / len(keys) if keys else 0.0,
        "mean_completion_tokens": (
            statistics.fmean(rows[key]["completion_tokens"] for key in keys) if keys else 0.0
        ),
        "mean_thinking_tokens": (
            statistics.fmean(rows[key]["thinking_tokens"] for key in keys) if keys else 0.0
        ),
        "truncated": sum(bool(rows[key]["truncated"]) for key in keys),
        "parse_errors": sum(rows[key]["parse_error"] is not None for key in keys),
    }


def exact_mcnemar_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail_end = min(wins, losses)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(tail_end + 1)
    ]
    largest = max(log_terms)
    log_tail = largest + math.log(math.fsum(math.exp(value - largest) for value in log_terms))
    log_two_sided = math.log(2.0) + log_tail
    return 1.0 if log_two_sided >= 0.0 else math.exp(log_two_sided)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stratified_bootstrap_delta(
    baseline: dict[str, dict],
    challenger: dict[str, dict],
    keys: list[str],
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for key in keys:
        by_task[baseline[key]["task"]].append(
            int(bool(challenger[key]["correct"])) - int(bool(baseline[key]["correct"]))
        )
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        total_differences = np.zeros(replicates, dtype=np.int64)
        for values in by_task.values():
            counts = [values.count(-1), values.count(0), values.count(1)]
            size = len(values)
            sampled = rng.multinomial(
                size, [count / size for count in counts], size=replicates
            )
            total_differences += sampled[:, 2] - sampled[:, 0]
        return (total_differences / len(keys)).tolist()
    except ImportError:
        rng = random.Random(seed)
        samples: list[float] = []
        for _ in range(replicates):
            difference = 0
            for values in by_task.values():
                difference += sum(rng.choice(values) for _ in values)
            samples.append(difference / len(keys))
        return samples


def paired_comparison(
    baseline: dict[str, dict],
    challenger: dict[str, dict],
    keys: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict:
    baseline_metrics = metrics(baseline, keys)
    challenger_metrics = metrics(challenger, keys)
    wins = sum(
        not bool(baseline[key]["correct"]) and bool(challenger[key]["correct"])
        for key in keys
    )
    losses = sum(
        bool(baseline[key]["correct"]) and not bool(challenger[key]["correct"])
        for key in keys
    )
    bootstrap = stratified_bootstrap_delta(
        baseline,
        challenger,
        keys,
        replicates=replicates,
        seed=seed,
    )
    delta = challenger_metrics["accuracy"] - baseline_metrics["accuracy"]
    baseline_tokens = baseline_metrics["mean_completion_tokens"]
    return {
        "accuracy_difference": delta,
        "accuracy_point_difference": 100 * delta,
        "paired_wins": wins,
        "paired_losses": losses,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(wins, losses),
        "task_stratified_bootstrap_95": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "completion_token_reduction": (
            1.0 - challenger_metrics["mean_completion_tokens"] / baseline_tokens
            if baseline_tokens
            else 0.0
        ),
    }


def build_markdown(result: dict) -> str:
    full = result["paper_comparison"]
    frozen = result["frozen_test"]
    lines = [
        "# Gemma 4 native-thinking BBEH comparison",
        "",
        "This is the separate best-public reproduction profile. It does not replace the "
        "preregistered greedy non-thinking evaluation.",
        "",
        "## Full 4,520-row paper-comparison cell",
        "",
        "| Cell | Correct | Accuracy | Mean completion tokens | Truncated | Parse errors |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Base E2B reproduction | {full['base']['correct']}/{full['base']['examples']} | "
            f"{full['base']['accuracy']:.2%} | {full['base']['mean_completion_tokens']:.1f} | "
            f"{full['base']['truncated']} | {full['base']['parse_errors']} |"
        ),
        f"| Gemma 4 report, Table 5 | - | {full['paper_accuracy']:.2%} | - | - | - |",
        "",
        f"Reproduction minus paper: {full['difference_pp']:+.2f} percentage points. "
        "The report does not publish its BBEH output-token ceiling, seed, sample count, or full "
        "internal harness, so this gap is descriptive rather than an implementation-equivalence test.",
        "",
        "## Frozen test (3,370 matched rows)",
        "",
        "| Model | Correct | Accuracy | Mean completion tokens | Truncated | Parse errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("base", "grpo", "volt"):
        row = frozen["models"][name]
        lines.append(
            f"| {name.upper()} | {row['correct']}/{row['examples']} | {row['accuracy']:.2%} | "
            f"{row['mean_completion_tokens']:.1f} | {row['truncated']} | {row['parse_errors']} |"
        )
    lines.extend(
        [
            "",
            "## Paired improvements",
            "",
            "| Comparison | Accuracy delta | Bootstrap 95% | Wins/losses | McNemar p | Token change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("grpo_vs_base", "volt_vs_base", "volt_vs_grpo"):
        row = frozen["comparisons"][name]
        low, high = row["task_stratified_bootstrap_95"]
        lines.append(
            f"| {name.replace('_', ' ')} | {row['accuracy_point_difference']:+.2f} pp | "
            f"[{100 * low:+.2f}, {100 * high:+.2f}] pp | "
            f"{row['paired_wins']}/{row['paired_losses']} | "
            f"{row['exact_mcnemar_two_sided_p']:.4g} | "
            f"{-100 * row['completion_token_reduction']:+.1f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--grpo", type=Path, required=True)
    parser.add_argument("--volt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    base = load_predictions(args.base)
    grpo = load_predictions(args.grpo)
    volt = load_predictions(args.volt)
    if set(grpo) != set(volt):
        raise RuntimeError("GRPO and VOLT prediction keys do not match")
    frozen_keys = sorted(grpo)
    if not set(frozen_keys) <= set(base):
        raise RuntimeError("full base predictions do not cover every frozen-test key")
    if len(base) != 4520 or len(frozen_keys) != 3370:
        raise RuntimeError(
            f"unexpected denominators: full base={len(base)}, frozen={len(frozen_keys)}"
        )
    if any(int(base[key]["index"]) < 50 for key in frozen_keys):
        raise RuntimeError("adapter predictions contain non-frozen rows")

    full_keys = sorted(base)
    paper_accuracy = 0.219
    full_base = metrics(base, full_keys)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": "gemma4_public_native_thinking_bbeh_v3",
        "paper_comparison": {
            "base": full_base,
            "paper_accuracy": paper_accuracy,
            "difference_pp": 100 * (full_base["accuracy"] - paper_accuracy),
        },
        "frozen_test": {
            "models": {
                "base": metrics(base, frozen_keys),
                "grpo": metrics(grpo, frozen_keys),
                "volt": metrics(volt, frozen_keys),
            },
            "comparisons": {
                "grpo_vs_base": paired_comparison(
                    base,
                    grpo,
                    frozen_keys,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed,
                ),
                "volt_vs_base": paired_comparison(
                    base,
                    volt,
                    frozen_keys,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + 1,
                ),
                "volt_vs_grpo": paired_comparison(
                    grpo,
                    volt,
                    frozen_keys,
                    replicates=args.bootstrap_replicates,
                    seed=args.seed + 2,
                ),
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    markdown = build_markdown(result)
    (args.output_dir / "comparison.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
