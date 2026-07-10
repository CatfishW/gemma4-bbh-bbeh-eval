#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import random
import statistics
from typing import Iterable


ExampleKey = tuple[str, str, int]
TaskKey = tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run task-cluster sensitivity analyses on frozen E2B test predictions."
    )
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="direct_answer")
    parser.add_argument("--primary", default="cbrr_policy")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--randomization-replicates", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args()


def prediction_path(root: Path) -> Path:
    for candidate in (root / "predictions.jsonl", root / "predictions.jsonl.gz"):
        if candidate.exists():
            return candidate
    raise SystemExit(f"missing predictions under {root}")


def read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def example_key(row: dict) -> ExampleKey:
    return str(row["benchmark"]), str(row["task"]), int(row["index"])


def load_rows(root: Path) -> dict[ExampleKey, dict]:
    rows = {example_key(row): row for row in read_jsonl(prediction_path(root))}
    if not rows:
        raise SystemExit(f"empty predictions under {root}")
    return rows


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def task_statistics(
    baseline: dict[ExampleKey, dict],
    primary: dict[ExampleKey, dict],
) -> list[dict]:
    grouped: dict[TaskKey, list[ExampleKey]] = defaultdict(list)
    for key in baseline:
        grouped[(key[0], key[1])].append(key)
    rows = []
    for task in sorted(grouped):
        keys = grouped[task]
        direct_correct = sum(bool(baseline[key]["correct"]) for key in keys)
        primary_correct = sum(bool(primary[key]["correct"]) for key in keys)
        wins = sum(
            not baseline[key]["correct"] and primary[key]["correct"] for key in keys
        )
        losses = sum(
            baseline[key]["correct"] and not primary[key]["correct"] for key in keys
        )
        rows.append(
            {
                "benchmark": task[0],
                "task": task[1],
                "total": len(keys),
                "direct_correct": direct_correct,
                "primary_correct": primary_correct,
                "correct_difference": primary_correct - direct_correct,
                "accuracy_difference": (primary_correct - direct_correct) / len(keys),
                "paired_wins": wins,
                "paired_losses": losses,
            }
        )
    return rows


def cluster_bootstrap(
    tasks: list[dict],
    replicates: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    pooled = []
    macro = []
    for _ in range(replicates):
        sample = [rng.choice(tasks) for _ in tasks]
        pooled.append(
            sum(row["correct_difference"] for row in sample)
            / sum(row["total"] for row in sample)
        )
        macro.append(statistics.mean(row["accuracy_difference"] for row in sample))
    return {
        "replicates": replicates,
        "seed": seed,
        "pooled_micro_difference_95": [
            percentile(pooled, 0.025),
            percentile(pooled, 0.975),
        ],
        "macro_task_difference_95": [
            percentile(macro, 0.025),
            percentile(macro, 0.975),
        ],
    }


def cluster_sign_flip_p(tasks: list[dict], replicates: int, seed: int) -> float:
    rng = random.Random(seed)
    observed = abs(sum(row["correct_difference"] for row in tasks))
    at_least_as_extreme = 0
    differences = [int(row["correct_difference"]) for row in tasks]
    for _ in range(replicates):
        permuted = abs(sum(value if rng.getrandbits(1) else -value for value in differences))
        at_least_as_extreme += permuted >= observed
    return (at_least_as_extreme + 1) / (replicates + 1)


def leave_one_task_out(tasks: list[dict]) -> dict:
    total_examples = sum(row["total"] for row in tasks)
    total_difference = sum(row["correct_difference"] for row in tasks)
    values = []
    for row in tasks:
        difference = (total_difference - row["correct_difference"]) / (
            total_examples - row["total"]
        )
        values.append(
            {
                "omitted_task": f"{row['benchmark']}/{row['task']}",
                "pooled_accuracy_difference": difference,
            }
        )
    ordered = sorted(values, key=lambda row: row["pooled_accuracy_difference"])
    return {
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "all": ordered,
    }


def generation_tokens(row: dict) -> int:
    generations = row.get("generations") or []
    if not generations:
        return 0
    usage = generations[0].get("usage") or {}
    return int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)


def policy_audit(
    primary: dict[ExampleKey, dict], selection: dict, actual_cap: int
) -> dict:
    manifest = {str(row["name"]): row for row in selection["manifest"]}
    selected_counts = Counter(str(row.get("prompt_strategy")) for row in primary.values())
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for row in primary.values():
        by_strategy[str(row.get("prompt_strategy"))].append(row)
    strategy_rows = {}
    registered_cap_exceedances = 0
    actual_cap_bindings = 0
    for strategy, rows in sorted(by_strategy.items()):
        registered_cap = int(manifest[strategy]["max_tokens"])
        token_values = [generation_tokens(row) for row in rows]
        exceeded = sum(value > registered_cap for value in token_values)
        bound = sum(value >= actual_cap for value in token_values)
        registered_cap_exceedances += exceeded
        actual_cap_bindings += bound
        strategy_rows[strategy] = {
            "examples": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "registered_screening_max_tokens": registered_cap,
            "mean_generation_completion_tokens": statistics.mean(token_values),
            "above_registered_screening_cap": exceeded,
            "at_actual_policy_cap": bound,
        }
    return {
        "actual_policy_max_tokens": actual_cap,
        "selected_prompt_counts": dict(selected_counts),
        "by_selected_prompt": strategy_rows,
        "above_registered_screening_cap": registered_cap_exceedances,
        "at_actual_policy_cap": actual_cap_bindings,
    }


def write_report(path: Path, analysis: dict) -> None:
    bootstrap = analysis["task_cluster_bootstrap"]
    influence = analysis["leave_one_task_out"]
    audit = analysis["policy_audit"]
    lines = [
        "# E2B Task-Cluster Robustness",
        "",
        f"The primary CBRR effect is {100 * analysis['pooled_accuracy_difference']:+.2f} "
        f"percentage points over {analysis['test_examples']:,} test examples in "
        f"{analysis['task_clusters']} task clusters.",
        "",
        "## Cluster sensitivity",
        "",
        f"- Task-cluster bootstrap pooled 95% interval: "
        f"[{100 * bootstrap['pooled_micro_difference_95'][0]:+.2f}, "
        f"{100 * bootstrap['pooled_micro_difference_95'][1]:+.2f}] percentage points.",
        f"- Task-cluster bootstrap macro-task 95% interval: "
        f"[{100 * bootstrap['macro_task_difference_95'][0]:+.2f}, "
        f"{100 * bootstrap['macro_task_difference_95'][1]:+.2f}] percentage points.",
        f"- Fixed-seed task sign-flip sensitivity p-value: "
        f"{analysis['task_sign_flip']['p_value']:.6g}.",
        f"- Leave-one-task-out pooled range: "
        f"[{100 * influence['minimum']['pooled_accuracy_difference']:+.2f}, "
        f"{100 * influence['maximum']['pooled_accuracy_difference']:+.2f}] percentage points.",
        "",
        "## Policy audit",
        "",
        f"CBRR outputs above their arm's registered screening cap: "
        f"{audit['above_registered_screening_cap']:,}. Outputs at the actual "
        f"{audit['actual_policy_max_tokens']}-token test cap: "
        f"{audit['at_actual_policy_cap']:,}.",
        "",
        "| Selected prompt | Examples | Accuracy | Mean completion tokens | Above registered cap | At test cap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, row in audit["by_selected_prompt"].items():
        lines.append(
            f"| `{strategy}` | {row['examples']} | {100 * row['accuracy']:.2f}% | "
            f"{row['mean_generation_completion_tokens']:.2f} | "
            f"{row['above_registered_screening_cap']} | {row['at_actual_policy_cap']} |"
        )
    lines.extend(
        [
            "",
            "This is a preregistered sensitivity analysis. The exact paired McNemar test "
            "remains the primary inferential result.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    if not selection.get("selection_frozen") or selection.get("test_rows_read") != 0:
        raise SystemExit("selection integrity gate is not satisfied")
    baseline = load_rows(args.test_root / args.baseline)
    primary = load_rows(args.test_root / args.primary)
    if set(baseline) != set(primary):
        raise SystemExit("baseline and primary coverage differ")
    if any(key[2] < 50 for key in baseline):
        raise SystemExit("test predictions contain calibration or validation rows")

    primary_config = json.loads(
        (args.test_root / args.primary / "run_config.json").read_text()
    )
    actual_policy_cap = int(primary_config["max_tokens"])
    tasks = task_statistics(baseline, primary)
    total_examples = len(baseline)
    total_difference = sum(row["correct_difference"] for row in tasks)
    analysis = {
        "extension_id": "gemma4-e2b-confirmatory-v1-analysis-extension-001",
        "baseline": args.baseline,
        "primary": args.primary,
        "test_examples": total_examples,
        "task_clusters": len(tasks),
        "pooled_accuracy_difference": total_difference / total_examples,
        "macro_task_accuracy_difference": statistics.mean(
            row["accuracy_difference"] for row in tasks
        ),
        "task_cluster_bootstrap": cluster_bootstrap(
            tasks, args.bootstrap_replicates, args.seed
        ),
        "task_sign_flip": {
            "replicates": args.randomization_replicates,
            "seed": args.seed,
            "p_value": cluster_sign_flip_p(
                tasks, args.randomization_replicates, args.seed
            ),
        },
        "leave_one_task_out": leave_one_task_out(tasks),
        "policy_audit": policy_audit(primary, selection, actual_policy_cap),
        "tasks": tasks,
        "system_messages_sent": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cluster_robustness.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    write_report(args.output_dir / "cluster_robustness.md", analysis)
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
