#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks import (
    Example,
    load_bbeh,
    load_bbh,
    load_unpuzzles_simple_reasoning,
)


ExampleKey = tuple[str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen confirmatory test predictions.")
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path)
    parser.add_argument("--robustness-root", type=Path)
    parser.add_argument("--baseline", default="direct_answer")
    parser.add_argument("--primary", default="cbrr_policy")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260709)
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


def load_arm_rows(root: Path) -> dict[ExampleKey, dict]:
    rows = {example_key(row): row for row in read_jsonl(prediction_path(root))}
    if not rows:
        raise SystemExit(f"empty predictions under {root}")
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def completion_tokens(row: dict) -> int:
    usage = row.get("usage") or {}
    return int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)


def prompt_tokens(row: dict) -> int:
    usage = row.get("usage") or {}
    return int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)


def total_tokens(row: dict) -> int:
    usage = row.get("usage") or {}
    value = usage.get("total_tokens")
    return int(value) if value is not None else completion_tokens(row) + prompt_tokens(row)


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    margin /= denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def metrics(rows: dict[ExampleKey, dict], keys: list[ExampleKey]) -> dict:
    correct = sum(bool(rows[key]["correct"]) for key in keys)
    completion = [completion_tokens(rows[key]) for key in keys]
    prompt = [prompt_tokens(rows[key]) for key in keys]
    total = [total_tokens(rows[key]) for key in keys]
    elapsed = [float(rows[key].get("elapsed_seconds") or 0.0) for key in keys]
    errors = sum(bool(rows[key].get("error")) for key in keys)
    return {
        "correct": correct,
        "total": len(keys),
        "accuracy": correct / len(keys) if keys else 0.0,
        "accuracy_wilson_95": wilson_interval(correct, len(keys)),
        "errors": errors,
        "error_rate": errors / len(keys) if keys else 0.0,
        "completion_tokens": {
            "sum": sum(completion),
            "mean": statistics.mean(completion) if completion else 0.0,
            "median": statistics.median(completion) if completion else 0.0,
            "p95": percentile([float(value) for value in completion], 0.95),
        },
        "prompt_tokens": {"sum": sum(prompt), "mean": statistics.mean(prompt) if prompt else 0.0},
        "total_tokens": {"sum": sum(total), "mean": statistics.mean(total) if total else 0.0},
        "elapsed_seconds": {
            "sum": sum(elapsed),
            "mean": statistics.mean(elapsed) if elapsed else 0.0,
            "median": statistics.median(elapsed) if elapsed else 0.0,
            "p95": percentile(elapsed, 0.95),
        },
    }


def exact_mcnemar_p(paired_wins: int, paired_losses: int) -> float:
    discordant = paired_wins + paired_losses
    if discordant == 0:
        return 1.0
    tail_end = min(paired_wins, paired_losses)
    log_probability_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(tail_end + 1)
    ]
    largest = max(log_probability_terms)
    log_tail = largest + math.log(
        math.fsum(math.exp(value - largest) for value in log_probability_terms)
    )
    log_two_sided = math.log(2.0) + log_tail
    return 1.0 if log_two_sided >= 0.0 else math.exp(log_two_sided)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: (p_values[name], name))
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def stratified_bootstrap_delta(
    baseline: dict[ExampleKey, dict],
    challenger: dict[ExampleKey, dict],
    keys: list[ExampleKey],
    replicates: int,
    seed: int,
) -> list[float]:
    by_task: dict[tuple[str, str], list[int]] = defaultdict(list)
    for key in keys:
        by_task[(key[0], key[1])].append(
            int(bool(challenger[key]["correct"])) - int(bool(baseline[key]["correct"]))
        )
    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        total_differences = np.zeros(replicates, dtype=np.int64)
        total_examples = 0
        for values in by_task.values():
            counts = [values.count(-1), values.count(0), values.count(1)]
            n = len(values)
            sampled = rng.multinomial(n, [count / n for count in counts], size=replicates)
            total_differences += sampled[:, 2] - sampled[:, 0]
            total_examples += n
        return (total_differences / total_examples).tolist()
    except ImportError:
        rng = random.Random(seed)
        samples = []
        total_examples = len(keys)
        task_values = list(by_task.values())
        for _ in range(replicates):
            delta = 0
            for values in task_values:
                delta += sum(rng.choice(values) for _ in values)
            samples.append(delta / total_examples)
        return samples


def semantic_digest(rows: dict[ExampleKey, dict]) -> str:
    payload = [
        [*key, str(row.get("normalized_prediction") or ""), bool(row.get("correct"))]
        for key, row in sorted(rows.items())
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def comparison(
    baseline_rows: dict[ExampleKey, dict],
    challenger_rows: dict[ExampleKey, dict],
    keys: list[ExampleKey],
    replicates: int,
    seed: int,
) -> dict:
    wins = sum(
        not baseline_rows[key]["correct"] and challenger_rows[key]["correct"] for key in keys
    )
    losses = sum(
        baseline_rows[key]["correct"] and not challenger_rows[key]["correct"] for key in keys
    )
    baseline_accuracy = sum(bool(baseline_rows[key]["correct"]) for key in keys) / len(keys)
    challenger_accuracy = sum(bool(challenger_rows[key]["correct"]) for key in keys) / len(keys)
    deltas = stratified_bootstrap_delta(
        baseline_rows,
        challenger_rows,
        keys,
        replicates,
        seed,
    )
    error_rate = 1.0 - baseline_accuracy
    delta = challenger_accuracy - baseline_accuracy
    return {
        "accuracy_difference": delta,
        "accuracy_point_difference": 100 * delta,
        "relative_error_reduction": delta / error_rate if error_rate else 0.0,
        "paired_wins": wins,
        "paired_losses": losses,
        "discordant_total": wins + losses,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(wins, losses),
        "task_stratified_bootstrap_95": [
            percentile(deltas, 0.025),
            percentile(deltas, 0.975),
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def benchmark_metrics(rows: dict[ExampleKey, dict], keys: list[ExampleKey]) -> dict:
    result = {}
    for benchmark in sorted({key[0] for key in keys}):
        selected = [key for key in keys if key[0] == benchmark]
        result[benchmark] = metrics(rows, selected)
    return result


def task_macro_accuracy(rows: dict[ExampleKey, dict], keys: list[ExampleKey]) -> float:
    by_task: dict[tuple[str, str], list[ExampleKey]] = defaultdict(list)
    for key in keys:
        by_task[(key[0], key[1])].append(key)
    accuracies = [
        sum(bool(rows[key]["correct"]) for key in task_keys) / len(task_keys)
        for task_keys in by_task.values()
    ]
    return statistics.mean(accuracies) if accuracies else 0.0


def load_examples(root: Path) -> dict[ExampleKey, Example]:
    examples = [
        *load_bbh(root, None),
        *load_bbeh(root, None),
        *load_unpuzzles_simple_reasoning(root, None),
    ]
    return {(row.benchmark, row.task, row.index): row for row in examples}


def markdown_text(value: str, limit: int = 1800) -> str:
    value = value.strip().replace("```", "` ` `")
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def benchmark_stratified_keys(
    keys: list[ExampleKey], limit_per_benchmark: int
) -> list[ExampleKey]:
    by_benchmark: dict[str, list[ExampleKey]] = defaultdict(list)
    for key in sorted(keys):
        by_benchmark[key[0]].append(key)
    return [
        key
        for benchmark in sorted(by_benchmark)
        for key in by_benchmark[benchmark][:limit_per_benchmark]
    ]


def write_examples(
    path: Path,
    examples: dict[ExampleKey, Example],
    baseline: dict[ExampleKey, dict],
    primary: dict[ExampleKey, dict],
    keys: list[ExampleKey],
) -> None:
    wins = [key for key in keys if not baseline[key]["correct"] and primary[key]["correct"]]
    losses = [key for key in keys if baseline[key]["correct"] and not primary[key]["correct"]]
    agreements = [key for key in keys if baseline[key]["correct"] and primary[key]["correct"]]
    chosen = [
        ("CBRR win", key) for key in benchmark_stratified_keys(wins, 3)
    ]
    chosen += [
        ("CBRR loss", key) for key in benchmark_stratified_keys(losses, 2)
    ]
    chosen += [
        ("Both correct", key) for key in benchmark_stratified_keys(agreements, 1)
    ]
    lines = [
        "# Confirmatory E2B Examples",
        "",
        "Examples are selected deterministically within each benchmark from the sorted "
        "untouched test rows after scoring.",
        "",
    ]
    for label, key in chosen:
        base = baseline[key]
        routed = primary[key]
        lines.extend(
            [
                f"## {label}: {key[0]}/{key[1]} #{key[2]}",
                "",
                f"- Ground truth: `{markdown_text(str(routed['target']), 500)}`",
                f"- Direct answer: `{markdown_text(str(base['prediction']), 500)}`",
                f"- Direct normalized: `{markdown_text(str(base['normalized_prediction']), 500)}`",
                f"- Direct correct: `{str(bool(base['correct'])).lower()}`",
                f"- CBRR strategy: `{routed.get('prompt_strategy', '')}`",
                f"- CBRR answer: `{markdown_text(str(routed['prediction']), 500)}`",
                f"- CBRR normalized: `{markdown_text(str(routed['normalized_prediction']), 500)}`",
                f"- CBRR correct: `{str(bool(routed['correct'])).lower()}`",
                "",
                "Question:",
                "",
                "```text",
                markdown_text(examples[key].input),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def robustness_summary(
    root: Path | None,
    primary_rows: dict[str, dict[ExampleKey, dict]],
    arms: tuple[str, str],
) -> dict:
    if root is None or not root.exists():
        return {"available": False}
    result = {"available": True, "seeds": {}}
    for seed_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        seed_payload = {}
        loaded = {}
        for arm in arms:
            arm_dir = seed_dir / arm
            if not arm_dir.exists():
                continue
            rows = load_arm_rows(arm_dir)
            loaded[arm] = rows
            keys = sorted(rows)
            agreement = None
            if arm in primary_rows and set(rows) == set(primary_rows[arm]):
                agreement = sum(
                    rows[key].get("normalized_prediction")
                    == primary_rows[arm][key].get("normalized_prediction")
                    for key in keys
                ) / len(keys)
            seed_payload[arm] = {
                **metrics(rows, keys),
                "semantic_sha256": semantic_digest(rows),
                "agreement_with_primary_seed": agreement,
            }
        if set(arms) <= set(loaded):
            keys = sorted(loaded[arms[0]])
            if set(keys) == set(loaded[arms[1]]):
                seed_payload["paired_accuracy_difference"] = (
                    sum(bool(loaded[arms[1]][key]["correct"]) for key in keys)
                    - sum(bool(loaded[arms[0]][key]["correct"]) for key in keys)
                ) / len(keys)
        result["seeds"][seed_dir.name] = seed_payload
    return result


def write_report(path: Path, analysis: dict) -> None:
    lines = [
        "# Gemma 4 E2B Confirmatory Results",
        "",
        f"Test rows: {analysis['test_examples']}. Primary seed: {analysis['primary_seed']}.",
        "",
        "## Finalist results",
        "",
        "| Arm | Correct | Accuracy | Delta (pp) | Bootstrap 95% (pp) | Wins/Losses | McNemar p | Holm p | Mean completion tokens | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in analysis["arm_order"]:
        arm_metrics_row = analysis["arms"][arm]
        if arm == analysis["baseline"]:
            delta = ci = discordance = p_value = adjusted = "-"
        else:
            comp = analysis["comparisons_vs_direct"][arm]
            delta = f"{comp['accuracy_point_difference']:+.2f}"
            ci = (
                f"[{100 * comp['task_stratified_bootstrap_95'][0]:+.2f}, "
                f"{100 * comp['task_stratified_bootstrap_95'][1]:+.2f}]"
            )
            discordance = f"{comp['paired_wins']}/{comp['paired_losses']}"
            p_value = f"{comp['exact_mcnemar_two_sided_p']:.4g}"
            adjusted = f"{comp['holm_adjusted_p']:.4g}"
        lines.append(
            f"| `{arm}` | {arm_metrics_row['correct']}/{arm_metrics_row['total']} | "
            f"{100 * arm_metrics_row['accuracy']:.2f}% | {delta} | {ci} | {discordance} | "
            f"{p_value} | {adjusted} | {arm_metrics_row['completion_tokens']['mean']:.2f} | "
            f"{arm_metrics_row['elapsed_seconds']['mean']:.3f} |"
        )
    primary = analysis["comparisons_vs_direct"][analysis["primary"]]
    lines.extend(
        [
            "",
            "## Primary comparison",
            "",
            f"CBRR changed the number of correct answers by "
            f"{primary['paired_wins'] - primary['paired_losses']:+d}: "
            f"{primary['paired_wins']} direct-to-correct wins and "
            f"{primary['paired_losses']} correct-to-wrong losses. The exact two-sided McNemar "
            f"p-value is {primary['exact_mcnemar_two_sided_p']:.6g}; the absolute effect is "
            f"{primary['accuracy_point_difference']:+.2f} percentage points.",
            "",
            "## Primary effect by benchmark",
            "",
            "| Benchmark | Direct | CBRR | Delta (pp) |",
            "|---|---:|---:|---:|",
        ]
    )
    for benchmark, payload in analysis["primary_by_benchmark"].items():
        lines.append(
            f"| `{benchmark}` | {100 * payload['direct']['accuracy']:.2f}% | "
            f"{100 * payload['primary']['accuracy']:.2f}% | "
            f"{100 * payload['accuracy_difference']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- Finalists were frozen from indices below 50 before test launch.",
            "- No system-role messages were sent.",
            "- Prediction records include per-example seeds, raw outputs, normalized outputs, usage, latency, and errors.",
            "- Prompt selection generalizes to new rows of known tasks; it is not an unseen-task router.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text())
    if not selection.get("selection_frozen") or selection.get("test_rows_read") != 0:
        raise SystemExit("selection integrity gate is not satisfied")

    arm_order = [str(row["name"]) for row in selection["finalists"]]
    rows_by_arm = {arm: load_arm_rows(args.test_root / arm) for arm in arm_order}
    if args.baseline not in rows_by_arm:
        raise SystemExit("baseline test arm is missing")
    if args.primary not in rows_by_arm:
        raise SystemExit("primary test arm is missing")
    all_keys = set(rows_by_arm[args.baseline])
    for arm, rows in rows_by_arm.items():
        if set(rows) != all_keys:
            raise SystemExit(f"test coverage mismatch for {arm}")
    keys = sorted(all_keys)
    if any(key[2] < 50 for key in keys):
        raise SystemExit("test predictions contain calibration or validation rows")

    arms = {}
    for arm in arm_order:
        arms[arm] = metrics(rows_by_arm[arm], keys)
        arms[arm]["macro_task_accuracy"] = task_macro_accuracy(rows_by_arm[arm], keys)
        arms[arm]["by_benchmark"] = benchmark_metrics(rows_by_arm[arm], keys)
        arms[arm]["semantic_sha256"] = semantic_digest(rows_by_arm[arm])

    comparisons = {}
    raw_p_values = {}
    for index, arm in enumerate(arm_order):
        if arm == args.baseline:
            continue
        comparisons[arm] = comparison(
            rows_by_arm[args.baseline],
            rows_by_arm[arm],
            keys,
            args.bootstrap_replicates,
            args.bootstrap_seed + index,
        )
        raw_p_values[arm] = comparisons[arm]["exact_mcnemar_two_sided_p"]
    adjusted = holm_adjust(raw_p_values)
    for arm in comparisons:
        comparisons[arm]["holm_adjusted_p"] = adjusted[arm]
        comparisons[arm]["significant_at_0_05_after_holm"] = adjusted[arm] < 0.05

    primary_by_benchmark = {}
    for benchmark in sorted({key[0] for key in keys}):
        benchmark_keys = [key for key in keys if key[0] == benchmark]
        direct_metrics = metrics(rows_by_arm[args.baseline], benchmark_keys)
        primary_metrics = metrics(rows_by_arm[args.primary], benchmark_keys)
        primary_by_benchmark[benchmark] = {
            "direct": direct_metrics,
            "primary": primary_metrics,
            "accuracy_difference": primary_metrics["accuracy"] - direct_metrics["accuracy"],
        }

    robustness = robustness_summary(
        args.robustness_root,
        {args.baseline: rows_by_arm[args.baseline], args.primary: rows_by_arm[args.primary]},
        (args.baseline, args.primary),
    )
    analysis = {
        "protocol": "gemma4-e2b-confirmatory-v1",
        "primary_seed": 20260709,
        "test_examples": len(keys),
        "baseline": args.baseline,
        "primary": args.primary,
        "arm_order": arm_order,
        "arms": arms,
        "comparisons_vs_direct": comparisons,
        "primary_by_benchmark": primary_by_benchmark,
        "robustness": robustness,
        "bootstrap_replicates": args.bootstrap_replicates,
        "system_messages_sent": 0,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    write_report(args.output_dir / "report.md", analysis)
    if args.datasets_root is not None:
        examples = load_examples(args.datasets_root)
        write_examples(
            args.output_dir / "examples.md",
            examples,
            rows_by_arm[args.baseline],
            rows_by_arm[args.primary],
            keys,
        )
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
