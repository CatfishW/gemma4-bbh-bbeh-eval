#!/usr/bin/env python3
"""Summarize RL training runs and evaluation outputs into markdown tables.

Reads metrics.jsonl from one or more training run directories and optional
eval summary.json files, and emits:
- a training-efficiency table (validation-probe accuracy vs cumulative
  generated tokens at matched budgets)
- a final-results table (frozen-split accuracies per benchmark, mean
  completion tokens)
- per-run diagnostics (nonzero-advantage fraction, allocation entropy,
  reward trajectory)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def probe_rows(metrics: list[dict]) -> list[dict]:
    return [row for row in metrics if "val_probe_accuracy" in row]


def training_efficiency_table(runs: dict[str, list[dict]]) -> str:
    lines = [
        "| Run | Iteration | Cumulative generated tokens | Val-probe accuracy | Mean reward | Nonzero-advantage fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in runs.items():
        for row in probe_rows(metrics):
            lines.append(
                "| {name} | {it} | {tokens:,} | {acc:.2%} | {reward:.3f} | {adv:.2f} |".format(
                    name=name,
                    it=row["iteration"] + 1,
                    tokens=row["cumulative_generated_tokens"],
                    acc=row["val_probe_accuracy"],
                    reward=row["mean_reward"],
                    adv=row["nonzero_advantage_fraction"],
                )
            )
    return "\n".join(lines)


def diagnostics_table(runs: dict[str, list[dict]]) -> str:
    lines = [
        "| Run | Iterations | Total generated tokens | Mean nonzero-adv | Mean alloc entropy | Final mean reward | Mean peak GB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in runs.items():
        if not metrics:
            continue
        nonzero = [m["nonzero_advantage_fraction"] for m in metrics]
        entropy = [m.get("allocation_entropy", 0.0) for m in metrics]
        peaks = [m["peak_memory_gb"] for m in metrics if "peak_memory_gb" in m]
        tail = metrics[-min(5, len(metrics)) :]
        lines.append(
            "| {name} | {n} | {tokens:,} | {adv:.2f} | {ent:.2f} | {reward:.3f} | {peak:.1f} |".format(
                name=name,
                n=len(metrics),
                tokens=metrics[-1]["cumulative_generated_tokens"],
                adv=sum(nonzero) / len(nonzero),
                ent=sum(entropy) / len(entropy),
                reward=sum(m["mean_reward"] for m in tail) / len(tail),
                peak=sum(peaks) / len(peaks) if peaks else 0.0,
            )
        )
    return "\n".join(lines)


def eval_table(eval_summaries: dict[str, Path]) -> str:
    lines = [
        "| Eval | Split | Strategy | Overall | BBH | BBEH | USR | Mean completion tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, path in eval_summaries.items():
        payload = json.loads(path.read_text())
        overall = payload["overall"]
        by_benchmark = payload["by_benchmark"]

        def cell(bench: str) -> str:
            entry = by_benchmark.get(bench)
            if not entry or not entry["examples"]:
                return "-"
            return f"{entry['correct']}/{entry['examples']} ({entry['accuracy']:.2%})"

        lines.append(
            "| {name} | {split} | {strategy} | {corr}/{total} ({acc:.2%}) | {bbh} | {bbeh} | {usr} | {tokens:.1f} |".format(
                name=name,
                split=payload["split"],
                strategy=payload["prompt_strategy"],
                corr=overall["correct"],
                total=overall["examples"],
                acc=overall["accuracy"],
                bbh=cell("bbh"),
                bbeh=cell("bbeh"),
                usr=cell("usr"),
                tokens=overall["mean_completion_tokens"],
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="training run directory, repeatable",
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        metavar="NAME=SUMMARY_JSON",
        help="eval summary.json, repeatable",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs: dict[str, list[dict]] = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name] = load_metrics(Path(path))
    evals: dict[str, Path] = {}
    for spec in args.eval:
        name, _, path = spec.partition("=")
        evals[name] = Path(path)

    sections = []
    if runs:
        sections.append("## Training diagnostics\n\n" + diagnostics_table(runs))
        sections.append(
            "## Validation-probe accuracy vs generated tokens\n\n"
            + training_efficiency_table(runs)
        )
    if evals:
        sections.append("## Frozen-split evaluations\n\n" + eval_table(evals))
    report = "\n\n".join(sections) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
