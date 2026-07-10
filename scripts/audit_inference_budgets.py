#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit generation and selection token-cap binding across strategy runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run root containing screening/, arms/, or direct arm directories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--run must use LABEL=PATH: {value}")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise SystemExit(f"--run must use LABEL=PATH: {value}")
    return label, Path(raw_path)


def arm_root(run_root: Path) -> Path:
    for name in ("screening", "arms"):
        candidate = run_root / name
        if candidate.is_dir():
            return candidate
    return run_root


def prediction_path(root: Path) -> Path:
    for candidate in (root / "predictions.jsonl", root / "predictions.jsonl.gz"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing predictions under {root}")


def read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def completion_tokens(usage: dict | None) -> int:
    usage = usage or {}
    return int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)


def accuracy(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def arm_budget_metrics(label: str, name: str, config: dict, rows: list[dict]) -> dict:
    generation_cap = int(config["max_tokens"])
    selection_cap = int(config.get("selection_max_tokens") or 0)
    capped_rows = []
    uncapped_rows = []
    generation_tokens = []
    generation_capped = 0
    selection_tokens = []
    selection_capped = 0
    final_marker = 0
    for row in rows:
        generations = row.get("generations") or []
        row_is_capped = False
        for generation in generations:
            tokens = completion_tokens(generation.get("usage"))
            generation_tokens.append(tokens)
            if tokens >= generation_cap:
                generation_capped += 1
                row_is_capped = True
        selection = row.get("selection")
        if selection is not None:
            tokens = completion_tokens(selection.get("usage"))
            selection_tokens.append(tokens)
            if selection_cap and tokens >= selection_cap:
                selection_capped += 1
                row_is_capped = True
        (capped_rows if row_is_capped else uncapped_rows).append(row)
        prediction = str(row.get("prediction") or "").lower()
        if "the final answer is" in prediction or "<answer>" in prediction:
            final_marker += 1

    total_completion = [completion_tokens(row.get("usage")) for row in rows]
    return {
        "model_label": label,
        "arm": name,
        "prompt_strategy": str(config.get("prompt_strategy") or ""),
        "response_selection": str(config.get("response_selection") or ""),
        "self_consistency_k": int(config.get("self_consistency_k") or 1),
        "examples": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": accuracy(rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "generation_max_tokens": generation_cap,
        "generation_samples": len(generation_tokens),
        "generation_cap_bindings": generation_capped,
        "generation_cap_binding_rate": (
            generation_capped / len(generation_tokens) if generation_tokens else 0.0
        ),
        "examples_with_any_cap_binding": len(capped_rows),
        "example_cap_binding_rate": len(capped_rows) / len(rows) if rows else 0.0,
        "accuracy_with_cap_binding": accuracy(capped_rows),
        "accuracy_without_cap_binding": accuracy(uncapped_rows),
        "selection_max_tokens": selection_cap,
        "selection_samples": len(selection_tokens),
        "selection_cap_bindings": selection_capped,
        "selection_cap_binding_rate": (
            selection_capped / len(selection_tokens) if selection_tokens else 0.0
        ),
        "mean_total_completion_tokens": (
            statistics.mean(total_completion) if total_completion else 0.0
        ),
        "p95_total_completion_tokens": percentile(total_completion, 0.95),
        "final_answer_marker_rate": final_marker / len(rows) if rows else 0.0,
    }


def collect_run(label: str, run_root: Path) -> list[dict]:
    root = arm_root(run_root)
    results = []
    for arm_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = arm_dir / "run_config.json"
        if not config_path.exists():
            continue
        try:
            predictions = prediction_path(arm_dir)
        except FileNotFoundError:
            continue
        config = json.loads(config_path.read_text())
        rows = list(read_jsonl(predictions))
        results.append(arm_budget_metrics(label, arm_dir.name, config, rows))
    if not results:
        raise SystemExit(f"no completed arm predictions found under {run_root}")
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def display_accuracy(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def write_report(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Inference-Budget Audit",
        "",
        "Cap binding is reported separately from accuracy so truncated reasoning is not "
        "misinterpreted as evidence that additional computation is intrinsically harmful.",
        "Capped-versus-uncapped accuracies are descriptive, not causal: easier examples may "
        "naturally terminate sooner.",
        "",
        "| Model | Arm | Accuracy | Generation cap rate | Example cap rate | Capped accuracy | Uncapped accuracy | Mean completion tokens | Errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | `{row['arm']}` | "
            f"{display_accuracy(row['accuracy'])} | "
            f"{100 * row['generation_cap_binding_rate']:.2f}% | "
            f"{100 * row['example_cap_binding_rate']:.2f}% | "
            f"{display_accuracy(row['accuracy_with_cap_binding'])} | "
            f"{display_accuracy(row['accuracy_without_cap_binding'])} | "
            f"{row['mean_total_completion_tokens']:.2f} | {row['errors']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    rows = []
    for value in args.run:
        rows.extend(collect_run(*parse_run(value)))
    rows.sort(key=lambda row: (row["model_label"], row["arm"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inference_budget_audit.json").write_text(
        json.dumps({"arms": rows}, indent=2) + "\n"
    )
    write_csv(args.output_dir / "inference_budget_audit.csv", rows)
    write_report(args.output_dir / "inference_budget_audit.md", rows)
    print(json.dumps({"arms": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
