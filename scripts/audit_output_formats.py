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
    parser = argparse.ArgumentParser(description="Audit strict-output compliance separately from accuracy.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=path to a strict_json arm directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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


def parse_raw_json(value: str) -> dict | None:
    try:
        payload = json.loads(value.strip())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def remove_single_code_fence(value: str) -> str:
    lines = value.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value.strip()


def exact_answer_schema(payload: dict | None) -> bool:
    return payload is not None and set(payload) == {"answer"} and isinstance(payload["answer"], str)


def completion_tokens(row: dict) -> int:
    usage = row.get("usage") or {}
    return int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)


def audit_rows(rows: list[dict]) -> dict:
    raw_payloads = [parse_raw_json(str(row.get("prediction") or "")) for row in rows]
    recovered_text = [remove_single_code_fence(str(row.get("prediction") or "")) for row in rows]
    recovered_payloads = [parse_raw_json(value) for value in recovered_text]
    raw_valid = [payload is not None for payload in raw_payloads]
    raw_schema = [exact_answer_schema(payload) for payload in raw_payloads]
    recovered_valid = [payload is not None for payload in recovered_payloads]
    recovered_schema = [exact_answer_schema(payload) for payload in recovered_payloads]
    correct = [bool(row.get("correct")) for row in rows]
    tokens = [completion_tokens(row) for row in rows]
    return {
        "examples": len(rows),
        "answer_correct": sum(correct),
        "answer_accuracy": sum(correct) / len(rows) if rows else 0.0,
        "raw_json_valid": sum(raw_valid),
        "raw_json_valid_rate": sum(raw_valid) / len(rows) if rows else 0.0,
        "raw_exact_schema_valid": sum(raw_schema),
        "raw_exact_schema_valid_rate": sum(raw_schema) / len(rows) if rows else 0.0,
        "recoverable_json_valid": sum(recovered_valid),
        "recoverable_json_valid_rate": sum(recovered_valid) / len(rows) if rows else 0.0,
        "recoverable_exact_schema_valid": sum(recovered_schema),
        "recoverable_exact_schema_valid_rate": (
            sum(recovered_schema) / len(rows) if rows else 0.0
        ),
        "markdown_fenced": sum(text != str(row.get("prediction") or "").strip() for text, row in zip(recovered_text, rows)),
        "correct_and_raw_schema_valid": sum(a and b for a, b in zip(correct, raw_schema)),
        "correct_and_recoverable_schema_valid": sum(
            a and b for a, b in zip(correct, recovered_schema)
        ),
        "mean_completion_tokens": statistics.mean(tokens) if tokens else 0.0,
        "request_errors": sum(bool(row.get("error")) for row in rows),
    }


def markdown_text(value: str, limit: int = 900) -> str:
    value = value.strip().replace("```", "` ` `")
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    invalid_examples = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit("--run must use LABEL=PATH")
        label, raw_path = spec.split("=", 1)
        root = Path(raw_path)
        rows = list(read_jsonl(prediction_path(root)))
        payload = {"label": label, "path": str(root), **audit_rows(rows)}
        payload["by_benchmark"] = {
            benchmark: audit_rows([row for row in rows if row["benchmark"] == benchmark])
            for benchmark in sorted({str(row["benchmark"]) for row in rows})
        }
        results.append(payload)
        for row in rows:
            parsed = parse_raw_json(str(row.get("prediction") or ""))
            if exact_answer_schema(parsed):
                continue
            invalid_examples.append((label, row))
            if sum(item[0] == label for item in invalid_examples) >= 8:
                break

    summary = {"runs": results, "metric_note": "Raw exact schema requires a JSON object with only one string answer field and no Markdown fence."}
    (args.output_dir / "format_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [key for key in results[0] if key not in {"by_benchmark"}]
    with (args.output_dir / "format_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in results)

    lines = [
        "# Strict JSON Format Audit",
        "",
        "Raw validity requires exactly one JSON object with one string `answer` field and no Markdown fence. Recoverable validity permits one surrounding code fence.",
        "",
        "| Run | Answer accuracy | Raw schema | Recoverable schema | Fenced | Mean tokens | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| `{row['label']}` | {100 * row['answer_accuracy']:.2f}% | "
            f"{100 * row['raw_exact_schema_valid_rate']:.2f}% | "
            f"{100 * row['recoverable_exact_schema_valid_rate']:.2f}% | "
            f"{row['markdown_fenced']}/{row['examples']} | "
            f"{row['mean_completion_tokens']:.2f} | {row['request_errors']} |"
        )
    lines.extend(["", "## Invalid examples", ""])
    for label, row in invalid_examples:
        lines.extend(
            [
                f"### {label}: {row['benchmark']}/{row['task']} #{row['index']}",
                "",
                f"- Ground truth: `{markdown_text(str(row['target']), 300)}`",
                f"- Answer correct: `{bool(row['correct'])}`",
                "- Raw response:",
                "",
                "```text",
                markdown_text(str(row.get("prediction") or "")),
                "```",
                "",
            ]
        )
    (args.output_dir / "format_audit.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
