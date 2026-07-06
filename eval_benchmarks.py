#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from typing import Iterable
from urllib import error, request


@dataclass(frozen=True)
class Example:
    benchmark: str
    task: str
    index: int
    input: str
    target: str


def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[-1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[-1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[-1]
    return response


def extract_answer(sample: str) -> str:
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
        "Final answer:",
        "Answer:",
    ]
    answer = sample
    for prefix in answer_prefixes:
        if prefix in answer:
            answer = answer.split(prefix)[-1].strip()
    answer = answer.strip().strip("`").strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def preprocess_prediction(sample: str) -> str:
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0].strip()
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    prediction = prediction.strip("\"'")
    return prediction


def preprocess_reference(reference: str) -> str:
    return reference.strip().lower().replace(", ", ",").strip("\"'")


def fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False


def evaluate_correctness(sample: str, reference: str) -> bool:
    return fuzzy_match(preprocess_prediction(sample), preprocess_reference(reference))


def harmonic_mean(values: Iterable[float]) -> float:
    positives = [value for value in values if value > 0]
    values = list(values)
    if len(positives) != len(values) or not values:
        return 0.0
    return statistics.harmonic_mean(values)


def load_bbh(root: Path, limit_per_task: int | None) -> list[Example]:
    task_dir = root / "BIG-Bench-Hard" / "bbh"
    examples: list[Example] = []
    for task_file in sorted(task_dir.glob("*.json")):
        payload = json.loads(task_file.read_text())
        task = task_file.stem
        rows = payload.get("examples", [])
        if limit_per_task is not None:
            rows = rows[:limit_per_task]
        for index, row in enumerate(rows):
            examples.append(
                Example(
                    benchmark="bbh",
                    task=task,
                    index=index,
                    input=str(row["input"]),
                    target=str(row["target"]),
                )
            )
    return examples


def load_bbeh(root: Path, limit_per_task: int | None) -> list[Example]:
    task_root = root / "bbeh" / "bbeh" / "benchmark_tasks"
    examples: list[Example] = []
    for task_file in sorted(task_root.glob("*/task.json")):
        payload = json.loads(task_file.read_text())
        task = task_file.parent.name
        rows = payload.get("examples", [])
        if limit_per_task is not None:
            rows = rows[:limit_per_task]
        for index, row in enumerate(rows):
            examples.append(
                Example(
                    benchmark="bbeh",
                    task=task,
                    index=index,
                    input=str(row["input"]),
                    target=str(row["target"]),
                )
            )
    return examples


def build_prompt(example_input: str, mode: str) -> str:
    if mode == "raw":
        return example_input
    if mode == "answer_only":
        return (
            f"{example_input}\n\n"
            "Return only the final answer. Do not include reasoning, explanation, or extra text."
        )
    raise ValueError(f"unknown prompt mode: {mode}")


def post_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = request.Request(url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (error.HTTPError, error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"chat completion failed after {retries + 1} attempts: {last_error}")


def run_one(args: argparse.Namespace, example: Example) -> dict:
    prompt = build_prompt(example.input, args.prompt_mode)
    started = time.time()
    try:
        response = post_chat_completion(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
        )
        content = response["choices"][0]["message"].get("content") or ""
        correct = evaluate_correctness(content, example.target)
        error_text = None
        usage = response.get("usage", {})
    except Exception as exc:  # Keep long runs moving; errors are scored incorrect.
        content = ""
        correct = False
        error_text = str(exc)
        usage = {}
    elapsed = time.time() - started
    return {
        "benchmark": example.benchmark,
        "task": example.task,
        "index": example.index,
        "target": example.target,
        "prediction": content,
        "normalized_prediction": preprocess_prediction(content),
        "normalized_target": preprocess_reference(example.target),
        "correct": correct,
        "elapsed_seconds": elapsed,
        "usage": usage,
        "error": error_text,
    }


def summarize(records: list[dict]) -> dict:
    by_benchmark: dict[str, list[dict]] = {}
    by_task: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        by_benchmark.setdefault(record["benchmark"], []).append(record)
        by_task.setdefault((record["benchmark"], record["task"]), []).append(record)

    task_rows = []
    for (benchmark, task), rows in sorted(by_task.items()):
        correct = sum(1 for row in rows if row["correct"])
        total = len(rows)
        task_rows.append(
            {
                "benchmark": benchmark,
                "task": task,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )

    benchmark_rows = []
    for benchmark, rows in sorted(by_benchmark.items()):
        correct = sum(1 for row in rows if row["correct"])
        total = len(rows)
        task_accs = [
            row["accuracy"] for row in task_rows if row["benchmark"] == benchmark
        ]
        benchmark_rows.append(
            {
                "benchmark": benchmark,
                "correct": correct,
                "total": total,
                "micro_accuracy": correct / total if total else 0.0,
                "macro_accuracy": statistics.mean(task_accs) if task_accs else 0.0,
                "harmonic_task_accuracy": harmonic_mean(task_accs),
                "tasks": len(task_accs),
            }
        )

    return {"benchmarks": benchmark_rows, "tasks": task_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8888/v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "SubTokenLLM"))
    parser.add_argument("--benchmarks", default="bbh,bbeh")
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--prompt-mode", choices=["answer_only", "raw"], default="answer_only")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    examples: list[Example] = []
    if "bbh" in selected:
        examples.extend(load_bbh(args.datasets_root, args.limit_per_task))
    if "bbeh" in selected:
        examples.extend(load_bbeh(args.datasets_root, args.limit_per_task))
    if not examples:
        raise SystemExit("no examples loaded")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    records_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"

    records: list[dict] = []
    with records_path.open("w") as out:
        with ThreadPoolExecutor(max_workers=max(args.parallel, 1)) as executor:
            futures = [executor.submit(run_one, args, example) for example in examples]
            for n, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                if n % 25 == 0 or n == len(examples):
                    correct = sum(1 for row in records if row["correct"])
                    print(f"{n}/{len(examples)} correct={correct} acc={correct / n:.3f}", flush=True)

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "system_messages_sent": 0,
        "request_message_shape": [{"role": "user", "content": "<benchmark prompt>"}],
        "limit_per_task": args.limit_per_task,
        "parallel": args.parallel,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        **summarize(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

