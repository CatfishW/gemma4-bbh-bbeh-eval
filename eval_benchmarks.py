#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
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


@dataclass(frozen=True)
class PromptStrategy:
    name: str
    description: str
    template: str


PROMPT_STRATEGIES: dict[str, PromptStrategy] = {
    "raw": PromptStrategy(
        name="raw",
        description="Dataset input exactly as stored.",
        template="{input}",
    ),
    "direct_answer": PromptStrategy(
        name="direct_answer",
        description="Direct answer baseline; final answer only.",
        template=(
            "{input}\n\n"
            "Return only the final answer. Do not include reasoning, explanation, or extra text."
        ),
    ),
    "strict_json": PromptStrategy(
        name="strict_json",
        description="Strict JSON answer object.",
        template=(
            "{input}\n\n"
            "Return a valid JSON object only, exactly matching this schema: "
            '{{"answer": "<final answer>"}}'
        ),
    ),
    "concise_cot": PromptStrategy(
        name="concise_cot",
        description="Concise chain-of-thought with final answer delimiter.",
        template=(
            "{input}\n\n"
            "Think briefly and solve the problem. Keep the reasoning concise.\n"
            "End with exactly one line in this format: The final answer is: <answer>"
        ),
    ),
    "chain_of_draft": PromptStrategy(
        name="chain_of_draft",
        description="Chain-of-Draft: very short scratch reasoning before final answer.",
        template=(
            "{input}\n\n"
            "Use Chain-of-Draft: write only terse intermediate notes, no full sentences, "
            "then give the answer.\n"
            "End with exactly one line in this format: The final answer is: <answer>"
        ),
    ),
    "plan_and_solve": PromptStrategy(
        name="plan_and_solve",
        description="Plan-and-Solve with final answer delimiter.",
        template=(
            "{input}\n\n"
            "First write a short plan. Then solve according to the plan.\n"
            "End with exactly one line in this format: The final answer is: <answer>"
        ),
    ),
    "step_back": PromptStrategy(
        name="step_back",
        description="Step-back prompting: identify the general rule/principle, then answer.",
        template=(
            "{input}\n\n"
            "Step back and identify the general rule, pattern, or principle needed. "
            "Then apply it to this specific problem.\n"
            "End with exactly one line in this format: The final answer is: <answer>"
        ),
    ),
    "premise_conclusion": PromptStrategy(
        name="premise_conclusion",
        description="Explicit premise-to-conclusion reasoning template.",
        template=(
            "{input}\n\n"
            "List the key premises briefly, derive the conclusion, and avoid irrelevant text.\n"
            "End with exactly one line in this format: The final answer is: <answer>"
        ),
    ),
    "symbolic_proof": PromptStrategy(
        name="symbolic_proof",
        description="Symbolic translation or proof sketch before final answer.",
        template=(
            "{input}\n\n"
            "Translate the problem into compact symbols, equations, constraints, or a proof sketch "
            "when useful. Then solve.\n"
            "End with exactly one line in this format: The final answer is: <answer>"
        ),
    ),
}


def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    boxed = re.findall(r"boxed\{([^}]*)\}", response)
    if boxed:
        response = boxed[-1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[-1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[-1]
    return response


def extract_answer(sample: str) -> str:
    json_answer = extract_json_answer(sample)
    if json_answer is not None:
        return json_answer
    xml_answer = extract_xml_answer(sample)
    if xml_answer is not None:
        return xml_answer
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
    count_answer = extract_count_answer(answer)
    if count_answer is not None:
        return count_answer
    return strip_latex(answer)


def extract_xml_answer(sample: str) -> str | None:
    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        sample,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1).strip()


def extract_count_answer(sample: str) -> str | None:
    match = re.search(r"\bappear(?:s)?\s+(-?\d+)\s+time", sample, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def extract_json_answer(sample: str) -> str | None:
    candidate = sample.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("answer", "final_answer", "final"):
        value = payload.get(key)
        if value is not None:
            return str(value).strip()
    return None


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


def unpuzzles_repo_root(root: Path) -> Path:
    candidates = [
        root / "unpuzzles_and_simple_reasoning",
        root,
    ]
    for candidate in candidates:
        if (candidate / "datasets" / "simple_reasoning.json").exists():
            return candidate
    raise FileNotFoundError(
        "could not find unpuzzles_and_simple_reasoning/datasets under "
        f"{root}"
    )


def load_unpuzzles_simple_reasoning(root: Path, limit_per_task: int | None) -> list[Example]:
    repo = unpuzzles_repo_root(root)
    examples: list[Example] = []
    examples.extend(load_simple_reasoning(repo, limit_per_task))
    examples.extend(load_unpuzzles(repo, limit_per_task))
    examples.extend(load_shifted_unpuzzles(repo, limit_per_task))
    return examples


def load_simple_reasoning(repo: Path, limit_per_task: int | None) -> list[Example]:
    rows = json.loads((repo / "datasets" / "simple_reasoning.json").read_text())
    examples: list[Example] = []
    per_task_counts: dict[str, int] = {}
    for row in rows:
        source_task = str(row["task"])
        count = per_task_counts.get(source_task, 0)
        if limit_per_task is not None and count >= limit_per_task:
            continue
        per_task_counts[source_task] = count + 1
        examples.append(
            Example(
                benchmark="usr",
                task=f"simple_reasoning/{source_task}",
                index=count,
                input=str(row["input"]),
                target=str(row.get("text_target", row["target"])),
            )
        )
    return examples


def load_unpuzzles(repo: Path, limit_per_task: int | None) -> list[Example]:
    rows = json.loads((repo / "datasets" / "unpuzzles.json").read_text())
    specs = [
        ("unpuzzles/original", "original_puzzle", "original_answer"),
        ("unpuzzles/unpuzzle", "unpuzzle", "unpuzzle_answer"),
    ]
    return load_unpuzzle_variants(rows, specs, limit_per_task)


def load_shifted_unpuzzles(repo: Path, limit_per_task: int | None) -> list[Example]:
    rows = json.loads((repo / "datasets" / "shifted_unpuzzles.json").read_text())
    specs = [
        ("shifted_unpuzzles/original", "original_puzzle", "original_answer"),
        ("shifted_unpuzzles/unpuzzle", "unpuzzle", "unpuzzle_answer"),
        ("shifted_unpuzzles/shifted", "shifted_unpuzzle", "shifted_unpuzzle_answer"),
    ]
    return load_unpuzzle_variants(rows, specs, limit_per_task)


def load_unpuzzle_variants(
    rows: list[dict],
    specs: list[tuple[str, str, str]],
    limit_per_task: int | None,
) -> list[Example]:
    examples: list[Example] = []
    per_task_counts: dict[str, int] = {}
    for row in rows:
        name = str(row.get("puzzle_name", "")).strip()
        for task, question_key, answer_key in specs:
            target = str(row.get(answer_key, "")).strip()
            question = str(row.get(question_key, "")).strip()
            if not target or not question:
                continue
            count = per_task_counts.get(task, 0)
            if limit_per_task is not None and count >= limit_per_task:
                continue
            per_task_counts[task] = count + 1
            prompt = question
            if name:
                prompt = f"Puzzle: {name}\n\n{question}"
            examples.append(
                Example(
                    benchmark="usr",
                    task=task,
                    index=count,
                    input=prompt,
                    target=target,
                )
            )
    return examples


def build_prompt(example_input: str, strategy_name: str) -> str:
    try:
        strategy = PROMPT_STRATEGIES[strategy_name]
    except KeyError as exc:
        raise ValueError(f"unknown prompt strategy: {strategy_name}") from exc
    return strategy.template.format(input=example_input)


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
    prompt = build_prompt(example.input, args.prompt_strategy)
    started = time.time()
    generations = []
    try:
        for sample_index in range(args.self_consistency_k):
            response = post_chat_completion(
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
            )
            sample_content = response["choices"][0]["message"].get("content") or ""
            generations.append(
                {
                    "sample_index": sample_index,
                    "prediction": sample_content,
                    "normalized_prediction": preprocess_prediction(sample_content),
                    "usage": response.get("usage", {}),
                }
            )
        chosen_normalized = majority_vote(generations)
        chosen = next(
            item["prediction"]
            for item in generations
            if item["normalized_prediction"] == chosen_normalized
        )
        content = chosen
        correct = evaluate_correctness(content, example.target)
        error_text = None
    except Exception as exc:  # Keep long runs moving; errors are scored incorrect.
        content = ""
        correct = False
        error_text = str(exc)
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
        "self_consistency_k": args.self_consistency_k,
        "generations": generations,
        "usage": combine_usage([item.get("usage", {}) for item in generations]),
        "error": error_text,
    }


def majority_vote(generations: list[dict]) -> str:
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for index, item in enumerate(generations):
        key = str(item["normalized_prediction"])
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, index)
    return sorted(counts, key=lambda key: (-counts[key], first_seen[key]))[0]


def combine_usage(usages: list[dict]) -> dict:
    combined: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, int):
                combined[key] = combined.get(key, 0) + value
    return combined


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
    parser.add_argument(
        "--prompt-strategy",
        choices=sorted(PROMPT_STRATEGIES),
        default="direct_answer",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["answer_only", "raw"],
        help="Backward-compatible alias for --prompt-strategy.",
    )
    parser.add_argument("--self-consistency-k", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.prompt_mode == "answer_only":
        args.prompt_strategy = "direct_answer"
    elif args.prompt_mode == "raw":
        args.prompt_strategy = "raw"
    if args.self_consistency_k < 1:
        raise SystemExit("--self-consistency-k must be >= 1")
    return args


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_config(args: argparse.Namespace, examples: list[Example]) -> None:
    strategy = PROMPT_STRATEGIES[args.prompt_strategy]
    by_benchmark: dict[str, int] = {}
    by_task: dict[str, int] = {}
    for example in examples:
        by_benchmark[example.benchmark] = by_benchmark.get(example.benchmark, 0) + 1
        key = f"{example.benchmark}/{example.task}"
        by_task[key] = by_task.get(key, 0) + 1
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "benchmarks": args.benchmarks,
        "limit_per_task": args.limit_per_task,
        "parallel": args.parallel,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "retries": args.retries,
        "prompt_strategy": strategy.name,
        "prompt_strategy_description": strategy.description,
        "prompt_template": strategy.template,
        "self_consistency_k": args.self_consistency_k,
        "system_messages_sent": 0,
        "request_message_shape": [{"role": "user", "content": "<rendered prompt>"}],
        "example_count": len(examples),
        "examples_by_benchmark": by_benchmark,
        "examples_by_task": by_task,
        "dataset_revisions": {
            "bbh": git_revision(args.datasets_root / "BIG-Bench-Hard"),
            "bbeh": git_revision(args.datasets_root / "bbeh"),
            "usr": git_revision(args.datasets_root / "unpuzzles_and_simple_reasoning"),
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> int:
    args = parse_args()
    selected = {item.strip() for item in args.benchmarks.split(",") if item.strip()}
    examples: list[Example] = []
    if "bbh" in selected:
        examples.extend(load_bbh(args.datasets_root, args.limit_per_task))
    if "bbeh" in selected:
        examples.extend(load_bbeh(args.datasets_root, args.limit_per_task))
    if selected & {"usr", "unpuzzles_and_simple_reasoning"}:
        examples.extend(load_unpuzzles_simple_reasoning(args.datasets_root, args.limit_per_task))
    if "simple_reasoning" in selected:
        repo = unpuzzles_repo_root(args.datasets_root)
        examples.extend(load_simple_reasoning(repo, args.limit_per_task))
    if "unpuzzles" in selected:
        repo = unpuzzles_repo_root(args.datasets_root)
        examples.extend(load_unpuzzles(repo, args.limit_per_task))
    if "shifted_unpuzzles" in selected:
        repo = unpuzzles_repo_root(args.datasets_root)
        examples.extend(load_shifted_unpuzzles(repo, args.limit_per_task))
    if not examples:
        raise SystemExit("no examples loaded")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(args, examples)
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
        "prompt_strategy": args.prompt_strategy,
        "prompt_strategy_description": PROMPT_STRATEGIES[args.prompt_strategy].description,
        "prompt_template": PROMPT_STRATEGIES[args.prompt_strategy].template,
        "self_consistency_k": args.self_consistency_k,
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
