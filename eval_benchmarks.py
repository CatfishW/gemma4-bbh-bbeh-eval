#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
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
    "native_format": PromptStrategy(
        name="native_format",
        description="Direct answer while preserving the format requested by the benchmark item.",
        template=(
            "{input}\n\n"
            "Answer in exactly the format requested by the question. If the question asks for "
            "<answer> tags, use those tags. If it asks for a sentence such as 'The word appears "
            "# times.', use that sentence. If it gives options, output only the option label. "
            "Do not include reasoning or extra text."
        ),
    ),
    "canonical_short": PromptStrategy(
        name="canonical_short",
        description="Canonical shortest answer with explicit normalization rules.",
        template=(
            "{input}\n\n"
            "Output only the canonical final answer.\n"
            "- For multiple choice, output exactly one option label such as (A) or A.\n"
            "- For yes/no questions, output exactly Yes or No.\n"
            "- For numeric answers, output digits only unless units are required.\n"
            "- For lists, output the list only.\n"
            "No explanation."
        ),
    ),
    "private_verify": PromptStrategy(
        name="private_verify",
        description="Solve and verify privately, then emit only the final answer.",
        template=(
            "{input}\n\n"
            "Solve the problem privately, check the answer once for mistakes, and then output "
            "only the final answer. Do not show reasoning, notes, or verification."
        ),
    ),
    "selective_verify": PromptStrategy(
        name="selective_verify",
        description="Verify only concrete failure modes and otherwise preserve the first answer.",
        template=(
            "{input}\n\n"
            "Solve directly. Before answering, look for one concrete contradiction caused by a "
            "missed negation, ignored constraint, arithmetic or sign error, or option-label mismatch. "
            "Keep the first answer unless that check finds a specific error. Output only the final "
            "answer, with no reasoning or extra text."
        ),
    ),
    "compare_then_commit": PromptStrategy(
        name="compare_then_commit",
        description="Contrast the two strongest candidates privately before committing.",
        template=(
            "{input}\n\n"
            "Privately identify the two most plausible answers. Compare both against the exact "
            "question and its constraints, reject the weaker candidate, and commit to one answer. "
            "Output only the final answer, with no reasoning or extra text."
        ),
    ),
    "fast_slow_gate": PromptStrategy(
        name="fast_slow_gate",
        description="Use a direct answer unless uncertainty warrants one verification pass.",
        template=(
            "{input}\n\n"
            "Solve once directly. If the answer is clear, keep it. Only if two answers remain "
            "plausible or a condition may have been missed, take one brief private verification "
            "pass and then commit. Output only the final answer."
        ),
    ),
    "constraint_guard": PromptStrategy(
        name="constraint_guard",
        description="Check the candidate against the decisive constraints before answering.",
        template=(
            "{input}\n\n"
            "Privately identify the decisive constraints, derive a candidate, and test that "
            "candidate against each decisive constraint once. Then output only the final answer. "
            "Do not include the constraint list or any explanation."
        ),
    ),
    "negation_label_guard": PromptStrategy(
        name="negation_label_guard",
        description="Protect negations, quantifiers, and answer-choice label mapping.",
        template=(
            "{input}\n\n"
            "Pay special attention to NOT, EXCEPT, all, none, and other negations or quantifiers. "
            "For multiple choice, decide the answer content first and then map it to the exact "
            "option label. Output only that label; otherwise output only the shortest final answer."
        ),
    ),
    "draft_verify": PromptStrategy(
        name="draft_verify",
        description="Use a terse private draft followed by one targeted check.",
        template=(
            "{input}\n\n"
            "Make a very short private scratch draft using keywords, symbols, or equations rather "
            "than prose. Check the resulting answer once against the exact question, then output "
            "only the final answer. Do not show the draft or verification."
        ),
    ),
    "option_elimination": PromptStrategy(
        name="option_elimination",
        description="Multiple-choice elimination done privately with answer-only output.",
        template=(
            "{input}\n\n"
            "If answer choices are provided, compare the choices privately and eliminate wrong "
            "ones before deciding. Output only the final option label. If there are no choices, "
            "output only the shortest final answer. No explanation."
        ),
    ),
    "answer_type_router": PromptStrategy(
        name="answer_type_router",
        description="Route to answer type, then output the shortest parseable answer.",
        template=(
            "{input}\n\n"
            "First identify the required answer type privately: option label, boolean, number, "
            "word/phrase, tuple/list, or requested tag format. Then output only that answer in "
            "the most parseable form. Do not include reasoning."
        ),
    ),
    "careful_direct": PromptStrategy(
        name="careful_direct",
        description="Direct answer with a careful-read instruction and no visible reasoning.",
        template=(
            "{input}\n\n"
            "Read the problem carefully, including every condition and answer-choice label. "
            "Compute or infer the answer, then return only the final answer. No explanation."
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
    "plan_and_solve_plus": PromptStrategy(
        name="plan_and_solve_plus",
        description="Detailed Plan-and-Solve with explicit variable extraction and verification.",
        template=(
            "{input}\n\n"
            "First identify the relevant facts, variables, and the exact quantity or label requested. "
            "Make a compact plan, carry it out without skipping calculations or constraints, and "
            "verify the result against the question.\n"
            "End with exactly one line in this format: The final answer is: <answer>"
        ),
    ),
    "least_to_most": PromptStrategy(
        name="least_to_most",
        description="Decompose into simpler subproblems and solve them in dependency order.",
        template=(
            "{input}\n\n"
            "Break the problem into the smallest useful subproblems. Solve them from simplest to "
            "hardest, carrying each result forward exactly once.\n"
            "End with exactly one line in this format: The final answer is: <answer>"
        ),
    ),
    "condition_reconstruction": PromptStrategy(
        name="condition_reconstruction",
        description="Novel single-pass adaptation of key-condition verification.",
        template=(
            "{input}\n\n"
            "Privately derive a candidate answer. Identify the one condition most capable of making "
            "that candidate wrong, temporarily hide that condition, and reconstruct what it must be "
            "from the candidate. Compare the reconstruction with the actual condition and correct "
            "the candidate if they conflict. Output only the final answer, with no explanation."
        ),
    ),
    "counterexample_guard": PromptStrategy(
        name="counterexample_guard",
        description="Try one targeted counterexample or violated constraint before committing.",
        template=(
            "{input}\n\n"
            "Solve directly, then privately try to disprove the candidate with one targeted "
            "counterexample or violated constraint. Change it only if that check succeeds. Output "
            "only the final answer, with no reasoning or extra text."
        ),
    ),
    "rank_two_paths": PromptStrategy(
        name="rank_two_paths",
        description="Generate two compact reasoning paths privately and rank them before answering.",
        template=(
            "{input}\n\n"
            "Privately form two genuinely different compact solution paths. Compare their crucial "
            "steps against the given facts, rank the paths by correctness, and use the stronger one. "
            "Output only the final answer, with no reasoning or extra text."
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


def example_task_key(example: Example) -> str:
    return f"{example.benchmark}/{example.task}"


def generation_seed(base_seed: int, example: Example, sample_index: int) -> int:
    identity = (
        f"{base_seed}|{example.benchmark}|{example.task}|{example.index}|{sample_index}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def resolve_prompt_strategy(args: argparse.Namespace, example: Example) -> str:
    policy = getattr(args, "prompt_policy_map", {})
    default_strategy = getattr(args, "prompt_policy_default", args.prompt_strategy)
    return policy.get(example_task_key(example), default_strategy)


def prompt_run_metadata(args: argparse.Namespace) -> dict:
    policy = getattr(args, "prompt_policy_payload", None)
    if policy is not None:
        name = str(policy.get("name") or args.prompt_policy.stem)
        description = str(
            policy.get("description")
            or "Task-conditioned prompt strategy selected from calibration rewards."
        )
        selected = sorted(set(args.prompt_policy_map.values()) | {args.prompt_policy_default})
        return {
            "prompt_strategy": f"policy:{name}",
            "prompt_strategy_description": description,
            "prompt_template": None,
            "prompt_policy_path": str(args.prompt_policy),
            "prompt_policy_name": name,
            "prompt_policy_default": args.prompt_policy_default,
            "prompt_policy_selected_strategies": selected,
            "prompt_policy": policy,
        }
    strategy = PROMPT_STRATEGIES[args.prompt_strategy]
    return {
        "prompt_strategy": strategy.name,
        "prompt_strategy_description": strategy.description,
        "prompt_template": strategy.template,
    }


def build_selection_prompt(
    mode: str,
    question: str,
    generations: list[dict],
) -> str:
    candidates = "\n\n".join(
        f"Candidate {index + 1}:\n{item['prediction']}"
        for index, item in enumerate(generations)
    )
    if mode == "self_rank":
        return (
            f"Question:\n{question}\n\n{candidates}\n\n"
            "Compare the candidates against the exact question and every decisive constraint. "
            "Do not vote by wording or length. Select or correct the answer that is best supported. "
            "Return only the final answer, with no explanation."
        )
    if mode == "key_condition_refine":
        return (
            f"Question:\n{question}\n\n{candidates}\n\n"
            "Verify the candidate by identifying the single key condition most likely to expose an "
            "error. Reconstruct or recompute that condition independently, correct the candidate if "
            "needed, and return only the final answer with no explanation."
        )
    raise ValueError(f"unknown response selection mode: {mode}")


def post_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    timeout: int,
    retries: int,
) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
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
    strategy_name = resolve_prompt_strategy(args, example)
    prompt = build_prompt(example.input, strategy_name)
    started = time.time()
    generations = []
    selection = None
    try:
        for sample_index in range(args.self_consistency_k):
            seed = generation_seed(args.seed, example, sample_index)
            response = post_chat_completion(
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=seed,
                timeout=args.timeout,
                retries=args.retries,
            )
            sample_content = response["choices"][0]["message"].get("content") or ""
            generations.append(
                {
                    "sample_index": sample_index,
                    "seed": seed,
                    "prediction": sample_content,
                    "normalized_prediction": preprocess_prediction(sample_content),
                    "usage": response.get("usage", {}),
                }
            )
        if args.response_selection == "majority_vote":
            chosen_normalized = majority_vote(generations)
            content = next(
                item["prediction"]
                for item in generations
                if item["normalized_prediction"] == chosen_normalized
            )
        else:
            selection_prompt = build_selection_prompt(
                args.response_selection,
                example.input,
                generations,
            )
            selection_seed = generation_seed(
                args.seed,
                example,
                args.self_consistency_k,
            )
            selection_response = post_chat_completion(
                base_url=args.base_url,
                model=args.model,
                prompt=selection_prompt,
                max_tokens=args.selection_max_tokens,
                temperature=0.0,
                seed=selection_seed,
                timeout=args.timeout,
                retries=args.retries,
            )
            content = selection_response["choices"][0]["message"].get("content") or ""
            selection = {
                "mode": args.response_selection,
                "seed": selection_seed,
                "prompt": selection_prompt,
                "prediction": content,
                "normalized_prediction": preprocess_prediction(content),
                "usage": selection_response.get("usage", {}),
            }
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
        "prompt_strategy": strategy_name,
        "base_seed": args.seed,
        "self_consistency_k": args.self_consistency_k,
        "response_selection": args.response_selection,
        "generations": generations,
        "selection": selection,
        "usage": combine_usage(
            [item.get("usage", {}) for item in generations]
            + ([selection.get("usage", {})] if selection is not None else [])
        ),
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
    parser.add_argument("--seed", type=int, default=20260709)
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
    parser.add_argument(
        "--prompt-policy",
        type=Path,
        help=(
            "JSON policy with task_strategies mapping benchmark/task keys to prompt strategy "
            "names. Missing tasks use default_strategy or --prompt-strategy."
        ),
    )
    parser.add_argument(
        "--skip-per-task",
        type=int,
        default=0,
        help="Skip the first N indexed examples of every task, for held-out evaluation.",
    )
    parser.add_argument("--self-consistency-k", type=int, default=1)
    parser.add_argument(
        "--response-selection",
        choices=["majority_vote", "self_rank", "key_condition_refine"],
        default="majority_vote",
    )
    parser.add_argument("--selection-max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.prompt_mode == "answer_only":
        args.prompt_strategy = "direct_answer"
    elif args.prompt_mode == "raw":
        args.prompt_strategy = "raw"
    if args.self_consistency_k < 1:
        raise SystemExit("--self-consistency-k must be >= 1")
    if args.response_selection == "self_rank" and args.self_consistency_k < 2:
        raise SystemExit("--response-selection self_rank requires --self-consistency-k >= 2")
    if args.response_selection == "key_condition_refine" and args.self_consistency_k != 1:
        raise SystemExit(
            "--response-selection key_condition_refine requires --self-consistency-k 1"
        )
    if args.skip_per_task < 0:
        raise SystemExit("--skip-per-task must be >= 0")
    args.prompt_policy_payload = None
    args.prompt_policy_map = {}
    args.prompt_policy_default = args.prompt_strategy
    if args.prompt_policy is not None:
        payload = json.loads(args.prompt_policy.read_text())
        task_strategies = payload.get("task_strategies")
        if not isinstance(task_strategies, dict):
            raise SystemExit("--prompt-policy must contain a task_strategies object")
        default_strategy = str(payload.get("default_strategy", args.prompt_strategy))
        unknown = sorted(
            {
                str(strategy)
                for strategy in [default_strategy, *task_strategies.values()]
                if str(strategy) not in PROMPT_STRATEGIES
            }
        )
        if unknown:
            raise SystemExit(f"--prompt-policy contains unknown strategies: {', '.join(unknown)}")
        args.prompt_policy_payload = payload
        args.prompt_policy_map = {
            str(task): str(strategy) for task, strategy in task_strategies.items()
        }
        args.prompt_policy_default = default_strategy
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
    by_benchmark: dict[str, int] = {}
    by_task: dict[str, int] = {}
    by_prompt_strategy: dict[str, int] = {}
    for example in examples:
        by_benchmark[example.benchmark] = by_benchmark.get(example.benchmark, 0) + 1
        key = example_task_key(example)
        by_task[key] = by_task.get(key, 0) + 1
        strategy_name = resolve_prompt_strategy(args, example)
        by_prompt_strategy[strategy_name] = by_prompt_strategy.get(strategy_name, 0) + 1
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "benchmarks": args.benchmarks,
        "limit_per_task": args.limit_per_task,
        "parallel": args.parallel,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "timeout": args.timeout,
        "retries": args.retries,
        **prompt_run_metadata(args),
        "self_consistency_k": args.self_consistency_k,
        "response_selection": args.response_selection,
        "selection_max_tokens": args.selection_max_tokens,
        "skip_per_task": args.skip_per_task,
        "system_messages_sent": 0,
        "request_message_shape": [{"role": "user", "content": "<rendered prompt>"}],
        "example_count": len(examples),
        "examples_by_benchmark": by_benchmark,
        "examples_by_task": by_task,
        "examples_by_prompt_strategy": by_prompt_strategy,
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
    if args.skip_per_task:
        examples = [example for example in examples if example.index >= args.skip_per_task]
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
        **prompt_run_metadata(args),
        "self_consistency_k": args.self_consistency_k,
        "response_selection": args.response_selection,
        "selection_max_tokens": args.selection_max_tokens,
        "system_messages_sent": 0,
        "request_message_shape": [{"role": "user", "content": "<benchmark prompt>"}],
        "limit_per_task": args.limit_per_task,
        "skip_per_task": args.skip_per_task,
        "parallel": args.parallel,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        **summarize(records),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
