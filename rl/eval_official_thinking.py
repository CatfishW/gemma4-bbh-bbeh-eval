#!/usr/bin/env python3
"""Best-public reproduction of Gemma 4's native-thinking BBEH evaluation.

The Gemma 4 report publishes a BBEH score and says that Table 5 models use
thinking, but it does not disclose the BBEH output-token ceiling, seed, or all
prompting details.  This evaluator therefore pins the public model-card
contract and labels its output a reproduction rather than an exact recreation:

* official Gemma chat template with ``enable_thinking=True``;
* BBEH task input plus the benchmark paper's published evaluation suffix;
* temperature 1.0, top-p 0.95, top-k 64;
* final-channel-only scoring with the pinned upstream BBEH scorer;
* explicit model/dataset revisions, truncation, parsing, and resume metadata.

It does not modify or replace the repository's preregistered non-thinking
frozen evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_benchmarks import Example, load_bbeh  # noqa: E402
from rl.official_thinking import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SEED,
    EXPECTED_BBEH_REVISION,
    EXPECTED_MODEL_REVISION,
    EXPECTED_MODEL_SHA256,
    BBEH_EVALUATION_SUFFIX,
    OFFICIAL_TEMPERATURE,
    OFFICIAL_TOP_K,
    OFFICIAL_TOP_P,
    detect_huggingface_revision,
    native_thinking_prompt_token_ids,
    normalize_stop_token_ids,
    official_bbeh_prompt,
    parse_native_thinking_response,
    sha256_file,
    stable_batch_seed,
    trim_generated_token_ids,
    validate_native_thinking_prompt,
)
from rl.protocol import build_protocol_split, prompt_id  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalRequest:
    example: Example
    prompt_id: str
    prompt_token_ids: list[int]


@dataclass(frozen=True)
class GeneratedResponse:
    request: EvalRequest
    completion_token_ids: list[int]
    batch_seed: int
    batch_wall_seconds: float
    stopped: bool
    stop_token_id: int | None
    truncated: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def tracked_worktree_changes(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not inspect dataset worktree: {result.stderr.strip()}")
    return result.stdout.strip()


def acquire_output_lock(output_dir: Path):
    """Hold a non-blocking process lock for one evaluation output directory."""

    import fcntl

    output_dir.mkdir(parents=True, exist_ok=True)
    handle = (output_dir / ".eval.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"another evaluator holds the lock for {output_dir}") from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} acquired_at={utc_now()}\n")
    handle.flush()
    return handle


def load_official_bbeh_scorer(
    datasets_root: Path,
) -> tuple[Callable[[str, str], bool], Path, str]:
    scorer_path = datasets_root / "bbeh" / "bbeh" / "evaluate.py"
    if not scorer_path.is_file():
        raise FileNotFoundError(f"official BBEH scorer not found: {scorer_path}")
    spec = importlib.util.spec_from_file_location("pinned_bbeh_evaluate", scorer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load official BBEH scorer: {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    # The upstream file prints examples at import time. Suppress only those
    # demonstration prints; all evaluation output remains in our artifacts.
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    scorer = getattr(module, "evaluate_correctness", None)
    if not callable(scorer):
        raise RuntimeError("official BBEH scorer has no evaluate_correctness function")
    return scorer, scorer_path, sha256_file(scorer_path)


def select_examples(examples: list[Example], scope: str) -> list[Example]:
    ordered = sorted(examples, key=lambda example: (example.task, example.index))
    if scope == "all":
        return ordered
    if scope == "frozen_test":
        return sorted(
            build_protocol_split(examples).test,
            key=lambda example: (example.task, example.index),
        )
    raise ValueError(f"unknown scope: {scope}")


def plan_batches(
    requests: list[EvalRequest],
    *,
    batch_size: int,
    max_batch_tokens: int,
    max_new_tokens: int,
) -> list[list[EvalRequest]]:
    """Length-bucket requests under count and conservative sequence budgets."""

    ordered = sorted(requests, key=lambda item: (len(item.prompt_token_ids), item.prompt_id))
    batches: list[list[EvalRequest]] = []
    current: list[EvalRequest] = []
    for request in ordered:
        prospective = current + [request]
        longest = max(len(item.prompt_token_ids) for item in prospective)
        padded_budget = (longest + max_new_tokens) * len(prospective)
        if current and (len(prospective) > batch_size or padded_budget > max_batch_tokens):
            batches.append(current)
            current = [request]
        else:
            current = prospective
    if current:
        batches.append(current)
    return batches


def is_cuda_out_of_memory(error: BaseException) -> bool:
    try:
        import torch

        if isinstance(error, torch.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    text = str(error).lower()
    return "cuda" in text and "out of memory" in text


def generate_batch(
    model,
    tokenizer,
    batch: list[EvalRequest],
    *,
    base_seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    device: str,
) -> list[GeneratedResponse]:
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer must define pad_token_id")
    longest = max(len(request.prompt_token_ids) for request in batch)
    input_ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for row, request in enumerate(batch):
        ids = request.prompt_token_ids
        input_ids[row, longest - len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, longest - len(ids) :] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    batch_seed = stable_batch_seed(base_seed, (request.prompt_id for request in batch))
    torch.manual_seed(batch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(batch_seed)
    started = time.monotonic()
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=model.generation_config.eos_token_id,
        use_cache=True,
    )
    wall_seconds = time.monotonic() - started

    stop_ids = normalize_stop_token_ids(model.generation_config.eos_token_id)
    completions = generated[:, longest:].detach().cpu()
    results: list[GeneratedResponse] = []
    for row, request in enumerate(batch):
        trimmed = trim_generated_token_ids(
            completions[row].tolist(),
            stop_token_ids=stop_ids,
            pad_token_id=pad_id,
            max_new_tokens=max_new_tokens,
        )
        results.append(
            GeneratedResponse(
                request=request,
                completion_token_ids=trimmed.token_ids,
                batch_seed=batch_seed,
                batch_wall_seconds=wall_seconds,
                stopped=trimmed.stopped,
                stop_token_id=trimmed.stop_token_id,
                truncated=trimmed.truncated,
            )
        )
    return results


def generate_with_oom_splitting(model, tokenizer, batch: list[EvalRequest], **kwargs):
    import torch

    try:
        return generate_batch(model, tokenizer, batch, **kwargs)
    except Exception as error:
        if not is_cuda_out_of_memory(error):
            raise
    torch.cuda.empty_cache()
    if len(batch) == 1:
        raise RuntimeError(
            f"CUDA OOM on one prompt ({batch[0].prompt_id}); lower --max-new-tokens"
        )
    middle = len(batch) // 2
    LOGGER.warning("generation OOM: splitting batch of %d and retrying", len(batch))
    return generate_with_oom_splitting(model, tokenizer, batch[:middle], **kwargs) + (
        generate_with_oom_splitting(model, tokenizer, batch[middle:], **kwargs)
    )


def load_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
        key = record.get("prompt_id")
        if not isinstance(key, str):
            raise ValueError(f"missing prompt_id at {path}:{line_number}")
        if key in records:
            raise ValueError(f"duplicate prompt_id in {path}: {key}")
        records[key] = record
    return records


def append_records(path: Path, records: list[dict]) -> None:
    with path.open("a") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def metric_summary(records: list[dict]) -> dict:
    correct = sum(bool(record["correct"]) for record in records)
    return {
        "examples": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else 0.0,
        "mean_completion_tokens": (
            statistics.fmean(record["completion_tokens"] for record in records)
            if records
            else 0.0
        ),
        "mean_thinking_tokens": (
            statistics.fmean(record["thinking_tokens"] for record in records)
            if records
            else 0.0
        ),
        "mean_final_answer_tokens": (
            statistics.fmean(record["final_answer_tokens"] for record in records)
            if records
            else 0.0
        ),
        "truncated": sum(bool(record["truncated"]) for record in records),
        "parse_errors": sum(record["parse_error"] is not None for record in records),
        "thinking_present": sum(bool(record["thinking"]) for record in records),
    }


def build_summary(records: list[dict], run_config: dict, *, status: str) -> dict:
    by_task = {
        task: metric_summary([record for record in records if record["task"] == task])
        for task in sorted({record["task"] for record in records})
    }
    frozen = [record for record in records if int(record["index"]) >= 50]
    return {
        "status": status,
        "updated_at": utc_now(),
        "model_label": run_config["model_label"],
        "scope": run_config["protocol"]["scope"],
        "profile": run_config["protocol"]["profile_name"],
        "overall": metric_summary(records),
        "frozen_test": metric_summary(frozen),
        "by_task": by_task,
        "paper_reference": {
            "model": "Gemma 4 E2B IT",
            "benchmark": "Big Bench Extra Hard micro average",
            "accuracy": 0.219,
            "thinking": True,
            "source": "Gemma 4 Technical Report, Table 5",
            "directly_comparable_only_when": "model_label=base and scope=all",
        },
    }


def adapter_fingerprint(adapter: Path | None) -> dict | None:
    if adapter is None:
        return None
    weights = next(
        (candidate for candidate in (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin") if candidate.exists()),
        None,
    )
    if weights is None:
        raise FileNotFoundError(f"adapter weights not found in {adapter}")
    return {
        "path": str(adapter.resolve()),
        "weights_file": weights.name,
        "weights_sha256": sha256_file(weights),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/data/models/gemma-4-E2B-it"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--scope", choices=["all", "frozen_test"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=49152)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=OFFICIAL_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=OFFICIAL_TOP_P)
    parser.add_argument("--top-k", type=int, default=OFFICIAL_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--expected-model-sha256", default=EXPECTED_MODEL_SHA256)
    parser.add_argument("--expected-bbeh-revision", default=EXPECTED_BBEH_REVISION)
    parser.add_argument("--limit", type=int, help="first N task/index rows; smoke testing only")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.max_batch_tokens < 1:
        raise ValueError("batch and token limits must be positive")
    if args.temperature != OFFICIAL_TEMPERATURE or args.top_p != OFFICIAL_TOP_P or args.top_k != OFFICIAL_TOP_K:
        LOGGER.warning("sampling differs from the public Gemma model-card defaults")

    # Keep this handle alive for the process lifetime. The lock is released by
    # the OS on normal exit, exception, or signal termination.
    _output_lock = acquire_output_lock(args.output_dir)

    model_revision = detect_huggingface_revision(args.model_path)
    if model_revision != args.expected_model_revision:
        raise RuntimeError(
            f"model revision mismatch: detected {model_revision!r}, "
            f"expected {args.expected_model_revision!r}"
        )
    weights_path = args.model_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"model weights not found: {weights_path}")
    model_sha256 = sha256_file(weights_path)
    if model_sha256 != args.expected_model_sha256:
        raise RuntimeError(
            f"model weight SHA-256 mismatch: detected {model_sha256!r}, "
            f"expected {args.expected_model_sha256!r}"
        )
    bbeh_repo = args.datasets_root / "bbeh"
    bbeh_revision = git_revision(bbeh_repo)
    if bbeh_revision != args.expected_bbeh_revision:
        raise RuntimeError(
            f"BBEH revision mismatch: detected {bbeh_revision!r}, "
            f"expected {args.expected_bbeh_revision!r}"
        )
    dataset_changes = tracked_worktree_changes(bbeh_repo)
    if dataset_changes:
        raise RuntimeError(f"BBEH tracked files are modified:\n{dataset_changes}")
    scorer, scorer_path, scorer_sha256 = load_official_bbeh_scorer(args.datasets_root)

    import torch
    import transformers

    public_generation_config = transformers.GenerationConfig.from_pretrained(args.model_path)
    public_stop_ids = normalize_stop_token_ids(public_generation_config.eos_token_id)
    if not {1, 50, 106}.issubset(public_stop_ids):
        raise RuntimeError(
            f"unexpected Gemma generation stop tokens: {public_stop_ids}; expected 1, 50, 106"
        )

    examples = select_examples(load_bbeh(args.datasets_root, None), args.scope)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        examples = examples[: args.limit]
    requests: list[EvalRequest] = []

    from rl.modeling import load_policy_model, load_tokenizer

    tokenizer = load_tokenizer(str(args.model_path))
    for example in examples:
        token_ids = native_thinking_prompt_token_ids(
            tokenizer, official_bbeh_prompt(example.input)
        )
        if len(token_ids) > args.max_prompt_tokens:
            raise RuntimeError(
                f"prompt {prompt_id(example)} has {len(token_ids)} tokens, exceeding the "
                f"hard cap {args.max_prompt_tokens}; no rows may be silently skipped"
            )
        requests.append(EvalRequest(example, prompt_id(example), token_ids))
    if not requests:
        raise RuntimeError("selected evaluation scope contains no examples")
    validate_native_thinking_prompt(tokenizer, requests[0].prompt_token_ids)

    protocol = {
        "profile_name": "gemma4_public_native_thinking_bbeh_v3",
        "claim": "best-public reproduction; not an exact recreation of Google's internal evaluation",
        "benchmark": "bbeh",
        "scope": args.scope,
        "dataset_prompt": "task.json input plus the published BBEH Appendix-C suffix",
        "bbeh_evaluation_suffix": BBEH_EVALUATION_SUFFIX,
        "chat_messages": ["user"],
        "manual_system_message": False,
        "enable_thinking": True,
        "score_only_parsed_final_content": True,
        "scorer": "upstream bbeh/bbeh/evaluate.py:evaluate_correctness",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "limit": args.limit,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "generation_stop_token_ids": list(public_stop_ids),
        "published_protocol_sources": [
            "Gemma 4 Technical Report, Table 5",
            "Gemma 4 E2B IT model card, Best Practices",
            "BIG-Bench Extra Hard, Appendix C (Reproducibility)",
        ],
    }
    run_config = {
        "created_at": utc_now(),
        "model_label": args.model_label,
        "model_path": str(args.model_path.resolve()),
        "model_revision": model_revision,
        "model_weights_sha256": model_sha256,
        "model_file_sha256": {
            name: sha256_file(args.model_path / name)
            for name in (
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        },
        "adapter": adapter_fingerprint(args.adapter),
        "bbeh_revision": bbeh_revision,
        "official_scorer_path": str(scorer_path.resolve()),
        "official_scorer_sha256": scorer_sha256,
        "selected_examples": len(requests),
        "protocol": protocol,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "hardware": {
            "device_argument": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "implementation": {
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
            "helpers_sha256": sha256_file(
                Path(__file__).resolve().with_name("official_thinking.py")
            ),
        },
    }

    predictions_path = args.output_dir / "predictions.jsonl"
    config_path = args.output_dir / "run_config.json"
    existing_records = load_records(predictions_path)
    if (existing_records or config_path.exists()) and not args.resume:
        raise FileExistsError(
            f"output already contains a run: {args.output_dir}; pass --resume to continue it"
        )
    if args.resume:
        if not config_path.exists():
            raise FileNotFoundError(f"cannot resume without {config_path}")
        existing_config = json.loads(config_path.read_text())
        for key in (
            "model_label",
            "model_path",
            "model_revision",
            "model_weights_sha256",
            "model_file_sha256",
            "adapter",
            "bbeh_revision",
            "official_scorer_sha256",
            "selected_examples",
            "protocol",
            "software",
            "implementation",
        ):
            if existing_config.get(key) != run_config.get(key):
                raise RuntimeError(f"resume configuration mismatch for {key}")
        run_config = existing_config
    else:
        atomic_write_json(config_path, run_config)

    selected_ids = {request.prompt_id for request in requests}
    unexpected = set(existing_records) - selected_ids
    if unexpected:
        raise RuntimeError(f"existing predictions are outside this scope: {sorted(unexpected)[:3]}")
    remaining = [request for request in requests if request.prompt_id not in existing_records]
    LOGGER.info(
        "native-thinking BBEH: %d selected, %d already complete, %d remaining",
        len(requests),
        len(existing_records),
        len(remaining),
    )
    if not remaining:
        summary = build_summary(list(existing_records.values()), run_config, status="complete")
        atomic_write_json(args.output_dir / "summary.json", summary)
        LOGGER.info("evaluation already complete: %.2f%%", 100 * summary["overall"]["accuracy"])
        return 0

    model = load_policy_model(
        str(args.model_path), attn_implementation="sdpa", device=args.device
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        LOGGER.info("loaded adapter %s", args.adapter)
    model.eval()

    batches = plan_batches(
        remaining,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    started = time.monotonic()
    for batch_number, batch in enumerate(batches, start=1):
        generated = generate_with_oom_splitting(
            model,
            tokenizer,
            batch,
            base_seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
        new_records: list[dict] = []
        for result in generated:
            parsed = parse_native_thinking_response(tokenizer, result.completion_token_ids)
            prediction = parsed.prediction
            correct = bool(scorer(prediction, result.request.example.target))
            new_records.append(
                {
                    "benchmark": "bbeh",
                    "task": result.request.example.task,
                    "index": result.request.example.index,
                    "prompt_id": result.request.prompt_id,
                    "target": result.request.example.target,
                    "prediction": prediction,
                    "thinking": parsed.thinking,
                    "raw_response": parsed.raw_response,
                    "correct": correct,
                    "scorer": "official_bbeh",
                    "prompt_tokens": len(result.request.prompt_token_ids),
                    "completion_tokens": len(result.completion_token_ids),
                    "thinking_tokens": len(
                        tokenizer.encode(parsed.thinking, add_special_tokens=False)
                    ),
                    "final_answer_tokens": len(
                        tokenizer.encode(prediction, add_special_tokens=False)
                    ),
                    "stopped": result.stopped,
                    "stop_token_id": result.stop_token_id,
                    "truncated": result.truncated,
                    "parse_error": parsed.parse_error,
                    "batch_seed": result.batch_seed,
                    "batch_wall_seconds": result.batch_wall_seconds,
                }
            )
        append_records(predictions_path, new_records)
        existing_records.update({record["prompt_id"]: record for record in new_records})
        progress = build_summary(
            list(existing_records.values()),
            run_config,
            status="running" if len(existing_records) < len(requests) else "complete",
        )
        progress["selected_examples"] = len(requests)
        progress["completed_examples"] = len(existing_records)
        progress["remaining_examples"] = len(requests) - len(existing_records)
        progress["elapsed_seconds_this_process"] = time.monotonic() - started
        atomic_write_json(args.output_dir / "progress.json", progress)
        LOGGER.info(
            "batch %d/%d; complete %d/%d; accuracy %.2f%%; truncated %d; parse errors %d",
            batch_number,
            len(batches),
            len(existing_records),
            len(requests),
            100 * progress["overall"]["accuracy"],
            progress["overall"]["truncated"],
            progress["overall"]["parse_errors"],
        )

    records = list(existing_records.values())
    if len(records) != len(requests):
        raise RuntimeError(f"incomplete evaluation: {len(records)}/{len(requests)}")
    summary = build_summary(records, run_config, status="complete")
    summary["finished_at"] = utc_now()
    summary["elapsed_seconds_this_process"] = time.monotonic() - started
    atomic_write_json(args.output_dir / "summary.json", summary)
    LOGGER.info(
        "complete: %d/%d = %.2f%%; mean %.1f completion tokens",
        summary["overall"]["correct"],
        summary["overall"]["examples"],
        100 * summary["overall"]["accuracy"],
        summary["overall"]["mean_completion_tokens"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
