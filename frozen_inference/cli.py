"""Auditable opt-in runner and calibration commands, separate from legacy protocols."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from .backend import ChatClient, merge_usage
from .executor import Rejected, digest_text
from .memory import SkillLibrary
from .pipeline import Config, Pipeline
from .policy import RepairPolicy, fit_policy


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise Rejected(f"invalid JSON at {path.name}:{i}") from exc
            if not isinstance(row, dict):
                raise Rejected("JSONL rows must be objects")
            rows.append(row)
    return rows


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def source_hashes() -> dict:
    package = Path(__file__).resolve().parent
    paths = sorted(package.glob("*.py"))
    legacy = package.parent / "eval_benchmarks.py"
    if legacy.exists():
        paths.append(legacy)
    return {str(path.relative_to(package.parent)): file_hash(path) for path in paths}


def select_examples(rows: list[dict], split: str, tasks: list[str] | None = None) -> list[dict]:
    selected, seen = [], set()
    for row in rows:
        if not isinstance(row.get("input"), str) or not row["input"].strip():
            raise Rejected("nonempty input required")
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("benchmark", "task")):
            raise Rejected("benchmark and task strings required")
        index = row.get("index")
        if type(index) is not int or index < 0:
            raise Rejected("nonnegative integer index required")
        key = (row["benchmark"], row["task"], index)
        if key in seen:
            raise Rejected("duplicate input identity")
        seen.add(key)
        if tasks and f"{key[0]}/{key[1]}" not in tasks:
            continue
        include = (split == "external" or (split == "calibration" and index < 25)
                   or (split == "validation" and 25 <= index < 50) or (split == "test" and index >= 50))
        if include:
            selected.append(row)
    if not selected:
        raise Rejected("no selected examples")
    return selected


def summary(rows: list[dict]) -> dict:
    latencies = sorted(row["elapsed_seconds"] for row in rows)
    scored = [row for row in rows if type(row.get("correct")) is bool]
    by_task = {}
    for row in scored:
        by_task.setdefault(f"{row['benchmark']}/{row['task']}", []).append(row["correct"])
    counts = {}
    for row in rows:
        counts[row["route"]] = counts.get(row["route"], 0) + 1
    completion_tokens = []
    for row in rows:
        usage = row.get("usage", {})
        value = usage.get("completion_tokens", usage.get("output_tokens"))
        if row["model_calls"] == 0:
            value = 0
        completion_tokens.append(value)
    known = all(type(n) is int for n in completion_tokens) and all(row["usage_complete"] for row in rows)
    return {"examples": len(rows), "scored_examples": len(scored),
            "micro_exact_match": sum(r["correct"] for r in scored) / len(scored) if scored else None,
            "macro_task_exact_match": statistics.mean(statistics.mean(v) for v in by_task.values()) if by_task else None,
            "per_task": {k: {"correct": sum(v), "total": len(v), "accuracy": statistics.mean(v)} for k, v in by_task.items()},
            "mean_elapsed_seconds": statistics.mean(latencies), "median_elapsed_seconds": statistics.median(latencies),
            "p95_elapsed_seconds": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "p95_method": "nearest-rank", "routes": counts,
            "errors": sum(row["error"] is not None for row in rows),
            "model_calls": sum(row["model_calls"] for row in rows),
            "request_attempts": sum(row["request_attempts"] for row in rows),
            "zero_call_fraction": sum(row["model_calls"] == 0 for row in rows) / len(rows),
            "zero_call_answer_fraction": sum(row["model_calls"] == 0 and bool(row["prediction"]) for row in rows) / len(rows),
            "usage": merge_usage(rows), "usage_complete": known,
            "mean_completion_tokens": statistics.mean(completion_tokens) if known else None}


def run(args: argparse.Namespace) -> dict:
    if args.split == "test" and not args.allow_test:
        raise Rejected("test evaluation requires --allow-test and a separately declared --study-id")
    if args.parallel < 1 or args.parallel > 64 or args.limit is not None and args.limit < 1:
        raise Rejected("invalid parallelism/limit")
    legacy = None
    if args.datasets_root is not None:
        legacy = importlib.import_module("eval_benchmarks")
        loaders = {"bbh": legacy.load_bbh, "bbeh": legacy.load_bbeh, "usr": legacy.load_unpuzzles_simple_reasoning}
        names = args.benchmarks.split(",")
        if any(name not in loaders for name in names) or len(set(names)) != len(names):
            raise Rejected("unknown/duplicate benchmark")
        rows = [asdict(ex) for name in names for ex in loaders[name](args.datasets_root, None)]
        source = {"datasets_root": str(args.datasets_root), "benchmark_names": names}
    else:
        rows = read_jsonl(args.examples)
        source = {"examples_path": str(args.examples), "examples_sha256": file_hash(args.examples)}
    selected = select_examples(rows, args.split, args.task)
    if args.limit is not None:
        selected = selected[:args.limit]
    scoring = not args.unscored
    if scoring:
        if any(not isinstance(row.get("target"), str) or not row["target"] for row in selected):
            raise Rejected("scored runs require nonempty targets; use --unscored for deployment/smoke data")
        legacy = legacy or importlib.import_module("eval_benchmarks")
    skills = SkillLibrary.load(args.skills) if args.skills else None
    policy = RepairPolicy.load(args.policy) if args.policy else None
    config = Config(mode=args.mode, enable_compile=args.compile, max_calls=args.max_calls,
                    max_repairs=args.max_repairs, compile_tokens=args.compile_tokens,
                    answer_tokens=args.answer_tokens, repair_tokens=args.repair_tokens,
                    total_seconds=args.total_seconds, seed=args.seed)
    if policy is not None:
        binding = policy.payload.get("applicability")
        actual = {"model": args.model, "deployment_id": args.deployment_id,
                  **{k: getattr(config, k) for k in ("mode", "max_calls", "max_repairs", "compile_tokens", "answer_tokens", "repair_tokens")}}
        if binding != actual:
            raise Rejected("policy deployment/budget mismatch (or missing applicability binding)")
    if not args.offline and not args.base_url:
        raise Rejected("provide --base-url or explicitly select --offline")
    client = None if args.offline else ChatClient(args.base_url, args.model,
                                                os.environ.get(args.api_key_env) if args.api_key_env else None,
                                                args.timeout, args.retries)
    pipeline = Pipeline(client, config, skills=skills, policy=policy)
    # This complete configuration is persisted before any inference. No resume or
    # overwrite: previously scored data must never be silently mixed with a new arm.
    protocol = {"schema": 1, "study_id": args.study_id, "track": "exploratory-frozen-system",
                "created_at": datetime.now(timezone.utc).isoformat(), "split": args.split,
                "config": asdict(config), "source": source, "code_sha256": source_hashes(),
                "model": args.model, "base_url": args.base_url if not args.offline else None,
                "deployment_id": args.deployment_id, "offline": args.offline,
                "parallel": args.parallel, "limit": args.limit, "scoring": scoring,
                "scorer": "eval_benchmarks.evaluate_correctness" if scoring else None,
                "timeout": args.timeout, "retries": args.retries,
                "api_key_env_name": args.api_key_env, "system_messages_sent": 0,
                "rendered_template_audited": False,
                "selected_input_digest": digest_text(json.dumps(selected, sort_keys=True)),
                "example_count": len(selected), "policy_sha256": policy.fingerprint if policy else None,
                "skills_sha256": skills.fingerprint if skills else None}
    encoded = json.dumps(protocol, indent=2, sort_keys=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "run_config.json").write_text(encoded + "\n", encoding="utf-8")
    (args.output_dir / "run_config.sha256").write_text(digest_text(encoded) + "\n", encoding="utf-8")

    def evaluate(row: dict) -> dict:
        # Gold labels are deliberately outside Pipeline.predict and all model prompts.
        result = pipeline.predict(row["input"])
        result.update(benchmark=row["benchmark"], task=row["task"], index=row["index"], split=args.split,
                      case_id=f"{row['benchmark']}/{row['task']}/{row['index']}",
                      target=row.get("target") if scoring else None,
                      correct=bool(legacy.evaluate_correctness(result["prediction"], row["target"])) if scoring else None)
        return result

    started, outputs = time.perf_counter(), []
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(evaluate, row) for row in selected]
            for future in as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                outputs.append(result)
    report = summary(outputs)
    report.update(wall_seconds=time.perf_counter() - started, protocol_sha256=digest_text(encoded))
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    source = runner.add_mutually_exclusive_group(required=True)
    source.add_argument("--datasets-root", type=Path)
    source.add_argument("--examples", type=Path)
    runner.add_argument("--output-dir", type=Path, required=True)
    runner.add_argument("--study-id", required=True)
    runner.add_argument("--split", choices=["calibration", "validation", "test", "external"], default="calibration")
    runner.add_argument("--allow-test", action="store_true")
    runner.add_argument("--benchmarks", default="bbh,bbeh,usr")
    runner.add_argument("--task", action="append", help="exact benchmark/task key; repeatable")
    runner.add_argument("--limit", type=int)
    runner.add_argument("--unscored", action="store_true")
    runner.add_argument("--offline", action="store_true")
    runner.add_argument("--base-url")
    runner.add_argument("--model", default="SubTokenLLM-E2B")
    runner.add_argument("--deployment-id", default="unverified-serving-snapshot")
    runner.add_argument("--api-key-env", default="OPENAI_API_KEY")
    runner.add_argument("--mode", choices=["direct", "stable", "full"], default="stable")
    runner.add_argument("--compile", action="store_true", help="opt in to conditional neural compilation")
    runner.add_argument("--skills", type=Path)
    runner.add_argument("--policy", type=Path)
    runner.add_argument("--parallel", type=int, default=1)
    runner.add_argument("--max-calls", type=int, default=4)
    runner.add_argument("--max-repairs", type=int, default=2)
    runner.add_argument("--compile-tokens", type=int, default=512)
    runner.add_argument("--answer-tokens", type=int, default=64)
    runner.add_argument("--repair-tokens", type=int, default=32)
    runner.add_argument("--total-seconds", type=float, default=120)
    runner.add_argument("--timeout", type=float, default=60)
    runner.add_argument("--retries", type=int, default=0)
    runner.add_argument("--seed", type=int, default=20260904)
    fit = sub.add_parser("fit-policy")
    fit.add_argument("--baseline", type=Path, required=True, help="calibration run directory")
    fit.add_argument("--challenger", type=Path, required=True, help="calibration run directory")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--min-examples", type=int, default=20)
    fit.add_argument("--penalty-per-second", type=float, default=0.01)
    validate = sub.add_parser("validate-skills")
    validate.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run(args)
        elif args.command == "validate-skills":
            library = SkillLibrary.load(args.path)
            result = {"skills": [skill.name for skill in library.skills], "sha256": library.fingerprint,
                      "status": "development-tests-passed-not-a-semantic-proof"}
        else:
            configs = [json.loads((p / "run_config.json").read_text()) for p in (args.baseline, args.challenger)]
            if any(c.get("split") != "calibration" or c.get("scoring") is not True for c in configs):
                raise Rejected("both runs must be scored calibration runs")
            for field in ("model", "base_url", "deployment_id", "selected_input_digest", "scorer"):
                if configs[0].get(field) != configs[1].get(field):
                    raise Rejected(f"calibration protocol mismatch: {field}")
            result = fit_policy(read_jsonl(args.baseline / "predictions.jsonl"),
                                read_jsonl(args.challenger / "predictions.jsonl"),
                                min_examples=args.min_examples, penalty_per_second=args.penalty_per_second)
            candidate = configs[1]
            result["applicability"] = {"model": candidate["model"], "deployment_id": candidate["deployment_id"],
                                       **{k: candidate["config"][k] for k in
                                          ("mode", "max_calls", "max_repairs", "compile_tokens", "answer_tokens", "repair_tokens")}}
            result["run_config_digests"] = [file_hash(p / "run_config.json") for p in (args.baseline, args.challenger)]
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
        print(json.dumps(result, indent=2))
        return 0
    except (Rejected, ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
