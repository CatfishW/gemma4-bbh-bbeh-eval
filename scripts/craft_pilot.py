#!/usr/bin/env python3
"""Reproducible, resource-bounded CRAFT pilot; never loads indices >= 50.

Two independent workers use CUDA_VISIBLE_DEVICES to select one GPU each.
Task selection uses calibration lengths and names, never model outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_craft.core import Example
from rl_craft.data import file_hash, load_rows, prepare, write_json


def prepare_pilot(args):
    from eval_benchmarks import load_bbh, load_bbeh, load_unpuzzles_simple_reasoning
    from rl.modeling import load_tokenizer
    from rl_craft.hf_backend import HFBackend
    from rl_craft.core import Config

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    original = prepare(None, args.datasets_root, out / "all-calibration.jsonl")
    tokenizer = load_tokenizer(str(args.model_path))
    raw = [e for loader in (load_bbh, load_bbeh, load_unpuzzles_simple_reasoning)
           for e in loader(args.datasets_root, 50)]
    rows = [Example(f"{e.benchmark}/{e.task}/{e.index}", f"{e.benchmark}/{e.task}",
                    e.input, e.target, e.index) for e in raw]
    lengths = {}
    for e in rows:
        if e.index < 25:
            lengths[e.task] = max(lengths.get(e.task, 0), len(tokenizer.encode(e.question)))
    eligible = sorted(t for t, n in lengths.items() if n <= args.max_question_tokens)
    original_held = set(original["heldout_tasks"])
    seen, held = [], []
    for benchmark in ("bbh", "bbeh", "usr"):
        group = [t for t in eligible if t.startswith(benchmark + "/")]
        seen.extend([t for t in group if t not in original_held][:4])
        held.extend([t for t in group if t in original_held][:2])
    train = [e for e in rows if e.task in seen and e.index < 25]
    validation = [e for e in rows if e.task in seen + held and 25 <= e.index < 29]
    cfg = json.loads((ROOT / "experiments/craft/e2b.json").read_text())
    cfg.update(iterations=12, roots_per_step=4, max_context=2048, max_sampled_tokens=20000)
    # Render every selected stage before reserving GPUs; actual generated notes
    # remain subject to the backend's strict context guard during execution.
    stub = HFBackend.__new__(HFBackend)
    stub.tokenizer, stub.config = tokenizer, Config(**cfg)
    for e in train + validation:
        for stage in ("notes", "gate", "continue", "answer"):
            ctx = stub.prompt(stage, e.question)
            if len(ctx) + cfg["prefix_tokens"] + cfg["continue_tokens"] + cfg["answer_tokens"] + 128 > cfg["max_context"]:
                raise ValueError(f"pilot context preflight failed: {e.key}/{stage}")
    def write_rows(name, examples):
        with (out / name).open("x") as stream:
            for e in examples:
                stream.write(json.dumps(asdict(e)) + "\n")
    write_rows("source.jsonl", train)
    manifest = prepare(out / "source.jsonl", None, out / "train.jsonl", 0)
    # Declare the original family holdout in addition to this pilot's row subset.
    manifest.update(heldout_tasks=original["heldout_tasks"],
                    dataset_revisions=original["dataset_revisions"],
                    selection="craft_pilot.py: first four eligible non-heldout tasks per benchmark")
    (out / "train.jsonl.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_rows("validation.jsonl", validation)
    write_json(out / "full.json", cfg)
    write_json(out / "no_crossfit.json", {**cfg, "suffix_baseline": "history"})
    protocol = {
        "study_id": "craft-short-context-pilot-20260905",
        "claim": "exploratory one-seed pilot; not full benchmark or compute-matched evidence",
        "selection_rule": "calibration question <= threshold tokens; sorted first 4 seen and first 2 heldout tasks per benchmark",
        "max_question_tokens": args.max_question_tokens,
        "seen_tasks": seen, "heldout_tasks": held,
        "excluded_tasks": sorted(set(lengths) - set(seen + held)),
        "calibration_max_question_tokens": lengths,
        "training_rows": len(train), "validation_rows": len(validation),
        "validation_indices": [25, 26, 27, 28],
        "train_sha256": file_hash(out / "train.jsonl"),
        "validation_sha256": file_hash(out / "validation.jsonl"),
        "dataset_revisions": original["dataset_revisions"],
        "fixed_accuracy_floor": cfg["target_accuracy"],
        "floor_note": "explicit fixed floor, not calibrated base accuracy",
        "model_path": str(args.model_path.resolve()),
        "evaluations": {"full": ["sample", "always-continue"],
                        "no_crossfit": ["sample", "always-continue"],
                        "base": ["sample", "always-continue"]},
    }
    write_json(out / "study.json", protocol)
    print(json.dumps(protocol, indent=2))


def worker(args):
    out = args.output.resolve()
    study = json.loads((out / "study.json").read_text())
    def run(*argv):
        subprocess.run([sys.executable, "-u", "-m", "rl_craft", *map(str, argv)],
                       cwd=ROOT, check=True)
    arm = args.arm
    run("train", "--data", out / "train.jsonl", "--model-path", study["model_path"],
        "--config", out / f"{arm}.json", "--device", "cuda:0",
        "--study-id", study["study_id"] + "/" + arm, "--output", out / arm)
    latest = (out / arm / "latest").read_text().strip()
    modes = [(arm, latest, "sample"), (arm, latest, "always-continue")]
    # Split the untrained controls across the two workers.
    modes.append(("base", "checkpoint-00000", "sample" if arm == "full" else "always-continue"))
    for label, checkpoint, mode in modes:
        run("evaluate", "--checkpoint", out / arm / checkpoint,
            "--model-path", study["model_path"], "--data", out / "validation.jsonl",
            "--device", "cuda:0", "--split", "validation",
            "--study-id", study["study_id"], "--gate", mode,
            "--output", out / f"eval-{label}-{mode}")
    write_json(out / f"{arm}-complete.json", {"status": "complete", "checkpoint": latest})


def report(args):
    from scripts.compare_official_thinking_evals import exact_mcnemar_p, stratified_bootstrap_delta, percentile
    out = args.output
    study = json.loads((out / "study.json").read_text())
    if file_hash(out / "validation.jsonl") != study["validation_sha256"]:
        raise ValueError("validation file differs from declared study")
    examples = {e.key: e for e in load_rows(out / "validation.jsonl")}
    expected = set(examples)
    runs, summaries = {}, {}
    for label in ("base", "full", "no_crossfit"):
        for mode in ("sample", "always-continue"):
            name = f"{label}-{mode}"
            folder = out / f"eval-{name}"
            summary = json.loads((folder / "summary.json").read_text())
            records = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
            indexed = {r["key"]: r for r in records}
            if len(records) != len(expected) or set(indexed) != expected or summary["n"] != len(expected):
                raise ValueError(f"incomplete/mismatched run: {name}")
            if any(r["task"] != examples[k].task or r["index"] != examples[k].index
                   or type(r["correct"]) is not bool for k, r in indexed.items()):
                raise ValueError(f"prediction identity/outcome mismatch: {name}")
            runs[name] = indexed
            for split, tasks in (("seen", study["seen_tasks"]), ("heldout", study["heldout_tasks"])):
                group = [r for r in records if r["task"] in tasks]
                summary[split] = {"n": len(group), "correct": sum(r["correct"] for r in group)}
            summaries[name] = summary
    comparisons = {}
    keys = sorted(expected)
    for a, b in (("base-sample", "full-sample"), ("no_crossfit-sample", "full-sample"),
                 ("full-always-continue", "full-sample")):
        baseline, challenger = runs[a], runs[b]
        wins = sum(not baseline[k]["correct"] and challenger[k]["correct"] for k in keys)
        losses = sum(baseline[k]["correct"] and not challenger[k]["correct"] for k in keys)
        samples = stratified_bootstrap_delta(baseline, challenger, keys, replicates=10000, seed=20260905)
        comparisons[f"{b} vs {a}"] = {"wins": wins, "losses": losses,
            "accuracy_delta": (wins-losses)/len(keys), "mcnemar_p": exact_mcnemar_p(wins, losses),
            "task_stratified_bootstrap_95": [percentile(samples, .025), percentile(samples, .975)]}
    training = {}
    for arm in ("full", "no_crossfit"):
        folder = out / arm / (out / arm / "latest").read_text().strip()
        metrics = json.loads((folder / "metrics.json").read_text())
        training[arm] = {"iterations": len(metrics),
            "sampled_tokens": sum(m["sampled_tokens"] for m in metrics),
            "prefill_tokens": sum(m["generation_prefill_tokens"] for m in metrics),
            "step_wall_seconds": sum(m["wall_seconds"] for m in metrics),
            "peak_allocated_bytes": max(m["peak_allocated_bytes"] for m in metrics),
            "final_mean_stop_probability": metrics[-1]["mean_stop_probability"],
            "checkpoint": str(folder)}
    result = {"study": study, "evaluation": summaries, "paired_comparisons": comparisons, "training": training}
    write_json(out / "comparison.json", result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--datasets-root", type=Path, required=True)
    q.add_argument("--model-path", type=Path, required=True)
    q.add_argument("--max-question-tokens", type=int, default=600)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("worker")
    q.add_argument("--arm", choices=["full", "no_crossfit"], required=True)
    q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("report")
    q.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    {"prepare": prepare_pilot, "worker": worker, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
