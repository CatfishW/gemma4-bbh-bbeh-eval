#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from math import comb
from pathlib import Path
from statistics import mean
from typing import Iterable


ExampleKey = tuple[str, str, int]
TaskKey = tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze E2B finalists from calibration and validation predictions."
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--e4b-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="direct_answer")
    parser.add_argument("--calibration-end", type=int, default=25)
    parser.add_argument("--validation-end", type=int, default=50)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    names = [str(row["name"]) for row in rows]
    if len(names) != len(set(names)):
        raise SystemExit("arm manifest contains duplicate names")
    return rows


def prediction_path(root: Path, arm: str) -> Path:
    for candidate in (
        root / arm / "predictions.jsonl",
        root / arm / "predictions.jsonl.gz",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit(f"missing predictions for arm {arm}")


def read_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def example_key(row: dict) -> ExampleKey:
    return str(row["benchmark"]), str(row["task"]), int(row["index"])


def task_key(key: ExampleKey) -> TaskKey:
    return key[0], key[1]


def task_name(task: TaskKey) -> str:
    return f"{task[0]}/{task[1]}"


def completion_tokens(row: dict) -> int:
    usage = row.get("usage") or {}
    return int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def beta_superiority_probability(wins: int, losses: int) -> float:
    """P(p > 0.5) for p ~ Beta(wins + 1, losses + 1)."""
    alpha = wins + 1
    beta = losses + 1
    n = alpha + beta - 1
    return sum(comb(n, j) for j in range(alpha)) / (2**n)


def arm_metrics(rows: dict[ExampleKey, dict], keys: list[ExampleKey]) -> dict:
    correct = sum(bool(rows[key]["correct"]) for key in keys)
    token_values = [completion_tokens(rows[key]) for key in keys]
    elapsed = [float(rows[key].get("elapsed_seconds") or 0.0) for key in keys]
    errors = sum(bool(rows[key].get("error")) for key in keys)
    return {
        "correct": correct,
        "total": len(keys),
        "accuracy": correct / len(keys) if keys else 0.0,
        "mean_completion_tokens": mean(token_values) if token_values else 0.0,
        "mean_elapsed_seconds": mean(elapsed) if elapsed else 0.0,
        "errors": errors,
    }


def fit_task_policy(
    rows_by_arm: dict[str, dict[ExampleKey, dict]],
    router_arms: list[str],
    baseline: str,
    calibration_keys: list[ExampleKey],
    minimum_net_wins: int,
    posterior_threshold: float,
    manifest_index: dict[str, int],
) -> tuple[dict[TaskKey, str], list[dict]]:
    by_task: dict[TaskKey, list[ExampleKey]] = defaultdict(list)
    for key in calibration_keys:
        by_task[task_key(key)].append(key)

    policy: dict[TaskKey, str] = {}
    details = []
    for task in sorted(by_task):
        keys = by_task[task]
        candidates = []
        for arm in router_arms:
            if arm == baseline:
                continue
            wins = sum(
                not rows_by_arm[baseline][key]["correct"]
                and rows_by_arm[arm][key]["correct"]
                for key in keys
            )
            losses = sum(
                rows_by_arm[baseline][key]["correct"]
                and not rows_by_arm[arm][key]["correct"]
                for key in keys
            )
            posterior = beta_superiority_probability(wins, losses)
            net_wins = wins - losses
            metrics = arm_metrics(rows_by_arm[arm], keys)
            candidates.append(
                {
                    "arm": arm,
                    "paired_wins": wins,
                    "paired_losses": losses,
                    "net_wins": net_wins,
                    "posterior_probability_superior": posterior,
                    "calibration_correct": metrics["correct"],
                    "mean_completion_tokens": metrics["mean_completion_tokens"],
                    "eligible": (
                        net_wins >= minimum_net_wins and posterior >= posterior_threshold
                    ),
                }
            )
        eligible = [row for row in candidates if row["eligible"]]
        if eligible:
            selected = max(
                eligible,
                key=lambda row: (
                    row["posterior_probability_superior"],
                    row["net_wins"],
                    row["calibration_correct"],
                    -row["mean_completion_tokens"],
                    -manifest_index[row["arm"]],
                ),
            )["arm"]
        else:
            selected = baseline
        policy[task] = selected
        details.append(
            {
                "task": task_name(task),
                "selected_arm": selected,
                "calibration_examples": len(keys),
                "candidate_statistics": candidates,
            }
        )
    return policy, details


def policy_metrics(
    policy: dict[TaskKey, str],
    rows_by_arm: dict[str, dict[ExampleKey, dict]],
    keys: list[ExampleKey],
) -> dict:
    selected_rows = {key: rows_by_arm[policy[task_key(key)]][key] for key in keys}
    return arm_metrics(selected_rows, keys)


def deduplicate_finalists(rows: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        identity = (row["kind"], row.get("arm"), row.get("policy_path"))
        if identity in seen:
            continue
        seen.add(identity)
        result.append(row)
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    manifest_by_name = {str(row["name"]): row for row in manifest}
    manifest_index = {str(row["name"]): index for index, row in enumerate(manifest)}
    if args.baseline not in manifest_by_name:
        raise SystemExit("baseline is absent from manifest")

    rows_by_arm: dict[str, dict[ExampleKey, dict]] = {}
    source_files = {}
    source_sha256 = {}
    for arm in manifest_by_name:
        path = prediction_path(args.runs_root, arm)
        rows = {example_key(row): row for row in read_jsonl(path)}
        if len(rows) == 0:
            raise SystemExit(f"empty predictions for arm {arm}")
        rows_by_arm[arm] = rows
        source_files[arm] = str(path)
        source_sha256[arm] = file_sha256(path)

    all_keys = set(rows_by_arm[args.baseline])
    for arm, rows in rows_by_arm.items():
        if set(rows) != all_keys:
            raise SystemExit(
                f"coverage mismatch for {arm}: missing={len(all_keys - set(rows))} "
                f"extra={len(set(rows) - all_keys)}"
            )
    if any(key[2] >= args.validation_end for key in all_keys):
        raise SystemExit("screening predictions contain test-split rows")

    calibration_keys = sorted(key for key in all_keys if key[2] < args.calibration_end)
    validation_keys = sorted(
        key for key in all_keys if args.calibration_end <= key[2] < args.validation_end
    )
    if not calibration_keys or not validation_keys:
        raise SystemExit("calibration or validation split is empty")

    rankings = []
    for arm in manifest_by_name:
        rankings.append(
            {
                "arm": arm,
                "calibration": arm_metrics(rows_by_arm[arm], calibration_keys),
                "validation": arm_metrics(rows_by_arm[arm], validation_keys),
                "manifest_index": manifest_index[arm],
            }
        )
    rankings.sort(
        key=lambda row: (
            -row["validation"]["correct"],
            row["validation"]["mean_completion_tokens"],
            row["manifest_index"],
        )
    )
    universal_winner = rankings[0]["arm"]

    router_arms = [
        str(row["name"]) for row in manifest if bool(row.get("eligible_for_router"))
    ]
    grid_results = []
    grid_index = 0
    for minimum_net_wins in (1, 2, 3, 4):
        for posterior_threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            policy, details = fit_task_policy(
                rows_by_arm,
                router_arms,
                args.baseline,
                calibration_keys,
                minimum_net_wins,
                posterior_threshold,
                manifest_index,
            )
            metrics = policy_metrics(policy, rows_by_arm, validation_keys)
            grid_results.append(
                {
                    "grid_index": grid_index,
                    "minimum_net_wins": minimum_net_wins,
                    "posterior_probability_threshold": posterior_threshold,
                    "routed_tasks": sum(arm != args.baseline for arm in policy.values()),
                    "selected_arm_counts": dict(Counter(policy.values())),
                    "validation": metrics,
                    "policy": policy,
                    "task_details": details,
                }
            )
            grid_index += 1
    grid_results.sort(
        key=lambda row: (
            -row["validation"]["correct"],
            row["validation"]["mean_completion_tokens"],
            row["routed_tasks"],
            -row["posterior_probability_threshold"],
            -row["minimum_net_wins"],
            row["grid_index"],
        )
    )
    best_router = grid_results[0]
    policy_payload = {
        "name": "e2b_conservative_bayesian_reward_router",
        "description": (
            "Offline contextual-bandit prompt policy fitted on E2B calibration rows with a "
            "validation-selected conservative Beta-Bernoulli gate."
        ),
        "default_strategy": args.baseline,
        "calibration_index_range": [0, args.calibration_end],
        "validation_index_range": [args.calibration_end, args.validation_end],
        "minimum_net_wins": best_router["minimum_net_wins"],
        "posterior_probability_threshold": best_router[
            "posterior_probability_threshold"
        ],
        "selection_rule": (
            "For each task, choose the eligible arm with highest paired Beta posterior "
            "probability of superiority; direct_answer is the fallback."
        ),
        "strategy_order": router_arms,
        "task_strategies": {
            task_name(task): arm for task, arm in sorted(best_router["policy"].items())
        },
        "validation_metrics": best_router["validation"],
        "system_messages_sent": 0,
    }
    policy_path = args.output_dir / "cbrr_policy.json"
    policy_path.write_text(json.dumps(policy_payload, indent=2) + "\n")

    finalists = deduplicate_finalists(
        [
            {"name": "direct_answer", "kind": "arm", "arm": "direct_answer"},
            {"name": "private_verify", "kind": "arm", "arm": "private_verify"},
            {
                "name": "condition_reconstruction",
                "kind": "arm",
                "arm": "condition_reconstruction",
            },
            {
                "name": f"validation_winner__{universal_winner}",
                "kind": "arm",
                "arm": universal_winner,
            },
            {
                "name": "cbrr_policy",
                "kind": "policy",
                "policy_path": str(policy_path),
                "max_tokens": 256,
            },
            {
                "name": "e4b_policy_transfer",
                "kind": "policy",
                "policy_path": str(args.e4b_policy.resolve()),
                "max_tokens": 128,
            },
        ]
    )
    selection = {
        "selection_frozen": True,
        "test_rows_read": 0,
        "baseline": args.baseline,
        "calibration_examples": len(calibration_keys),
        "validation_examples": len(validation_keys),
        "screening_source_files": source_files,
        "screening_source_sha256": source_sha256,
        "universal_validation_winner": universal_winner,
        "universal_validation_metrics": rankings[0]["validation"],
        "router_hyperparameters": {
            "minimum_net_wins": best_router["minimum_net_wins"],
            "posterior_probability_threshold": best_router[
                "posterior_probability_threshold"
            ],
            "routed_tasks": best_router["routed_tasks"],
        },
        "router_validation_metrics": best_router["validation"],
        "router_selected_arm_counts": best_router["selected_arm_counts"],
        "primary_comparison": "cbrr_policy_vs_direct_answer",
        "finalists": finalists,
        "arm_rankings": rankings,
        "router_grid_rankings": [
            {
                key: value
                for key, value in row.items()
                if key not in {"policy", "task_details"}
            }
            for row in grid_results
        ],
        "selected_router_task_details": best_router["task_details"],
        "manifest": manifest,
        "system_messages_sent": 0,
    }
    selection_path = args.output_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    (args.output_dir / "selection.sha256").write_text(
        f"{file_sha256(selection_path)}  selection.json\n"
    )
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
