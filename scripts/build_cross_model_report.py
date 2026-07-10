#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready E2B/E4B result artifacts.")
    parser.add_argument("--e2b-run", type=Path, required=True)
    parser.add_argument("--e4b-run", type=Path, required=True)
    parser.add_argument("--e4b-prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = ((index + 1) + end) / 2
        for name, _ in ordered[index:end]:
            ranks[name] = rank
        index = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def load_selection(run: Path) -> dict:
    return json.loads((run / "selection" / "selection.json").read_text())


def ranking_map(selection: dict) -> dict[str, dict]:
    return {str(row["arm"]): row for row in selection["arm_rankings"]}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e2b_selection = load_selection(args.e2b_run)
    e4b_selection = load_selection(args.e4b_run)
    e2b_analysis = json.loads((args.e2b_run / "analysis" / "analysis.json").read_text())
    e4b_prior = json.loads(args.e4b_prior.read_text())
    e2b = ranking_map(e2b_selection)
    e4b = ranking_map(e4b_selection)
    arms = sorted(set(e2b) & set(e4b))
    if set(e2b) != set(e4b):
        raise SystemExit("E2B and E4B screening arm sets differ")

    direct_e2b = e2b["direct_answer"]["validation"]["accuracy"]
    direct_e4b = e4b["direct_answer"]["validation"]["accuracy"]
    e2b_accuracy = {arm: e2b[arm]["validation"]["accuracy"] for arm in arms}
    e4b_accuracy = {arm: e4b[arm]["validation"]["accuracy"] for arm in arms}
    e2b_ranks = average_ranks(e2b_accuracy)
    e4b_ranks = average_ranks(e4b_accuracy)
    rank_correlation = pearson(
        [e2b_ranks[arm] for arm in arms],
        [e4b_ranks[arm] for arm in arms],
    )

    screening_rows = []
    for arm in arms:
        screening_rows.append(
            {
                "arm": arm,
                "e2b_calibration_accuracy": e2b[arm]["calibration"]["accuracy"],
                "e2b_validation_accuracy": e2b_accuracy[arm],
                "e2b_delta_vs_direct_pp": 100 * (e2b_accuracy[arm] - direct_e2b),
                "e2b_validation_rank": e2b_ranks[arm],
                "e2b_mean_completion_tokens": e2b[arm]["validation"][
                    "mean_completion_tokens"
                ],
                "e4b_calibration_accuracy": e4b[arm]["calibration"]["accuracy"],
                "e4b_validation_accuracy": e4b_accuracy[arm],
                "e4b_delta_vs_direct_pp": 100 * (e4b_accuracy[arm] - direct_e4b),
                "e4b_validation_rank": e4b_ranks[arm],
                "e4b_mean_completion_tokens": e4b[arm]["validation"][
                    "mean_completion_tokens"
                ],
            }
        )
    screening_rows.sort(
        key=lambda row: (
            -(row["e2b_delta_vs_direct_pp"] + row["e4b_delta_vs_direct_pp"]),
            row["arm"],
        )
    )
    write_csv(
        args.output_dir / "cross_model_screening.csv",
        screening_rows,
        list(screening_rows[0]),
    )

    test_rows = []
    for arm in e2b_analysis["arm_order"]:
        metrics = e2b_analysis["arms"][arm]
        comparison = e2b_analysis["comparisons_vs_direct"].get(arm)
        test_rows.append(
            {
                "arm": arm,
                "correct": metrics["correct"],
                "total": metrics["total"],
                "accuracy": metrics["accuracy"],
                "delta_vs_direct_pp": (
                    comparison["accuracy_point_difference"] if comparison else 0.0
                ),
                "paired_wins": comparison["paired_wins"] if comparison else "",
                "paired_losses": comparison["paired_losses"] if comparison else "",
                "mcnemar_p": (
                    comparison["exact_mcnemar_two_sided_p"] if comparison else ""
                ),
                "holm_p": comparison["holm_adjusted_p"] if comparison else "",
                "bootstrap_low_pp": (
                    100 * comparison["task_stratified_bootstrap_95"][0]
                    if comparison
                    else ""
                ),
                "bootstrap_high_pp": (
                    100 * comparison["task_stratified_bootstrap_95"][1]
                    if comparison
                    else ""
                ),
                "mean_completion_tokens": metrics["completion_tokens"]["mean"],
                "mean_latency_seconds": metrics["elapsed_seconds"]["mean"],
            }
        )
    write_csv(args.output_dir / "e2b_confirmatory_test.csv", test_rows, list(test_rows[0]))

    summary = {
        "protocol": e2b_analysis["protocol"],
        "e2b_test_examples": e2b_analysis["test_examples"],
        "screening_arms_per_model": len(arms),
        "e2b_validation_winner": e2b_selection["universal_validation_winner"],
        "e4b_validation_winner": e4b_selection["universal_validation_winner"],
        "validation_rank_spearman": rank_correlation,
        "e2b_primary": e2b_analysis["comparisons_vs_direct"][e2b_analysis["primary"]],
        "e4b_prior_heldout": e4b_prior,
        "system_messages_sent": 0,
    }
    (args.output_dir / "cross_model_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    primary = summary["e2b_primary"]
    top = screening_rows[:10]
    lines = [
        "# Gemma 4 Prompt-Policy Study: Paper Artifact",
        "",
        "## Confirmatory result",
        "",
        f"The untouched E2B test contains {e2b_analysis['test_examples']:,} examples. "
        f"CBRR changed accuracy by {primary['accuracy_point_difference']:+.2f} percentage "
        f"points versus direct answer ({primary['paired_wins']} paired wins, "
        f"{primary['paired_losses']} paired losses; exact two-sided McNemar "
        f"p={primary['exact_mcnemar_two_sided_p']:.6g}, Holm-adjusted "
        f"p={primary['holm_adjusted_p']:.6g}).",
        "",
        "## Matched model-scale screening",
        "",
        f"The same {len(arms)} arms were evaluated on the same calibration/validation "
        f"indices for E2B and E4B. Spearman rank correlation on validation accuracy was "
        f"{rank_correlation:.3f}.",
        "",
        "| Arm | E2B delta (pp) | E4B delta (pp) | E2B tokens | E4B tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| `{row['arm']}` | {row['e2b_delta_vs_direct_pp']:+.2f} | "
            f"{row['e4b_delta_vs_direct_pp']:+.2f} | "
            f"{row['e2b_mean_completion_tokens']:.1f} | "
            f"{row['e4b_mean_completion_tokens']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- E2B test comparisons are confirmatory under the preregistered protocol.",
            "- E4B matched screening and earlier E4B held-out results are exploratory because E4B informed strategy development.",
            "- Routing generalizes to new examples of known tasks, not to unseen task identities.",
            "- Exact-match results do not establish reasoning faithfulness or absence of training-data contamination.",
            "- Accuracy, token use, latency, errors, negative arms, and raw predictions are all retained.",
            "",
        ]
    )
    (args.output_dir / "paper_summary.md").write_text("\n".join(lines))

    latex = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Arm & Correct & Accuracy (\\%) & $\\Delta$ (pp) & Holm $p$ \\\\",
        "\\midrule",
    ]
    for row in test_rows:
        holm = "--" if row["holm_p"] == "" else f"{float(row['holm_p']):.3g}"
        latex.append(
            f"{latex_escape(row['arm'])} & {row['correct']}/{row['total']} & "
            f"{100 * row['accuracy']:.2f} & {row['delta_vs_direct_pp']:+.2f} & {holm} \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    (args.output_dir / "e2b_confirmatory_table.tex").write_text("\n".join(latex))

    checklist = [
        "# Reproducibility Checklist",
        "",
        "- [x] Exact model repository, revision, and weight SHA-256 recorded.",
        "- [x] Exact dataset revisions and deterministic source-index splits recorded.",
        "- [x] Protocol committed before E2B benchmark inference.",
        "- [x] All prompt templates and inference configurations version controlled.",
        "- [x] Per-example seeds, raw predictions, normalized predictions, usage, latency, and errors retained.",
        "- [x] Finalists selected automatically without E2B test-label access.",
        "- [x] Paired exact tests, family-wise correction, and stratified bootstrap intervals reported.",
        "- [x] Direct answer and primary policy repeated at two additional seeds.",
        "- [x] Matched E4B screening labeled exploratory rather than confirmatory.",
        "- [x] Question, ground truth, direct answer, routed answer, wins, and losses included.",
        "- [x] No system-role messages or gateway system-prompt injection.",
        "- [x] Negative results and validity limitations retained.",
        "",
    ]
    (args.output_dir / "reproducibility_checklist.md").write_text("\n".join(checklist))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
