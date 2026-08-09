# Detailed Results

Full result tables for the prompt-strategy study. The README keeps only the
headline numbers; everything else lives here. All numbers are exact-match
accuracy produced by the scorer in `eval_benchmarks.py`.

## 1. Frozen E2B test split (9,550 examples, index >= 50)

The confirmatory comparison on the untouched test split. CBRR is the
conservative Bayesian reward router fitted on calibration rows only; model
weights unchanged.

| E2B arm | Correct | Accuracy | Delta vs direct | Mean completion tokens |
|---|---:|---:|---:|---:|
| `direct_answer` | 2,520/9,550 | 26.39% | - | 14.68 |
| `concise_cot_self_rank_k3` | 3,348/9,550 | 35.06% | +8.67 pp | 681.26 |
| `cbrr_policy` | 3,382/9,550 | 35.41% | +9.03 pp | 65.60 |

CBRR produced 1,100 paired wins and 238 losses versus direct answer. The exact
two-sided McNemar p-value is `1.33e-132`, the Holm-adjusted p-value is
`6.66e-132`, and the task-stratified bootstrap interval is +8.41 to +9.64
percentage points. A task-cluster bootstrap gives +4.19 to +14.59 points; both
additional fixed-seed repeats retain +8.89 to +8.96 points.

### By dataset

The test split contains 5,161 BBH, 3,370 BBEH, and 1,019 USR examples.

| Finalist | BBH | BBEH | USR | Overall |
|---|---:|---:|---:|---:|
| `direct_answer` | 2,102/5,161 (40.73%) | 364/3,370 (10.80%) | 54/1,019 (5.30%) | 2,520/9,550 (26.39%) |
| `private_verify` | 1,585/5,161 (30.71%) | 315/3,370 (9.35%) | 86/1,019 (8.44%) | 1,986/9,550 (20.80%) |
| `condition_reconstruction` | 1,689/5,161 (32.73%) | 348/3,370 (10.33%) | 32/1,019 (3.14%) | 2,069/9,550 (21.66%) |
| `concise_cot_self_rank_k3` | 2,958/5,161 (57.31%) | 347/3,370 (10.30%) | 43/1,019 (4.22%) | 3,348/9,550 (35.06%) |
| `cbrr_policy` | **2,959/5,161 (57.33%)** | 360/3,370 (10.68%) | 63/1,019 (6.18%) | **3,382/9,550 (35.41%)** |
| `e4b_policy_transfer` | 2,095/5,161 (40.59%) | 347/3,370 (10.30%) | 54/1,019 (5.30%) | 2,496/9,550 (26.14%) |

CBRR's effect is concentrated in BBH (+16.61 points). BBEH is flat to slightly
negative (-0.12 points), and USR improves modestly (+0.88 points). CBRR uses
about 4.5 times the completion tokens of direct answer, although it slightly
outperforms self-ranking with about one tenth of the completion tokens.

## 2. Matched validation split by model (675 BBH, 575 BBEH, 240 USR)

CBRR is task-conditioned, so its rows are separate from the universal prompt
matrix. E4B results are exploratory because E4B informed strategy development.

| Model and strategy | BBH | BBEH | USR | Overall |
|---|---:|---:|---:|---:|
| E2B `direct_answer` | 40.74% | 10.43% | 15.83% | 25.03% |
| E2B `concise_cot_self_rank_k3` | 59.11% | 11.30% | 17.50% | 33.96% |
| E2B `cbrr_policy` | 57.78% | 11.48% | 16.67% | 33.29% |
| E4B `direct_answer` | 50.52% | 14.61% | 25.00% | 32.55% |
| E4B `concise_cot_self_rank_k3` | 65.93% | 15.30% | 20.83% | 39.13% |
| E4B `cbrr_policy` | **67.56%** | **16.35%** | **29.17%** | **41.61%** |

## 3. All 29 universal arms by model and dataset (matched validation)

| Strategy | E2B BBH | E2B BBEH | E2B USR | E2B all | E4B BBH | E4B BBEH | E4B USR | E4B all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `concise_cot_self_rank_k3` | **59.11%** | 11.30% | **17.50%** | **33.96%** | **65.93%** | **15.30%** | 20.83% | **39.13%** |
| `canonical_short` | 45.33% | **11.48%** | 13.33% | 27.11% | 55.26% | 13.74% | 20.00% | 33.56% |
| `direct_answer` | 40.74% | 10.43% | 15.83% | 25.03% | 50.52% | 14.61% | **25.00%** | 32.55% |
| `option_elimination` | 41.63% | 9.22% | 9.58% | 23.96% | 52.74% | 11.30% | 19.17% | 31.34% |
| `compare_then_commit` | 35.26% | 10.96% | 13.33% | 22.35% | 53.48% | 14.09% | 18.75% | 32.68% |
| `counterexample_guard` | 35.85% | 9.39% | 13.33% | 22.01% | 52.59% | 14.78% | 18.33% | 32.48% |
| `native_format` | 38.37% | 7.83% | 15.00% | 22.82% | 52.59% | 11.30% | 20.42% | 31.48% |
| `rank_two_paths` | 34.67% | 10.43% | 13.33% | 21.88% | 52.44% | 14.09% | 18.75% | 32.21% |
| `selective_verify` | 37.48% | 8.35% | 14.58% | 22.55% | 51.11% | 14.09% | 17.92% | 31.48% |
| `private_verify` | 29.93% | 10.61% | 12.92% | 19.73% | 55.70% | 13.91% | 22.50% | 34.23% |
| `constraint_guard` | 33.04% | 10.61% | 11.67% | 20.94% | 54.07% | 14.09% | 18.33% | 32.89% |
| `careful_direct` | 31.11% | 8.70% | 14.17% | 19.73% | 51.85% | 14.09% | 17.50% | 31.74% |
| `condition_reconstruction` | 32.59% | 9.91% | 13.33% | 20.74% | 50.96% | 12.52% | 17.50% | 30.74% |
| `answer_type_router` | 36.15% | 9.91% | 10.83% | 21.95% | 46.67% | 13.74% | 10.42% | 28.12% |
| `chain_of_draft` | 34.22% | 2.78% | 6.67% | 17.65% | 54.22% | 3.48% | 17.50% | 28.72% |
| `concise_cot_sc_k3` | 38.81% | 2.26% | 5.00% | 19.26% | 50.22% | 3.30% | 7.92% | 25.30% |
| `concise_cot` | 35.41% | 1.57% | 4.58% | 17.38% | 46.67% | 3.83% | 10.42% | 24.30% |
| `premise_conclusion` | 32.74% | 2.09% | 5.42% | 16.51% | 47.56% | 2.09% | 8.75% | 23.76% |
| `draft_verify` | 15.26% | 6.61% | 5.42% | 10.34% | 42.81% | 14.09% | 17.08% | 27.58% |
| `negation_label_guard` | 23.56% | 4.00% | 2.50% | 12.62% | 33.78% | 8.70% | 9.17% | 20.13% |
| `direct_key_condition_refine` | 31.70% | 5.91% | 14.58% | 18.99% | 19.26% | 5.04% | 13.33% | 12.82% |
| `fast_slow_gate` | 13.04% | 4.17% | 6.25% | 8.52% | 28.74% | 8.52% | 17.08% | 19.06% |
| `strict_json` | 14.07% | 4.87% | 1.67% | 8.52% | 20.44% | 7.13% | 3.75% | 12.62% |
| `step_back` | 8.15% | 0.35% | 1.67% | 4.09% | 18.81% | 0.35% | 1.67% | 8.93% |
| `plan_and_solve` | 8.89% | 0.00% | 0.42% | 4.09% | 12.74% | 0.00% | 1.25% | 5.97% |
| `least_to_most` | 4.00% | 0.00% | 0.00% | 1.81% | 7.70% | 0.17% | 1.25% | 3.76% |
| `raw` | 1.48% | 3.13% | 0.83% | 2.01% | 0.74% | 4.00% | 2.08% | 2.21% |
| `symbolic_proof` | 4.00% | 0.00% | 0.00% | 1.81% | 4.44% | 0.17% | 0.83% | 2.21% |
| `plan_and_solve_plus` | 2.81% | 0.00% | 0.00% | 1.28% | 4.15% | 0.17% | 0.83% | 2.08% |

## 4. Reward-routed policy (earlier, pre-confirmatory iteration)

See [PROMPT_OPTIMIZATION_RESEARCH.md](PROMPT_OPTIMIZATION_RESEARCH.md) for the
v1/v2 routed-policy results on the 11,040-example held-out set
(`reward_routed_v2`: 36.41% vs 32.90% direct, fully live, zero request
errors), the research basis, and negative results.

## 5. Related artifacts

- [Confirmatory report](../results/e2b-confirmatory-20260709_231405/analysis/report.md)
- [Cluster sensitivity](../paper/e2b-e4b-study/cluster-robustness/cluster_robustness.md)
- [Strict JSON audit](../paper/e2b-e4b-study/format-audit/format_audit.md)
- [Cross-model CSV](../paper/e2b-e4b-study/cross_model_screening.csv)
- [Direct-fallback replay](../paper/e2b-e4b-study/budget-sensitivity/fallback_replay_sensitivity.md)
- [Deployment snapshot](../paper/e2b-e4b-study/deployment-verification-20260710.md)
