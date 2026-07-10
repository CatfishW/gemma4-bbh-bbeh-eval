# E2B Task-Cluster Robustness

The primary CBRR effect is +9.03 percentage points over 9,550 test examples in 59 task clusters.

## Cluster sensitivity

- Task-cluster bootstrap pooled 95% interval: [+4.19, +14.59] percentage points.
- Task-cluster bootstrap macro-task 95% interval: [+3.51, +12.56] percentage points.
- Fixed-seed task sign-flip sensitivity p-value: 0.000269997.
- Leave-one-task-out pooled range: [+7.59, +9.40] percentage points.

## Policy audit

CBRR outputs above their arm's registered screening cap: 1,296. Outputs at the actual 256-token test cap: 1,700.

| Selected prompt | Examples | Accuracy | Mean completion tokens | Above registered cap | At test cap |
|---|---:|---:|---:|---:|---:|
| `answer_type_router` | 350 | 43.14% | 2.86 | 0 | 0 |
| `canonical_short` | 1137 | 45.56% | 6.53 | 14 | 14 |
| `chain_of_draft` | 630 | 59.21% | 170.02 | 271 | 250 |
| `compare_then_commit` | 200 | 52.50% | 2.00 | 0 | 0 |
| `concise_cot` | 400 | 84.00% | 172.69 | 0 | 61 |
| `condition_reconstruction` | 300 | 9.67% | 5.08 | 0 | 0 |
| `counterexample_guard` | 150 | 0.67% | 3.75 | 1 | 1 |
| `direct_answer` | 3387 | 21.64% | 64.06 | 791 | 781 |
| `draft_verify` | 300 | 23.67% | 67.59 | 76 | 75 |
| `negation_label_guard` | 350 | 15.14% | 90.55 | 121 | 120 |
| `option_elimination` | 690 | 48.41% | 8.03 | 7 | 7 |
| `premise_conclusion` | 806 | 52.11% | 193.90 | 0 | 376 |
| `private_verify` | 350 | 28.57% | 4.34 | 0 | 0 |
| `rank_two_paths` | 500 | 31.60% | 14.22 | 15 | 15 |

This is a preregistered sensitivity analysis. The exact paired McNemar test remains the primary inferential result.
