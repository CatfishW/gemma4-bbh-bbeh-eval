# Gemma 4 Prompt-Policy Study: Paper Artifact

## Confirmatory result

The untouched E2B test contains 9,550 examples. CBRR changed accuracy by +9.03 percentage points versus direct answer (1100 paired wins, 238 paired losses; exact two-sided McNemar p=1.33202e-132, Holm-adjusted p=6.66009e-132).

## Matched model-scale screening

The same 29 arms were evaluated on the same calibration/validation indices for E2B and E4B. Spearman rank correlation on validation accuracy was 0.888.

| Arm | E2B delta (pp) | E4B delta (pp) | E2B tokens | E4B tokens |
|---|---:|---:|---:|---:|
| `concise_cot_self_rank_k3` | +8.93 | +6.58 | 690.0 | 674.6 |
| `canonical_short` | +2.08 | +1.01 | 19.8 | 13.8 |
| `direct_answer` | +0.00 | +0.00 | 12.9 | 6.8 |
| `option_elimination` | -1.07 | -1.21 | 23.9 | 23.6 |
| `compare_then_commit` | -2.68 | +0.13 | 20.3 | 13.6 |
| `counterexample_guard` | -3.02 | -0.07 | 23.9 | 12.0 |
| `native_format` | -2.21 | -1.07 | 30.4 | 23.4 |
| `rank_two_paths` | -3.15 | -0.34 | 14.5 | 11.6 |
| `selective_verify` | -2.48 | -1.07 | 28.8 | 13.2 |
| `private_verify` | -5.30 | +1.68 | 17.8 | 8.2 |

## Interpretation boundary

- E2B test comparisons are confirmatory under the preregistered protocol.
- E4B matched screening and earlier E4B held-out results are exploratory because E4B informed strategy development.
- Routing generalizes to new examples of known tasks, not to unseen task identities.
- Exact-match results do not establish reasoning faithfulness or absence of training-data contamination.
- Accuracy, token use, latency, errors, negative arms, and raw predictions are all retained.
