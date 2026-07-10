# Gemma 4 E2B Confirmatory Results

Test rows: 9550. Primary seed: 20260709.

## Finalist results

| Arm | Correct | Accuracy | Delta (pp) | Bootstrap 95% (pp) | Wins/Losses | McNemar p | Holm p | Mean completion tokens | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_answer` | 2520/9550 | 26.39% | - | - | - | - | - | 14.68 | 0.585 |
| `private_verify` | 1986/9550 | 20.80% | -5.59 | [-6.23, -4.94] | 378/912 | 2.783e-51 | 8.349e-51 | 20.27 | 0.667 |
| `condition_reconstruction` | 2069/9550 | 21.66% | -4.72 | [-5.41, -4.03] | 440/891 | 1.291e-35 | 2.582e-35 | 20.76 | 0.684 |
| `validation_winner__concise_cot_self_rank_k3` | 3348/9550 | 35.06% | +8.67 | [+7.81, +9.51] | 1548/720 | 4.411e-69 | 1.764e-68 | 681.26 | 11.228 |
| `cbrr_policy` | 3382/9550 | 35.41% | +9.03 | [+8.41, +9.64] | 1100/238 | 1.332e-132 | 6.66e-132 | 65.60 | 1.287 |
| `e4b_policy_transfer` | 2496/9550 | 26.14% | -0.25 | [-0.74, +0.24] | 318/342 | 0.3707 | 0.3707 | 25.59 | 0.728 |

## Primary comparison

CBRR changed the number of correct answers by +862: 1100 direct-to-correct wins and 238 correct-to-wrong losses. The exact two-sided McNemar p-value is 1.33202e-132; the absolute effect is +9.03 percentage points.

## Primary effect by benchmark

| Benchmark | Direct | CBRR | Delta (pp) |
|---|---:|---:|---:|
| `bbeh` | 10.80% | 10.68% | -0.12 |
| `bbh` | 40.73% | 57.33% | +16.61 |
| `usr` | 5.30% | 6.18% | +0.88 |

## Integrity

- Finalists were frozen from indices below 50 before test launch.
- No system-role messages were sent.
- Prediction records include per-example seeds, raw outputs, normalized outputs, usage, latency, and errors.
- Prompt selection generalizes to new rows of known tasks; it is not an unseen-task router.
