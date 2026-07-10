# Inference-Budget Audit

Cap binding is reported separately from accuracy so truncated reasoning is not misinterpreted as evidence that additional computation is intrinsically harmful.
Capped-versus-uncapped accuracies are descriptive, not causal: easier examples may naturally terminate sooner.

| Model | Arm | Accuracy | Generation cap rate | Example cap rate | Capped accuracy | Uncapped accuracy | Mean completion tokens | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E2B | `answer_type_router` | 21.47% | 10.17% | 10.17% | 0.00% | 23.90% | 17.94 | 0 |
| E2B | `canonical_short` | 26.92% | 15.72% | 15.72% | 0.21% | 31.90% | 19.54 | 0 |
| E2B | `careful_direct` | 20.77% | 24.92% | 24.92% | 0.00% | 27.66% | 35.81 | 0 |
| E2B | `chain_of_draft` | 18.43% | 70.74% | 70.74% | 0.28% | 62.29% | 168.42 | 0 |
| E2B | `compare_then_commit` | 21.77% | 12.84% | 12.84% | 0.00% | 24.98% | 20.77 | 0 |
| E2B | `concise_cot` | 17.99% | 71.57% | 71.57% | 0.56% | 61.88% | 226.27 | 0 |
| E2B | `concise_cot_sc_k3` | 18.86% | 71.44% | 77.32% | 5.84% | 63.27% | 680.27 | 0 |
| E2B | `concise_cot_self_rank_k3` | 34.01% | 71.27% | 77.19% | 25.48% | 62.90% | 687.86 | 0 |
| E2B | `condition_reconstruction` | 21.07% | 10.03% | 10.03% | 0.00% | 23.42% | 17.77 | 0 |
| E2B | `constraint_guard` | 21.14% | 11.04% | 11.04% | 1.21% | 23.61% | 18.73 | 0 |
| E2B | `counterexample_guard` | 22.54% | 15.15% | 15.15% | 0.00% | 26.57% | 23.99 | 0 |
| E2B | `direct_answer` | 25.22% | 13.34% | 13.34% | 1.00% | 28.95% | 12.82 | 0 |
| E2B | `direct_key_condition_refine` | 19.16% | 13.21% | 30.60% | 0.00% | 27.61% | 35.70 | 0 |
| E2B | `draft_verify` | 10.94% | 47.22% | 47.22% | 0.07% | 20.66% | 71.94 | 0 |
| E2B | `fast_slow_gate` | 8.39% | 65.59% | 65.59% | 0.00% | 24.39% | 87.21 | 0 |
| E2B | `least_to_most` | 2.14% | 97.39% | 97.39% | 0.17% | 75.64% | 254.59 | 0 |
| E2B | `native_format` | 23.11% | 25.32% | 25.32% | 0.00% | 30.94% | 30.05 | 0 |
| E2B | `negation_label_guard` | 12.98% | 62.04% | 62.04% | 0.22% | 33.83% | 81.71 | 0 |
| E2B | `option_elimination` | 24.11% | 15.65% | 15.65% | 0.00% | 28.59% | 23.72 | 0 |
| E2B | `plan_and_solve` | 4.21% | 95.32% | 95.32% | 0.18% | 86.43% | 252.78 | 0 |
| E2B | `plan_and_solve_plus` | 1.44% | 98.03% | 98.03% | 0.07% | 69.49% | 255.07 | 0 |
| E2B | `premise_conclusion` | 17.16% | 73.51% | 73.51% | 0.32% | 63.89% | 230.72 | 0 |
| E2B | `private_verify` | 19.80% | 9.97% | 9.97% | 0.00% | 21.99% | 18.16 | 0 |
| E2B | `rank_two_paths` | 21.64% | 7.96% | 7.96% | 0.42% | 23.47% | 15.47 | 0 |
| E2B | `raw` | 1.74% | 78.53% | 78.53% | 0.21% | 7.32% | 226.38 | 0 |
| E2B | `selective_verify` | 23.18% | 18.80% | 18.80% | 0.00% | 28.54% | 28.33 | 0 |
| E2B | `step_back` | 4.55% | 93.55% | 93.55% | 0.14% | 68.39% | 251.89 | 0 |
| E2B | `strict_json` | 8.19% | 58.60% | 58.60% | 0.00% | 19.79% | 64.50 | 0 |
| E2B | `symbolic_proof` | 2.14% | 97.42% | 97.42% | 0.21% | 75.32% | 255.09 | 0 |
| E4B | `answer_type_router` | 27.49% | 3.61% | 3.61% | 0.00% | 28.52% | 9.55 | 0 |
| E4B | `canonical_short` | 32.47% | 9.67% | 9.67% | 0.00% | 35.95% | 13.96 | 0 |
| E4B | `careful_direct` | 31.24% | 8.96% | 8.96% | 0.00% | 34.31% | 15.85 | 0 |
| E4B | `chain_of_draft` | 29.40% | 58.83% | 58.83% | 0.11% | 71.24% | 153.53 | 0 |
| E4B | `compare_then_commit` | 31.74% | 7.22% | 7.22% | 0.00% | 34.21% | 13.93 | 0 |
| E4B | `concise_cot` | 24.45% | 66.86% | 66.86% | 0.10% | 73.56% | 220.50 | 0 |
| E4B | `concise_cot_sc_k3` | 25.52% | 66.92% | 73.98% | 7.87% | 75.71% | 663.11 | 0 |
| E4B | `concise_cot_self_rank_k3` | 39.00% | 67.26% | 74.45% | 28.89% | 68.46% | 673.18 | 0 |
| E4B | `condition_reconstruction` | 30.20% | 8.16% | 8.16% | 0.00% | 32.88% | 15.16 | 0 |
| E4B | `constraint_guard` | 31.91% | 6.09% | 6.09% | 0.00% | 33.97% | 12.61 | 0 |
| E4B | `counterexample_guard` | 31.30% | 5.55% | 5.55% | 0.00% | 33.14% | 12.13 | 0 |
| E4B | `direct_answer` | 31.84% | 3.11% | 3.11% | 0.00% | 32.86% | 6.80 | 0 |
| E4B | `direct_key_condition_refine` | 12.91% | 3.08% | 63.34% | 0.48% | 34.40% | 49.02 | 0 |
| E4B | `draft_verify` | 27.29% | 12.71% | 12.71% | 0.00% | 31.26% | 23.81 | 0 |
| E4B | `fast_slow_gate` | 19.13% | 47.79% | 47.79% | 0.00% | 36.64% | 65.36 | 0 |
| E4B | `least_to_most` | 3.58% | 96.02% | 96.02% | 0.10% | 87.39% | 253.74 | 0 |
| E4B | `native_format` | 30.47% | 19.30% | 19.30% | 0.00% | 37.75% | 23.72 | 0 |
| E4B | `negation_label_guard` | 19.50% | 53.48% | 53.48% | 0.00% | 41.91% | 71.39 | 0 |
| E4B | `option_elimination` | 30.13% | 16.32% | 16.32% | 0.00% | 36.01% | 24.85 | 0 |
| E4B | `plan_and_solve` | 6.45% | 92.21% | 92.21% | 0.11% | 81.55% | 251.13 | 0 |
| E4B | `plan_and_solve_plus` | 2.14% | 97.46% | 97.46% | 0.07% | 81.58% | 254.76 | 0 |
| E4B | `premise_conclusion` | 24.52% | 66.96% | 66.96% | 0.30% | 73.58% | 226.00 | 0 |
| E4B | `private_verify` | 33.21% | 2.37% | 2.37% | 0.00% | 34.02% | 8.11 | 0 |
| E4B | `rank_two_paths` | 31.17% | 5.62% | 5.62% | 0.00% | 33.03% | 11.98 | 0 |
| E4B | `raw` | 2.31% | 74.95% | 74.95% | 0.09% | 8.95% | 220.59 | 0 |
| E4B | `selective_verify` | 30.77% | 6.76% | 6.76% | 0.00% | 33.00% | 13.54 | 0 |
| E4B | `step_back` | 8.49% | 89.26% | 89.26% | 0.07% | 78.50% | 248.74 | 0 |
| E4B | `strict_json` | 12.24% | 70.77% | 70.77% | 0.00% | 41.88% | 72.81 | 0 |
| E4B | `symbolic_proof` | 2.21% | 97.79% | 97.79% | 0.34% | 84.85% | 254.86 | 0 |
