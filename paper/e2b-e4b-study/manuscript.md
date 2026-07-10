# Efficient Task-Conditioned Prompt Policies for Gemma 4 Reasoning

## Artifact status

This is a manuscript-oriented study summary, not a peer-reviewed paper. It is
generated from immutable prediction bundles and should be read with the
preregistered protocol, machine-readable results, and validity statement.

## Abstract

We study whether inference-time prompt selection can improve exact-match
reasoning accuracy for Gemma 4 E2B without modifying model weights or injecting
a system prompt. Twenty-nine prompt and selection arms were screened on fixed
calibration and validation indices from BBH, BBEH, and Unpuzzles and Simple
Reasoning (USR). The primary intervention, a conservative Bayesian reward router
(CBRR), treats benchmark task identity as context, prompt strategy as action,
and exact-match correctness as a binary reward. Its routing policy and gate were
frozen before any E2B test request. On 9,550 untouched test examples, CBRR
improved pooled accuracy from 26.39% to 35.41% (+9.03 percentage points; 1,100
paired wins and 238 losses; exact two-sided McNemar p=1.33e-132; task-stratified
bootstrap 95% interval +8.41 to +9.64 points). Two fixed-seed repeats retained
+8.89 and +8.96 points. A task-cluster bootstrap interval was +4.19 to +14.59
points. The effect was concentrated in BBH (+16.61 points), with BBEH flat to
slightly negative (-0.12) and USR modestly positive (+0.88). CBRR used 65.60
mean completion tokens versus 14.68 for direct answer, but slightly exceeded a
681.26-token self-ranking arm. Results support task-conditioned prompt allocation
as a compute-efficient intervention for known task families, not universal or
unseen-task reasoning improvement.

## Study design

### Models and serving

- `SubTokenLLM-E2B` serves `google/gemma-4-E2B-it` in BF16 on one RTX 6000 Ada.
- `SubTokenLLM` serves `google/gemma-4-E4B-it` in BF16 on a second RTX 6000 Ada.
- Every evaluation request contains exactly one `user` message. The router
  selects a backend from the existing `model` field and forwards request bytes
  without adding messages or prompt text.
- Model repository revisions, weight SHA-256 digests, launch commands, software
  versions, GPU snapshots, per-example seeds, outputs, usage, latency, and errors
  are retained in the result bundles.

The [official Gemma model card](https://huggingface.co/google/gemma-4-E2B-it)
defines the evaluated checkpoint family and its reported training cutoff.

### Data and splits

The study includes 12,540 deterministic source records: 6,511 BBH examples,
4,520 BBEH examples, and 1,509 USR examples. Within each of 60 tasks, indices
0-24 are calibration, 25-49 are validation, and indices 50 onward are test.
The short `usr/simple_reasoning/char_ct` task has no test rows, leaving 59 test
task clusters and 9,550 test examples. Dataset revisions and licenses are listed
in the [data statement](../../docs/DATA_MODEL_AND_EVALUATION_STATEMENT.md).

BBH is introduced by [Suzgun et al.](https://arxiv.org/abs/2210.09261), BBEH by
[Kazemi et al.](https://arxiv.org/abs/2502.19187), and USR by
[Dziri et al.](https://arxiv.org/abs/2507.07313).

### Prompt interventions

The 29-arm matrix includes direct and strict-output prompts, private verification,
candidate comparison, concise chain-of-thought and chain-of-draft, Plan-and-Solve,
Least-to-Most, symbolic templates, self-consistency, and bounded self-ranking.
The design was informed by [Plan-and-Solve](https://aclanthology.org/2023.acl-long.147/),
[self-consistency](https://iclr.cc/virtual/2023/poster/11718),
[Least-to-Most](https://iclr.cc/virtual/2023/poster/12263),
[ProCo](https://aclanthology.org/2024.emnlp-main.714/), and
[RankPrompt](https://aclanthology.org/2024.lrec-main.1183/).

CBRR is an offline contextual-bandit policy. For each known task, it estimates
paired arm-versus-direct reward from calibration rows under a Beta-Bernoulli
model. Validation selects a conservative minimum-win and posterior-probability
gate; direct answer is the fallback. The fixed policy then routes new examples
of those tasks. This is related to reward-driven discrete prompt selection in
[RLPrompt](https://arxiv.org/abs/2205.12548) and
[OPTS](https://arxiv.org/abs/2503.01163), but it is not weight-level RL.

### Confirmatory boundary

E4B informed strategy development and is exploratory. The E2B protocol, arm
manifest, split, hypotheses, selection rule, and primary analysis were publicly
committed before E2B benchmark inference. Finalist selection records
`test_rows_read: 0`, and no post-test prompt tuning is included in the
confirmatory claim.

## Results

### E2B untouched test

| Arm | Correct | Accuracy | Delta | Mean completion tokens | Mean latency |
|---|---:|---:|---:|---:|---:|
| Direct answer | 2,520/9,550 | 26.39% | - | 14.68 | 0.585 s |
| Private verify | 1,986/9,550 | 20.80% | -5.59 pp | 20.27 | 0.667 s |
| Condition reconstruction | 2,069/9,550 | 21.66% | -4.72 pp | 20.76 | 0.684 s |
| Self-rank k=3 | 3,348/9,550 | 35.06% | +8.67 pp | 681.26 | 11.228 s |
| CBRR | 3,382/9,550 | 35.41% | +9.03 pp | 65.60 | 1.287 s |
| Frozen E4B policy transfer | 2,496/9,550 | 26.14% | -0.25 pp | 25.59 | 0.728 s |

CBRR reduces direct-answer error by 12.26% relative. Its accuracy Wilson interval
is 34.46% to 36.38%. The primary task-stratified paired-bootstrap interval for
the difference is +8.41 to +9.64 points. The exact McNemar p-value is
`1.332e-132`; after Holm correction across finalist comparisons it is
`6.660e-132`.

CBRR obtains 34 more correct answers than self-ranking while using 10.4 times
fewer completion tokens and 8.7 times lower mean latency. It is not cost-neutral
against direct answer: completion tokens increase by about 4.5 times and mean
latency by about 2.2 times.

### Benchmark heterogeneity

| Benchmark | Direct | CBRR | Difference |
|---|---:|---:|---:|
| BBEH | 10.80% | 10.68% | -0.12 pp |
| BBH | 40.73% | 57.33% | +16.61 pp |
| USR | 5.30% | 6.18% | +0.88 pp |

The pooled claim should therefore not be paraphrased as a uniform improvement
across datasets. Most absolute gain comes from known BBH task families.

### Robustness and task dependence

- Seed 20260710: direct 26.51%, CBRR 35.40%, difference +8.89 points.
- Seed 20260711: direct 26.48%, CBRR 35.45%, difference +8.96 points.
- Semantic agreement with the primary seed is 96.27% to 97.03% for direct and
  95.18% to 95.23% for CBRR.
- Task-cluster bootstrap pooled 95% interval: +4.19 to +14.59 points.
- Task sign-flip sensitivity p-value: 0.000270.
- Leave-one-task-out pooled range: +7.59 to +9.40 points.

Temperature-zero inference is accuracy-stable here but not output-identical, so
the artifacts report both semantic agreement and score replication.

### Compute and truncation

The registered CBRR run has 1,700 outputs at its 256-token cap. A preregistered
cap audit reports 1,296 outputs above the selected arm's screening cap. A
post-hoc direct-fallback replay replaces all 3,387 CBRR rows assigned to direct
answer with their original 64-token-cap baseline outputs. It scores 3,381/9,550
(35.40%), one fewer correct than registered CBRR, at 49.86 mean completion
tokens. This isolates only the fallback cap; it is not a fully token-matched
comparison for specialized arms.

Visible reasoning prompts bind their generation caps frequently. On E2B
screening, concise CoT binds on 71.57% of examples; accuracy is 0.56% among
cap-bound rows and 61.88% among rows that terminate earlier. This association is
descriptive, not causal, because problem difficulty affects termination.

### Output-format reliability

The strict-JSON prompt is not reliable constrained decoding. E2B raw exact-schema
validity is 4.72% and recoverable validity is 39.36%; E4B raw validity is 9.87%
and recoverable validity is 26.79%. Answer accuracy is 8.19% and 12.24%,
respectively. Format validity and answer correctness are reported separately.

### Matched E4B exploration

Across the same 29 arms and matched indices, E2B/E4B validation ranks have
Spearman correlation 0.888, and self-ranking is the global validation winner for
both scales. Exploratory E4B validation scores are 32.55% for direct, 39.13% for
self-ranking, and 41.61% for an E4B-fitted CBRR. An earlier prospective E4B
held-out run scored 36.41% versus 32.90% for direct (+3.51 points). These E4B
figures are not confirmatory because E4B informed the strategy search.

## Negative results

- The preregistered novel single prompt `condition_reconstruction` loses 4.72
  points on E2B test.
- `private_verify`, an earlier E4B single-prompt winner used as a transfer
  candidate, loses 5.59 points on E2B.
- The frozen E4B task router transfers at -0.25 points, with no significant gain.
- Strict JSON, long planning prompts, symbolic proof, and several visible
  reasoning arms perform poorly under their tested caps.

These outcomes are retained in the arm matrix and prediction logs rather than
filtered from the study narrative.

## RL decision

Weight-level RL was considered and deliberately not mixed into this experiment.
Training an adapter on benchmark rewards would create a different model, require
a disjoint training corpus and a new untouched test, and confound prompt-policy
effects with checkpoint changes. CBRR provides an interpretable reward-allocation
test without changing either deployed weight set. A future GRPO or RLPrompt-style
adapter study should preregister a separate checkpoint, training data, baselines,
and holdout before optimization.

## Validity limitations

1. CBRR sees known task identity and calibration rewards. It does not route
   unseen task identities zero-shot.
2. The result is dominated by BBH; BBEH does not improve and USR improves only
   modestly.
3. Public benchmark contamination cannot be excluded, and exact match does not
   measure reasoning faithfulness.
4. E2B and E4B share a model family, so this is not cross-family replication.
5. Inference caps bind heavily for several reasoning arms, and cost comparisons
   are not globally token-matched.
6. Seed repeats show non-identical outputs even at temperature zero.
7. The direct-fallback replay is explicitly post hoc; the registered CBRR result
   remains the primary estimate.

## Analysis implementation note

After all registered inference completed, the first analysis attempt raised an
`OverflowError` while converting a large exact-binomial integer to floating
point. No inference was rerun and no prediction was changed. The McNemar tail was
reimplemented with a numerically stable log-sum-exp calculation, verified against
a high-precision integer/decimal reference, and covered by a large-discordance
unit test. The original traceback and before/after script hashes are preserved in
the [analysis-fix provenance](../../results/e2b-confirmatory-20260709_231405/provenance/analysis-fix.txt).

## Artifact map

- [Confirmatory report](../../results/e2b-confirmatory-20260709_231405/analysis/report.md)
- [Representative questions and answers](../../results/e2b-confirmatory-20260709_231405/analysis/examples.md)
- [Task-cluster sensitivity](cluster-robustness/cluster_robustness.md)
- [Direct-fallback replay](budget-sensitivity/fallback_replay_sensitivity.md)
- [Inference-budget audit](budget-audit/inference_budget_audit.md)
- [Strict-JSON audit](format-audit/format_audit.md)
- [Cross-model screening table](cross_model_screening.csv)
- [Confirmatory CSV](e2b_confirmatory_test.csv)
- [LaTeX result table](e2b_confirmatory_table.tex)
- [Reproducibility checklist](reproducibility_checklist.md)
- [Full bibliography](../references.bib)
