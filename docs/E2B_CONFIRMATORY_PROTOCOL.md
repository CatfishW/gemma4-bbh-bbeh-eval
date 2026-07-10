# Gemma 4 E2B Confirmatory Protocol

This protocol separates the completed Gemma 4 E4B prompt search from a confirmatory
Gemma 4 E2B replication. It is committed and pushed before any E2B benchmark item is
sent to the model. E4B findings remain exploratory; E2B test results are untouched
until every screening arm, selection rule, and statistical test is frozen.

## Research question

Can task-conditioned prompt selection improve a small frozen language model over a
direct-answer baseline, and do those gains survive a model-scale transfer from E4B to
E2B?

The primary intervention is a conservative Bayesian reward router (CBRR). A benchmark
task is the context, a prompt strategy is the action, and exact-match correctness is
the reward. This is an offline contextual-bandit formulation, not weight training.
It preserves the deployed model weights and is auditable through a small JSON policy.

## Frozen materials

- Model: `google/gemma-4-E2B-it` at revision
  `70af34e20bd4b7a91f0de6b22675850c43922a03`.
- BBH: `9ee07bd481feebf959a6b59d61ea57bdcf30964d`.
- BBEH: `80d12ca916b7158f22293fcf3144f4d3d854d4be`.
- Unpuzzles and Simple Reasoning:
  `39bc520a2f4c243eb04ce1cc27f28c7c61d12e42`.
- Arm definitions: `experiments/e2b_arm_manifest.jsonl`.
- E4B transfer policy: `experiments/e4b_reward_routed_v2_policy.json`.
- No request contains a system-role message, and the gateway must not inject one.

## Arms and motivation

All 26 registered one-pass prompts are screened, not only likely winners. Three
inference-scaling controls add self-consistency, candidate self-ranking, and a
two-pass key-condition correction. The additions are grounded in published work:

- [Plan-and-Solve](https://aclanthology.org/2023.acl-long.147/) motivates explicit
  decomposition and the `plan_and_solve_plus` arm.
- [Least-to-Most prompting](https://iclr.cc/virtual/2023/poster/12263) motivates
  dependency-ordered subproblems.
- [Self-Consistency](https://iclr.cc/virtual/2023/poster/11718) motivates diverse
  reasoning paths with normalized majority voting.
- [Key-condition verification](https://aclanthology.org/2024.emnlp-main.714/)
  motivates `condition_reconstruction` and the two-pass correction arm.
- [RankPrompt](https://aclanthology.org/2024.lrec-main.1183/) motivates explicit
  candidate comparison rather than naive verbosity-based selection.
- [RLPrompt](https://arxiv.org/abs/2205.12548) and
  [OPTS](https://arxiv.org/abs/2503.01163) motivate treating prompt choice as a
  reward-driven policy problem. CBRR uses a simpler frozen-model contextual bandit
  because the available actions are already interpretable prompts.
- [Inference-time computation evidence](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d3cb4d3573edb4404d39054c04e2b3c1-Abstract-Datasets_and_Benchmarks_Track.html)
  motivates reporting accuracy jointly with token and latency costs.

ReAct is excluded because the evaluated datasets provide neither a retrieval corpus
nor tools; adding them would change the task. Full tree search is excluded from the
confirmatory matrix because its branching and verifier choices are not cost-matched
to the prompt interventions. Candidate self-ranking is the registered bounded-compute
comparison arm.

## Split and selection

Within each task, source indices 0-24 are calibration, 25-49 are validation, and
indices 50 onward are test. Short tasks contribute only the rows they contain. Every
arm runs on calibration and validation. Test is accessed only after the selector has
written `selection.json` and a SHA-256 digest.

The universal arm maximizes validation exact matches, then minimizes completion
tokens, then follows manifest order. CBRR uses only calibration labels to estimate
paired arm-versus-direct rewards per task. A fixed grid of minimum net wins and Beta
posterior superiority thresholds is selected on validation. The final test set contains
direct answer, the E4B global winner (`private_verify`), the novel
`condition_reconstruction` prompt, the validation winner, CBRR, and the frozen E4B
router, with exact duplicates removed.

## Confirmatory inference

The primary outcome is pooled micro exact-match accuracy over all test rows. The
primary comparison is CBRR versus direct answer using an exact two-sided McNemar test.
Secondary finalist comparisons use the same paired test with Holm correction.
Absolute accuracy difference receives a 95% task-stratified paired-bootstrap interval
from 10,000 fixed-seed replicates. Reports also include relative error reduction,
discordant wins/losses, benchmark and task effects, tokens, latency, and errors.

Direct answer and CBRR are repeated over the full test split with seeds 20260710 and
20260711. Results are reported whether positive or negative. No E2B test-driven prompt
revision is allowed under this protocol; any later search must be labeled exploratory
and use a newly declared holdout.

## Validity boundaries

Task-conditioned routing measures generalization to new examples from known tasks,
not zero-shot routing to unseen task identities. Exact-match scoring can undercount
semantically equivalent free-form answers, so raw outputs and normalization fields are
retained. BBH-family data may have appeared in model pretraining; the USR suite adds a
structurally different replication but cannot prove absence of contamination. E2B and
E4B share a model family, so cross-family replication remains future work.
