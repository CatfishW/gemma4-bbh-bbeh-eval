# One Posterior to Rule the Rollouts: Unbiased Adaptive-Compute Reinforcement Learning for Verifiable Reasoning

**Working title alternatives:** "VOLT: Variance-Optimal Allocation of Tokens for
RLVR with Predictable Bayesian Baselines"

**Status:** manuscript draft; experiment tables are filled by
`scripts/summarize_rl_runs.py` outputs from the runs referenced in Section 5.
Placeholders marked `[[TBD]]`.

## Abstract

Group-relative policy optimization (GRPO) and its descendants spend their
generation budget uniformly: every prompt receives the same number of rollouts,
although prompts the policy already masters (or cannot yet touch) provably
contribute near-zero gradient. Recent work patches this waste by rejection
(DAPO's dynamic sampling), heuristic elimination (Reinforce-Ada), or batch-level
allocation with plug-in difficulty estimates (DynaMO, Knapsack-RL, DUET) — but
every such scheme couples the sampling rule to in-sample advantage estimates,
and none establishes that the resulting adaptively-collected gradient is still
an unbiased ascent direction. We give the missing piece and build a method
around it. A single discounted, hierarchically-pooled Beta posterior over
per-prompt success probability simultaneously (i) drives a variance-optimal
square-root water-filling allocation of rollouts under an explicit token
budget, (ii) supplies a *predictable* (filtration-measurable) baseline that
makes the policy gradient conditionally unbiased under *any* adaptive
allocation or stopping rule — a martingale argument that fails for group-mean
baselines — and (iii) modulates a primal-dual controller that constrains
deployed completion length. Degenerate all-correct/all-wrong groups, fatal to
GRPO and merely discarded by DAPO, become informative consolidation updates
with automatically annealing weight. On the frozen BBH/BBEH/USR evaluation
suite of a deployed Gemma 4 E2B model — training only on the 1,100-prompt
calibration split of a pre-registered protocol — VOLT reaches [[TBD]]% on the
untouched 9,550-example test split versus [[TBD]]% for compute-matched GRPO and
35.41% for the strongest prior prompt-policy result, while using [[TBD]]× fewer
generated tokens to reach GRPO's best validation accuracy and holding unseen-task
generalization ([[TBD]]). Theory, code, per-iteration telemetry, and the frozen
protocol are released.

## 1. Introduction

RL with verifiable rewards (RLVR) is the standard recipe for improving the
reasoning of small language models, and its cost is dominated by rollout
generation. The de-facto algorithm, GRPO, draws a fixed-size group of G
rollouts per prompt and normalizes rewards within the group. This design has a
known failure mode with binary rewards: when all G rollouts agree, the
advantage is identically zero and the group's tokens are wasted. The waste is
not a corner case — degenerate-group rates above 60-80% late in training are
reported across the literature — and it worsens exactly when learning succeeds.

The obvious response is to adapt the sampling: skip saturated prompts, resample
degenerate groups, or size groups by estimated difficulty. The literature has
converged on this at speed (DAPO; GRESO; SPEED; Reinforce-Ada; DynaMO;
Knapsack-RL; DUET; MoPPS; BOTS). But adaptivity has a price that has gone
unexamined: once *which* and *how many* rollouts depend on past outcomes, the
usual unbiasedness argument for the policy gradient no longer applies as
stated, and baselines computed from the same adaptively-collected pool (as in
Reinforce-Ada) or from the current group (as in GRPO/DAPO) entangle the
stopping rule with the estimator. Reviewer-facing versions of these methods
either do not notice or explicitly defer the question.

Our starting point is that the fix is classical: make every quantity the
estimator subtracts *predictable* — measurable with respect to strictly-past
information — and the adaptively-allocated gradient becomes a martingale
around a predictably-weighted combination of true per-prompt gradients
(Theorem 1). Predictability is not a technicality; it changes what the
algorithm can do:

1. **Allocation and estimation decouple.** Any allocation rule that reads only
   the past is licensed, so we can use the variance-*optimal* one: square-root
   water-filling in the posterior Bernoulli variance E[p(1-p)] under a token
   budget (Proposition 2), rather than rejection patches.
2. **Groups become unnecessary.** A single rollout carries unbiased signal
   against a posterior baseline, so budget can be spread as thinly as
   exploration demands (n_i = 1 probes) and concentrated where learning is
   fastest, without the n >= 2 constraint of group baselines.
3. **Degenerate groups become signal.** An all-correct group on a prompt with
   posterior mean b < 1 yields positive advantage 1 - b on every rollout:
   consolidation of fragile successes, annealing automatically as b -> 1.

The same posterior that prices difficulty for the allocator is the baseline —
one state, three roles (allocation, baseline, budget control). We pool the
posterior hierarchically across the benchmark's task structure, which is, to
our knowledge, the first use of partial pooling inside an RLVR training loop;
on task-structured suites the task prior supplies a calibrated baseline for
prompts the allocator has barely visited.

Contributions:

- **Theory.** (a) Unbiasedness of adaptively-allocated policy gradients under
  predictable baselines, with corollaries locating the bias/deadness of
  group-mean baselines and adaptive-stopping pool baselines (Theorem 1,
  Corollaries 1.1-1.3). (b) The variance-optimal token-budgeted allocation and
  its square-root water-filling form, with the variance model stated explicitly
  so its disagreement with Neyman-style rules (DUET) is a testable assumption
  (Proposition 2). (c) Variance dominance of posterior baselines over
  leave-one-out within its error radius (Proposition 3).
- **Algorithm.** VOLT: a group-free RLVR loop where a discounted hierarchical
  Beta posterior drives allocation, baselines, and a success-conditioned
  primal-dual length controller; GRPO, Dr. GRPO, and DAPO-style dynamic
  sampling are recovered in the same code path for controlled comparison.
- **A pre-registered-style evaluation.** We adopt an existing frozen protocol
  (calibration/validation/test by within-task index) on a deployed Gemma 4 E2B
  suite (BBH, BBEH, USR; 12,540 examples), train only on calibration rows,
  hold out 16 tasks entirely, and report single-shot frozen-test results
  directly comparable to the protocol's published prompt-optimization and
  bandit-routing results.

## 2. Related work

(Condensed here; full positioning in `method.md` Section 8.)

**Estimator fixes.** Dr. GRPO removes std-division and length-normalization
bias; RLOO uses leave-one-out baselines (n >= 2); SPO replaces the group
baseline with a KL-discounted Beta tracker — the closest prior work to our
baseline component, but with uniform sampling, no pooling, and unbiasedness
argued only for fixed (non-adaptive) sampling. KRPO (Kalman), A*-PO (offline
V*), OPO (optimal length-weighted baseline), James-Stein shrinkage baselines.

**Adaptive sampling and allocation.** DAPO resamples degenerate groups
(rejection; hyperbolic cost E[N|p] = 1/(p(1-p)) - 1); GRESO skips predicted
zero-variance prompts; SPEED and PCL select by SNR/predicted difficulty;
Reinforce-Ada runs successive elimination with a pooled baseline and leaves its
stopping bias unanalyzed; DynaMO derives sigma-proportional water-level
allocation (frequentist, no costs); Knapsack-RL solves a value/cost knapsack
with stale estimates; DUET couples Neyman allocation with a training-token
budget dual; MoPPS/BOTS do Beta-posterior Thompson selection. None makes the
allocation-estimation interaction unbiased, none pools difficulty
hierarchically, and none ties the difficulty state to the baseline.

**Budgeted/length-controlled RL.** L1/LCPO, Kimi k1.5, ALP, Leash, IBPO,
ThinkPrune, GFPO. Our controller is deliberately standard (success-conditioned
penalty, projected dual ascent); the delta is that it reads the same posterior
state as everything else.

**Classical.** Optimal baselines and variance analysis (Williams 1992; Weaver
& Tao 2001; Greensmith et al. 2004); martingale/optional-stopping arguments
from sequential analysis; Neyman allocation; non-stationary discounted
Bayesian tracking.

## 3. Method

See `method.md` for the operational description and `theory.md` for formal
statements and proofs. Summary:

- Discounted Beta posterior per training prompt; task-level partial pooling
  with prior strength m; snapshot frozen before each iteration (predictability
  boundary).
- Allocation: n_i ∝ sqrt(E[p_i(1-p_i)]) under rollout budget N, caps
  n_i <= n_max, LRU exploration floor (15% of budget).
- Advantage: A = r_tilde - b_tilde_i with the snapshot baseline; token-level
  REINFORCE with constant normalizer; LoRA-only updates; one optimizer step
  per iteration (strictly on-policy).
- Length control: r_tilde = r * (1 - lambda * min(1, len/L_max)) on correct
  rollouts only; lambda by projected dual ascent toward E[len | correct] <= L*.
- Baselines in the same loop: GRPO (fixed G, group mean/std), Dr. GRPO (no
  std), DAPO-style dynamic sampling (rejection + refill waves, discarded
  tokens metered).

## 4. Experimental protocol

- **Model.** Gemma 4 E2B instruction-tuned checkpoint (4.8B raw parameters,
  effective-2B architecture; multimodal towers offloaded; text stack only),
  LoRA rank 32 on all attention/MLP projections (48.3M trainable), bf16.
- **Data.** The frozen suite: BBH (27 tasks), BBEH (23 tasks), USR (10 task
  groups); 12,540 auto-scorable examples. Splits by within-task source index
  (calibration < 25, validation 25-49, test >= 50) exactly as the published
  confirmatory protocol; 16 of 60 tasks (every 4th alphabetically per
  benchmark) are excluded from RL training to measure unseen-task transfer;
  60 calibration prompts exceeding the 3,072-token prompt cap are dropped
  from training. Training pool: 1,040 prompts.
- **Reward.** The repository's frozen scorer (`evaluate_correctness`),
  byte-identical to the deployed evaluation; binary.
- **Budgets.** 48 iterations x 448 rollouts (~[[TBD]]M generated tokens per
  run), temperature 0.9, max 384 new tokens, concise_cot prompt template,
  single user message, no system message. All optimizer hyperparameters
  identical across methods (AdamW, lr 2e-5, warmup 5 iterations, clip 1.0).
- **Evaluation.** Greedy decoding with the frozen per-arm limits
  (direct_answer @ 64 tokens; concise_cot @ 256), the unchanged scorer, and
  single-shot frozen-test evaluation of the validation-selected checkpoint.
  Published reference points on the identical test split: direct_answer
  26.39%, concise_cot_self_rank_k3 35.06% (681 tokens/answer), CBRR bandit
  router 35.41%.
- **Hardware.** One shared RTX 6000 Ada (49GB, ~15GB available besides
  co-tenant services); the constrained-compute regime the method targets.

## 5. Results

[[TBD: filled from runs]]

### 5.1 Frozen-test accuracy (primary)

| Policy | Strategy | BBH | BBEH | USR | Overall | Tokens/answer |
|---|---|---:|---:|---:|---:|---:|
| Base E2B (published, API) | direct_answer | 40.73% | 10.80% | 5.30% | 26.39% | 14.7 |
| Base E2B (published, API) | self-rank k=3 | - | - | - | 35.06% | 681.3 |
| CBRR router (published) | routed | 57.33% | 10.68% | 6.18% | 35.41% | 65.6 |
| Base E2B (this pipeline) | direct_answer | [[TBD]] | | | | |
| Base E2B (this pipeline) | concise_cot | [[TBD]] | | | | |
| GRPO (compute-matched) | direct_answer | [[TBD]] | | | | |
| GRPO (compute-matched) | concise_cot | [[TBD]] | | | | |
| VOLT | direct_answer | [[TBD]] | | | | |
| VOLT | concise_cot | [[TBD]] | | | | |

### 5.2 Training efficiency (validation accuracy vs generated tokens)

[[TBD: probe curves at matched token budgets; tokens-to-match-GRPO-best.]]

### 5.3 Unseen-task generalization

[[TBD: frozen-test accuracy restricted to the 16 held-out tasks vs the 44
trained tasks, per method.]]

### 5.4 Diagnostics

[[TBD: nonzero-advantage fraction (VOLT ~1.0 by construction vs GRPO's
degenerate-group decay), allocation entropy trajectory, posterior calibration
(reliability of b_i vs realized success), dual multiplier trajectory.]]

### 5.5 Ablations

[[TBD as budget allows: uniform-allocation VOLT (SPO-like), flat prior
(no hierarchy), grpo_ds dynamic sampling, length shaping on/off.]]

## 6. Limitations

Single model family and scale (the deployment under study); binary rewards
only (the variance model and allocation exponent are Bernoulli-specific,
though the martingale argument is not); the task-adaptation setting shares
tasks between training and test rows by design of the frozen protocol — the
16-task holdout quantifies but does not eliminate this; compute constraints
limit seeds and ablation breadth; LoRA-only updates.

## 7. Reproducibility

Code in `rl/` (unit-tested math), configs in `experiments/rl/`, per-iteration
telemetry (`metrics.jsonl`), rollout samples, posterior state snapshots, and
eval outputs under `results/rl/`. Seeds fixed (20260709 family). The frozen
protocol, scorer, and split definitions predate this work and are reused
unchanged.
