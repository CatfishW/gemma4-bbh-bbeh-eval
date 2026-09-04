# CRAFT: Counterfactual Reasoning Allocation and Forked Training

**Status: research implementation, not a trained Gemma result.** This branch adds
`rl_craft/` on top of the existing `rl-volt` branch at
`e0dab8a08af7ab4dc2b95ce790112820b733afab`. It does not replace VOLT, modify its
training code, overwrite its results, change its scorer, or merge the separate
frozen-inference feature branch. Base parameters remain frozen; LoRA learns both
reasoning/content and an explicit stopping decision.

## Research proposal and novelty boundary

The project already allocates rollout counts by prompt (VOLT). CRAFT instead
allocates **credit and computation inside a sampled reasoning family**. Three
coupled mechanisms form the proposed contribution:

1. **Counterfactual stopping supervision without a teacher or critic.** At the
   same sampled prefix, evaluate both `finalize now` and `continue reasoning`.
   Integrate the two-action gate analytically, and differentiate its expected
   outcome rather than asking a model to self-report uncertainty.
2. **Branch-exact, cross-fitted credit.** Assign the shared prefix its mixture
   return once; the gate its probability-weighted action values; and each suffix
   a baseline computed only from *other* independently sampled trajectories.
   The derivative of the gate, prefix and content distribution are all included.
3. **Quality-constrained counterfactual allocation.** A lagged proposal uses
   stop/continue outcome disagreement and within-arm variability per cost. A
   uniform-mixture exploration floor and exact target/proposal weights preserve
   a fixed macro-task objective. Separate task multipliers increase the price of
   accuracy loss when optimizing inference cost.

LoRA, REINFORCE, leave-one-out baselines, Rao–Blackwellization, importance
sampling, and primal–dual optimization are **not new individually**. This is a
specific proposed integration with an explicit stochastic computation graph and
an executable estimator. No claim of worldwide priority or superiority over all
2026 work is made; a current literature search remains necessary before a paper.
Established starting points include [LoRA](https://arxiv.org/abs/2106.09685),
[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300),
[REINFORCE-style LLM optimization](https://arxiv.org/abs/2402.14740), and
[constrained policy optimization](https://arxiv.org/abs/1705.10528).

### Why this is not another VOLT configuration

| Property | Existing VOLT | CRAFT |
|---|---|---|
| Sample unit | Complete response to a prompt | Shared prefix + two counterfactual action subtrees |
| Stopping signal | Optional scalar length shaping | Measured answer-quality difference between stopping and continuing |
| Credit | One sequence advantage broadcast over completion | Separate prefix, gate, and independent suffix coefficients |
| Context for allocation | Historical success variability | Lagged counterfactual utility disagreement and cost |
| Sampling objective | Existing prompt allocation policy | Explicit macro-task target, corrected by exact p/q weights |
| Accuracy protection | Existing reward configuration | Separate per-task, lagged dual multipliers and optional frozen-base floors |
| Deployment | Ordinary adapter generation | One adapter + explicit staged gate; no reward model or labels |

CRAFT can learn to stop earlier on all-correct branches because the cost differs.
It **does not magically obtain correctness information from all-wrong branches**;
its length objective can still prefer shorter failures. Task constraints are a
soft training mechanism, not a proof against collapse. Infeasible floors and
quality/cost weights must be diagnosed rather than hidden.

## Stochastic computation graph

The same LoRA-adapted model implements each node:

```text
Question -> sampled brief notes Z (generated once per family)
                 |
           gate pi(F | Q,Z), pi(R | Q,Z)
            /                      \
  finalize now F               continue R
  K independent answers        K independent extensions + answers
            \                      /
      gold exact-match rewards, explicit inference-cost proxy
                       |
       prefix / gate / suffix policy-gradient contributions
```

During training both branches are sampled (`paired` estimator). During deployment
only one is used. `sampled` is a conventional single-action REINFORCE control that
samples the gate and only its selected branch. A gate is one forward read of two
existing vocabulary logits (`F` and `R`), not a trained classifier head or added
token embedding. Each label must be a single exactly decodable token or startup
fails. All nodes send a single user message through the pinned chat template,
with `enable_thinking=False` explicitly supplied. This is a new staged protocol,
not the repository's native-thinking protocol or its original concise-CoT arm.

The initial notes are bounded by `prefix_tokens`. Continue produces another
bounded note block; both arms then use an answer-only readout. A note block may
hit its budget. A final answer that does not reach the configured model EOS/turn
termination earns zero reward and is logged as truncated. The adapter can learn
shorter notes through normal termination tokens. There is one explicit gate per
response, not an unbounded agent loop.

**Important efficiency boundary:** the prefix is generated once, but readout
prompts are re-prefilled. The reference implementation does not clone/cache a
Gemma KV cache across branches. It does not claim that four short forwards are
always faster than one direct answer. At deployment the controller still makes
multiple local forwards; the benefit must be measured against the same strong
baseline used for the accuracy comparison.

## Objective and exact estimator

For task t, let lambda_t be a nonnegative multiplier, fixed for an iteration.
For outcome Y after action a, define

```text
U_t(Y) = (1 + lambda_t) * correctness(Y) - beta * C(Y) / cost_scale.
```

This is the policy-dependent part of the Lagrangian for maximizing accuracy minus
cost subject to `accuracy_t >= target_t`; the omitted `-lambda_t * target_t` is
constant with respect to policy parameters. Cost is generated tokens plus one
gate-position proxy, optionally augmented by `prefill_price * prefill_tokens`.
It is **not a measurement of GPU time or wall latency**. Prefix/gate cost appears
once in each *hypothetical deployment outcome*, while the logged training budget
counts the prefix once and every actually sampled counterfactual suffix.

For sampled notes Z, write p_a = pi(a | Q,Z), U_aj for K independently sampled
outcomes in arm a, and Qhat_a = mean_j U_aj. Let Vhat = sum_a p_a Qhat_a and b be a
historical prompt baseline frozen before Z was sampled. The raw gradient estimate:

```text
g_prefix = (Vhat - b) * grad log pi(Z | Q)
g_gate   = sum_a p_a * (Qhat_a - Vhat) * grad log pi(a | Q,Z)
g_suffix = sum_a p_a/K * sum_j (U_aj - b_aj) * grad log pi(Y_aj | Q,Z,a)
```

For a continue outcome, `log pi(Y)` includes BOTH its extension and its answer
readout. The default cross-fitted baseline b_aj is the mean utility of all 2K-1
other trajectories, excluding the complete trajectory receiving the advantage.
Those samples are independent of Y_aj conditional on the shared Z. It remains a
valid baseline even when the other action has a different return distribution.
The `history` ablation uses b instead.

**Derivation.** Factor the joint distribution as
`pi(Z|Q) * p(a|Q,Z) * pi(Y|Q,Z,a)` and differentiate each factor. The prefix term
uses an unbiased conditional mixture return. Enumerating the gate gives its
conditional expectation exactly given the sampled Q estimates. For each suffix,
`E[b_aj * grad log pi(Y_aj) | Q,Z,a] = 0` because the baseline excludes that
trajectory. Centering the gate by Vhat does not change its gradient because
`sum_a p_a grad log p_a = 0` for every fixed Z. Sharing neural parameters among
all stages does not change this factorization.

All numerical coefficients are detached before score-function backpropagation.
Do not differentiate p_a inside those coefficients and also multiply by its log
probability: that double-counts part of the derivative. Do not assign all leaf
returns to the root independently: that overweights shared actions. Do not leave
a continuation's own answer in its baseline while claiming independence.

### On-policy assumptions actually enforced

One optimizer step follows a completely collected batch. Dropout is disabled for
both collection and scoring. Sampling uses fresh generation configuration with
full softmax support (`top_p=1`, `top_k=0`, no repetition penalty), and the exact
same temperature appears in teacher-forced log probabilities. Startup compares
optimized head logits and generation scores against the model's own forward;
a mismatch aborts. There are no stale/replayed responses or PPO multi-epoch reuse.

The mathematical identity applies to the raw, unclipped score estimator. AdamW,
gradient clipping, finite precision, limited samples, evolving multipliers and
an approximate task floor do not turn it into a finite-sample convergence or
quality guarantee. `loss_normalizer` is a fixed experiment-wide scalar, not a
per-response random length denominator.

## Adaptive allocation and accuracy constraints

The declared example target is macro-task uniform:

```text
P(i) = 1 / (number_of_tasks * examples_in_task(i)).
```

A lagged record tracks counterfactual leverage, cost, and mean return. The paired
leverage observation is

```text
p_F*p_R*(Qhat_R - Qhat_F)^2 + probability-weighted within-arm utility variance.
```

The proposal is `q(i) = epsilon*P(i) + (1-epsilon)*normalize(P(i)*sqrt(v_i/c_i))`.
This is a heuristic information/cost proposal, **not** a theorem of optimal total
gradient allocation. Draw with replacement and weight each entire tree by
`P(i)/q(i)` without clipping or self-normalization. Weights are bounded by
`1/epsilon`. Target/proposal weights and all draws are recorded before current
outcomes; statistics update only after the optimizer step. The fixed objective is
therefore not silently changed into a curriculum weighted toward easy tasks.

Dual updates use a Horvitz–Thompson estimate of each task's constraint violation,
normalized by the fixed target task mass (1/T), not the random number of observed
examples of that task. Multipliers are projected to `[0, dual_max]`. Tasks absent
from an iteration are not updated. The fixed-config floor is explicitly a user
hypothesis; the recommended `calibrate-targets` command instead runs the initial
untrained adapter (= base) in always-continue mode on **calibration rows only**,
then freezes per-task Wilson lower bounds minus a tolerance. These bounds are
conservative descriptive references, not deployment guarantees. The target file
is bound to model bytes, source code, input bytes and behavior configuration.

HT training estimates may temporarily exceed [0,1] in a finite reweighted batch.
They are labeled `ht_train_accuracy_estimate`, not validation accuracy. Do not
clip them and then claim an unbiased constraint estimator.

## Install / environment

The pure math/data modules use the standard library. Training and all tests use
PyTorch. Live E2B training additionally needs the **same already working
Transformers and PEFT environment as `rl-volt`** and the existing local BF16
checkpoint. This code reuses `rl.modeling`'s multimodal-safe loader and exact
text-stack LoRA target selection. It makes no architecture name guesses beyond
that existing integration and performs no checkpoint downloads. The supplied
implementation was tested here with CPU PyTorch 2.10.0; actual Gemma/PEFT execution
is not asserted until run on the user's configured model environment.

The dedicated CI workflow installs CPU torch 2.5.1 on Python 3.11 and runs the
entire new suite, including gradient tests. The existing test workflow is
unchanged; optional torch tests skip where torch is absent.

## Run offline now

```bash
python -m unittest discover -s tests -p 'test_craft_*.py' -v
python -m rl_craft smoke --output /tmp/craft-smoke-v1 --iterations 12
python -m rl_craft smoke --output /tmp/craft-smoke-v1 --iterations 12 --resume
```

The smoke test runs a real torch optimizer on a tiny, deterministic-base LoRA
policy and checks frozen-base equality. It is an execution test, **not Gemma, not
a BBH score, and not evidence of production speed or general reasoning quality**.
Training targets are synthetic bits. Do not quote its training accuracy as a
benchmark result.

## Prepare data, calibrate floors, train

From the repository root in the existing RL environment:

```bash
python -m rl_craft prepare \
  --datasets-root /data/benwulab/gemma4-eval/datasets \
  --output /tmp/craft-calibration.jsonl

python -m rl_craft calibrate-targets \
  --data /tmp/craft-calibration.jsonl \
  --model-path /data/models/gemma-4-E2B-it \
  --config experiments/craft/e2b.json \
  --tolerance 0.02 --device cuda:0 \
  --output /data/benwulab/gemma4-rl/craft-reference-v1

python -m rl_craft train \
  --data /tmp/craft-calibration.jsonl \
  --model-path /data/models/gemma-4-E2B-it \
  --targets /data/benwulab/gemma4-rl/craft-reference-v1/targets.json \
  --config experiments/craft/e2b.json --device cuda:0 \
  --study-id craft-development-v1 \
  --output /data/benwulab/gemma4-rl/craft-v1
```

`prepare` requests only the first 25 examples per task from the existing loaders,
records clean local dataset Git revisions, and excludes every fourth sorted task
within each benchmark, as in `rl.protocol`. It does not read later examples to
train or to select rollout policies. Output data and a sidecar manifest are
immutable. External development JSONL is also supported via `--input`, with each
line shaped as follows:

```json
{"key":"suite/task/0","task":"suite/task","question":"...","target":"...","index":0}
```

Explicit inputs with any index >=25 are rejected, not silently filtered. Keep
related puzzle variants together when constructing a new source. `--holdout-stride
0` explicitly disables task holdout; it is not the default. Existing benchmark
indices are not made pristine by this command: their historical exposure still
matters. The new method must use a newly declared holdout for confirmatory claims.

`calibrate-targets` is an additional baseline measurement, and its computation
must be included in offline training/calibration cost. Omitting `--targets` uses
the explicit fixed-config floor and records that fact; it does not invent a base
accuracy estimate. No validation/test label enters the trainer, scheduler, or
target calibration.

## Checkpoints, resuming and evaluating

Every completed optimizer step writes a new directory, atomically renamed only
after the adapter, optimizer, RNG state, scheduler, dual multipliers, traces,
metrics and integrity hashes are written. `latest` points only to a completed
checkpoint. Checkpoint zero permits recovery before the first update. Resume
requires exact model/data/code/config/dependency-version matches and verifies
checkpoint file hashes before restricted `torch.load(weights_only=True)`.
Failed temporary directories are preserved for inspection; do not silently erase
or treat them as completed checkpoints. Each checkpoint retains cumulative
traces, so disk usage can be substantial. This correctness-first reference does
not implement checkpoint pruning, distributed training, or asynchronous rollout.

```bash
# Same command and configuration, with --resume:
python -m rl_craft train \
  --data /tmp/craft-calibration.jsonl \
  --model-path /data/models/gemma-4-E2B-it \
  --targets /data/benwulab/gemma4-rl/craft-reference-v1/targets.json \
  --config experiments/craft/e2b.json --device cuda:0 \
  --study-id craft-development-v1 \
  --output /data/benwulab/gemma4-rl/craft-v1 --resume

python -m rl_craft evaluate \
  --checkpoint /data/benwulab/gemma4-rl/craft-v1/checkpoint-00048 \
  --model-path /data/models/gemma-4-E2B-it \
  --data /path/to/declared-validation.jsonl \
  --split validation --study-id craft-validation-v1 \
  --gate sample --device cuda:0 --output /tmp/craft-validation-v1
```

Evaluation JSONL uses the same Example fields. Validation rows must all have
indices 25–49; test rows >=50 require `--allow-test`. `external` makes no claim
about legacy split membership. Each selected row remains in the denominator;
wrong-split files abort. Model errors/overlong contexts abort rather than skip
hard examples. Save partial predictions as incomplete; do not publish a partial
length-sorted run as full accuracy. The evaluator uses the existing scorer on the
final channel and reports micro and macro-task accuracy, per-task counts, token
counts, stop frequency, latency and answer truncation behavior.

The trained gate is stochastic (`--gate sample`). Greedy, always-stop, and
always-continue are explicit separate deployment controls. Changing the gate
mode changes the policy; do not silently evaluate greedy and describe it as the
exact training-distribution result. Stage generation uses the trained sampling
temperature for all four gate modes.

## Ablations and honest cost matching

The supplied configuration files provide:

| File | Change |
|---|---|
| `e2b.json` | Full CRAFT |
| `no_allocation.json` | Macro-task-uniform sampling, no adaptive proposal |
| `no_crossfit.json` | Historical suffix baseline, same paired graph |
| `no_dual.json` | Fixed accuracy-minus-cost reward, multipliers disabled |
| `sampled_action.json` | One sampled action/outcome, ordinary policy gradient |

Files are controls, not already measured results or automatically compute-matched
experiments. Compare identical checkpoints, data, reward definitions, optimizer
settings, and the same **measured total sampled tokens, prefills, GPU time and
wall time**. Equal optimizer steps are not equal compute. The hard generated-token
budget conservatively reserves room for each complete tree; it may leave unused
budget when a worst-case tree cannot fit. Prefix reuse saves generated actions,
not necessarily proportional GPU work. Checkpoint/reference-calibration costs
are separately visible, not free.

Also compare original GRPO/VOLT, plus a staged always-continue LoRA baseline, a
sampled-action RL baseline, and tuned native-thinking budgets. Report a Pareto
frontier against the same baseline for both accuracy and latency. Gate-aware
speed is a system-level property, not the speed of a bare merged adapter.

## Implementation validation

`tests/test_craft_math.py` exhaustively enumerates a finite two-stage MDP and
compares the expected implementation gradient with the exact differentiable
expected return, including prefix dependence, gate dependence and action-specific
continuation dependence. Paired K=1, paired K=2, historical-baseline and
sampled-action estimators agree within 1e-12 in float64. Another test checks the
importance-corrected target gradient under a nonuniform proposal.

Training tests check real LoRA-only updates, immutable base parameters, exact CPU
resume, budget accounting, frozen allocation snapshots, failure-before-update,
truncated answer rewards, checkpoint integrity, and chunked temperature-correct
large-vocabulary loss/gradient equivalence on small tensor fixtures. These do not
prove compatibility with every Transformers revision or demonstrate a Gemma gain.

## Known limits / next scientific tests

This first implementation uses a single fork point and discrete F/R gate,
sequential microbatch training and a cost proxy. It does not implement learned
per-token exit depths, adaptive LoRA rank, a verifier, replay, a teacher, full-tree
KV sharing, or tensor-parallel training. The head is checkpointed to avoid
retaining sequence-by-vocabulary logits, but the backbone still requires enough
activation memory for one segment. BF16 execution must be tested on the actual
48 GB GPU stack; no successful GPU run is claimed by CPU tests.

Prioritize whether (a) early finalization recovers correct answers hidden by long
or truncated traces, (b) conditional branch credit beats ordinary sampled-action
RL under a measured budget, (c) quality multipliers reduce per-task regressions,
and (d) gains survive unseen task families and longer, changed-premise questions.
If these ablations are negative, report them; do not equate the presence of a new
estimator with significant empirical improvement.
