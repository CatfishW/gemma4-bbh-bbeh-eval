# Top methodologies in the Gemma 4 E2B study

This guide uses **N = 8** representative methods. The repository contains 29
prompt arms plus two learned-adapter tracks, so a single raw ranking would mix
different splits, call budgets, and inference stacks. The selection below keeps
the strongest clean results and the distinct mechanisms needed to understand
the study.

## Results at a glance

The prompt-arm percentages below are matched validation results (E2B / E4B).
The CBRR and adapter rows are untouched E2B test results, because those are the
confirmatory/frozen comparisons for their tracks.

| Method | What changes | Evidence | Main trade-off |
|---|---|---:|---|
| `direct_answer` | Answer-only instruction; no weight change | 26.39% E2B frozen test; 14.68 completion tokens | Cheapest, but no reasoning scaffold |
| `canonical_short` | Normalizes answer format by type | 27.11% / 33.56% matched validation | Reduces parser failures; little extra reasoning |
| `option_elimination` | Privately remove inconsistent choices | 23.96% / 31.34% matched validation | Mainly useful for multiple choice |
| `private_verify` | Solve, then check once privately | 19.73% / 34.23% matched validation | Model- and benchmark-dependent |
| `concise_cot_self_rank_k3` | Three sampled concise solutions plus a selection pass | 35.06% E2B frozen test; 33.96% / 39.13% matched validation | Strong, but about 681 completion tokens |
| CBRR | Select one prompt arm per task from 25 calibration examples | 35.41% E2B frozen test; 65.6 tokens | No weight change; needs stable task IDs and calibration |
| GRPO LoRA | Fixed-group RL fine-tuning | 37.58% E2B frozen test; 162.1 tokens | Many homogeneous groups produce zero advantage |
| **VOLT LoRA** | Variance-aware RL with adaptive rollout allocation | **38.62% E2B frozen test; 127.0 tokens** | Requires training; no allocator at inference |

VOLT is the strongest complete frozen-test result. It beats compute-matched
GRPO by 1.04 percentage points and uses 21.6% fewer completion tokens. The
42.92% reward-routed-v1 number elsewhere in the repository is an offline replay
with partial live validation, so it is reported as exploratory rather than as
the best complete result.

## 1. `direct_answer`: the control condition

The evaluator appends an answer-only instruction: “Return only the final
answer. Do not include reasoning, explanation, or extra text.” The model makes
one ordinary request, and the benchmark parser normalizes the response. This
is the reference for every gain claim: it is fast and format-safe, but gives a
small model no explicit reasoning or verification scaffold.

## 2. `canonical_short`: format control

This prompt explicitly maps the answer to the task's expected type: one option
label for multiple choice, exactly `Yes`/`No` for booleans, digits for numbers,
and the list alone for list questions. It changes the surface form rather than
the model weights or the underlying reasoning. Its improvement over
`direct_answer` is evidence that exact-match formatting is a real bottleneck.

## 3. `option_elimination`: private multiple-choice search

The model eliminates choices that contradict the question before committing to
one remaining option, then emits only the parseable label. The extra internal
structure helps tasks where distractors encode common logical mistakes, but it
does not generalize as well to non-choice tasks.

## 4. `private_verify`: solve, then check

The model produces a candidate internally, performs one silent check, and
returns only the answer. The check is deliberately bounded: it is not a long
visible chain of thought. The E2B/E4B split shows the risk clearly—this method
is model-dependent and can cause unnecessary self-revision on the smaller
model while helping the stronger model.

## 5. `concise_cot_self_rank_k3`: sample, then select

The evaluator samples three concise chain-of-thought answers at temperature
0.7, normalizes their candidate answers, and runs a fourth selection pass that
chooses the most defensible candidate and prints only that answer. The second
pass is the key: it acts as both a verifier and a formatting repair step. It is
the best universal prompt arm, but costs roughly 681 completion tokens per
example.

## 6. CBRR: conservative Bayesian task routing

CBRR keeps model weights fixed. For each task, it evaluates candidate prompt
arms on the first 25 calibration examples, models paired wins/losses against
`direct_answer` with a Beta-Bernoulli posterior, and switches only when the
posterior and net-win thresholds are strong enough. The selected strategy is
then frozen for every later example of that task; there is no per-example
adaptation. This concentrates gains on BBH while preserving a one-call serving
path (35.41% on the frozen E2B test at 65.6 tokens).

## 7. GRPO LoRA: the compute-matched RL baseline

GRPO fine-tunes the E2B model through a rank-32, alpha-64 LoRA adapter. Each of
48 iterations selects 56 prompts and samples eight fixed rollouts per prompt
(448 rollouts total). Exact-match rewards are normalized by each eight-sample
group; all-correct and all-wrong groups have zero relative advantage. Only
5,432 of 21,504 training rollouts supplied a nonzero update signal. The frozen
test score is 37.58% with 162.1 mean completion tokens.

## 8. VOLT LoRA: variance-optimal rollout training

VOLT uses the same base model, data protocol, reward, and LoRA parameterization
as GRPO, but changes how training evidence is allocated and centered:

1. Before each iteration, a discounted hierarchical Beta posterior (discount
   0.92, prior strength 4.0) supplies a frozen success baseline
   `b_i = E[p_i]` and an uncertainty score `s_i = sqrt(E[p_i(1-p_i)])`.
2. A water-filling allocator spends the 448-rollout budget according to that
   score, reserves a 15% least-recently-sampled exploration floor, and caps a
   prompt at eight rollouts.
3. Fresh `concise_cot` completions are scored with the unchanged binary
   exact-match reward. The predictable advantage is `A_i = r_i - b_i`.
4. Token-level REINFORCE updates only the LoRA weights; an AdamW step uses a
   constant completion-length normalizer of 384. The posterior is updated only
   after the iteration, so allocation cannot peek at the current rewards.

The reported run disables optional length shaping. At deployment the adapter
is served as an ordinary one-user-message model: the allocator, posterior, and
reward function are not present. VOLT reaches 3,688/9,550 = 38.62% on the
frozen E2B test with 127.0 mean completion tokens.

## How to choose among them

- Use `direct_answer` for the lowest latency and as the control.
- Use `canonical_short` when parser/format errors dominate.
- Use `option_elimination` for multiple-choice-heavy workloads.
- Use `private_verify` when a stronger model can benefit from a bounded check.
- Use self-ranking when accuracy matters more than generation budget.
- Use CBRR when task IDs and a small labeled calibration slice are available.
- Use GRPO as the standard fixed-group RL comparison.
- Use VOLT when training is allowed and rollout budget is scarce; its main
  contribution is retaining informative updates while reducing generated text.

The companion figures in `figures/` show the method map, the prompt-only
decision flows, and the GRPO/VOLT LoRA training loop.
