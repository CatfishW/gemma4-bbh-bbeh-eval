# Gemma 4 E2B/E4B Reasoning Evaluation

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Language-English-0969DA?style=for-the-badge" alt="English"></a>
  <a href="./README.zh-CN.md"><img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-DE2910?style=for-the-badge" alt="简体中文"></a>
</p>

This repository measures — and then improves — how well two small deployed
language models (Gemma 4 E2B and E4B) solve hard reasoning benchmarks, without
ever touching the benchmark's test questions during tuning.

It contains three things:

1. **An evaluation harness** (`eval_benchmarks.py`) that asks a model
   benchmark questions one at a time and scores the answers by exact match.
2. **A prompt-strategy study**: 26 different ways of phrasing the request
   ("prompt strategies"), measured head-to-head, plus a calibrated per-task
   router (CBRR) that picks the best strategy for each task. No model weights
   change.
3. **A reinforcement-learning method (VOLT)** that does change model weights:
   LoRA fine-tuning driven by a novel budget-aware RL algorithm, trained only
   on the small calibration split of the same protocol. Method and theory live
   in [paper/volt/](paper/volt/).

## The benchmarks

| Benchmark | What it tests | Examples used |
|---|---|---:|
| **BBH** (BIG-Bench Hard, 27 tasks) | Logic, dates, tables, tracking objects, word sorting | 6,511 |
| **BBEH** (BIG-Bench Extra Hard, 23 tasks) | Much harder successors of BBH tasks | 4,520 |
| **USR** (Unpuzzles & Simple Reasoning) | Puzzles, "unpuzzled" trivial variants, simple reasoning | 1,509 |

12,540 examples total. Every example has a known correct answer, so scoring is
automatic: the model's reply is normalized (case, punctuation, option-label
forms like `(A)` vs `A`, numbers, LaTeX) and compared with the reference.

**The split rule (frozen before any tuning):** inside every task, examples are
numbered. Numbers 0-24 are *calibration* (may be used for tuning), 25-49 are
*validation* (may be used for selecting between tuned variants), and 50+ are
the *test* split (9,550 examples) that nothing is ever tuned on. Prompt-only
headline numbers below are from that untouched test split; results explicitly
labeled *validation* are not test claims.

## The models

- `SubTokenLLM-E2B` — Gemma 4 E2B instruction-tuned ("effective 2B" parameters).
- `SubTokenLLM` — Gemma 4 E4B ("effective 4B").

Both are served behind an OpenAI-compatible API. Every request contains exactly
one `user` message and no system prompt, so results reflect the model, not
hidden instructions:

```python
from openai import OpenAI

client = OpenAI(base_url="https://llm.agaii.org/llm/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="SubTokenLLM-E2B",
    messages=[{"role": "user", "content": "Reply with exactly: online"}],
    max_completion_tokens=16,
    temperature=0,
)
```

## Headline results (E2B, frozen test split, 9,550 examples)

| Approach | Weights changed? | Calls / question | Accuracy | Avg completion tokens | Mean API time / question |
|---|---|---:|---:|---:|---:|
| Ask for the answer directly (`direct_answer`) | No | 1 | 26.39% | 14.7 | 0.585 s |
| Best universal prompt (`concise_cot_self_rank_k3`) | No | 4 | 35.06% | 681.3 | 11.228 s |
| CBRR per-task prompt router | No | 1 | **35.41%** | 65.6 | 1.287 s |
| VOLT RL fine-tuning (this branch) | Yes (LoRA) | 1 | **Pending** | **Pending** | **Pending** |

How to read this: just asking better questions moves this small model from
26% to 35% — and the router gets the same accuracy as the best prompt while
generating one tenth as much text. The gain is statistically solid (McNemar
p ≈ 1e-132, bootstrap +8.4 to +9.6 points) and concentrated in BBH; BBEH
barely moves with any prompt. On the larger E4B model the router reaches
41.6% on the matched validation set.

VOLT and the compute-matched GRPO baseline have finished training and full
validation. Their frozen-test results are still pending, so no test accuracy is
reported for either adapter yet. All other prompt-only tables (per-dataset
breakdowns, both models, all 29 arms, robustness checks) are in
[docs/DETAILED_RESULTS.md](docs/DETAILED_RESULTS.md).

## Current best methods: what to use

There is no single winner for every constraint. CBRR is the best completed
frozen-test result and the strongest efficiency/accuracy tradeoff when task IDs
are available. Self-ranking is the strongest universal prompt-only method when
they are not. VOLT is the current full-validation winner among the trained
adapters, but its untouched-test result remains pending.

| Goal | Recommended method | Why | What it requires |
|---|---|---|---|
| Lowest cost and latency | `direct_answer`; also test `canonical_short` | One short deterministic call | Nothing beyond the model API |
| Best prompt-only efficiency | **CBRR** | One call, 35.41% frozen-test accuracy, about one tenth of self-ranking's tokens | Stable task ID and 25 labeled calibration rows per task |
| Best universal prompt-only accuracy | **Self-ranking, k=3** | No task metadata or weight update; 35.06% frozen-test accuracy | Four calls and a much larger token budget |
| Best current tuned-model validation accuracy | **VOLT LoRA** | 36.04% on full validation with fewer generated tokens than GRPO | The matching base checkpoint, training data, and LoRA deployment |
| Scientific RL baseline | **GRPO** | Standard compute-matched comparison for measuring VOLT's contribution | Fixed groups of multiple rollouts during training |

### Keep the two result tracks separate

The prompt-only table above is the registered API pipeline on all 9,550 frozen
test examples. Its time column is end-to-end request time in that API run, so
it fairly exposes self-ranking's three generation calls plus one selection
call. CBRR performs routing offline and still makes only one model call per
question.

The weight-tuning track currently has a completed 1,490-example full-validation
comparison using the training-matched `concise_cot` prompt and probe-selected
checkpoints:

| Model on full validation | Correct | Accuracy | Avg completion tokens | Observed eval wall time |
|---|---:|---:|---:|---:|
| Base E2B | 265 / 1,490 | 17.79% | 226.9 | ~633 s (0.425 s/example) |
| Compute-matched GRPO LoRA | 498 / 1,490 | 33.42% | 163.6 | ~960 s (0.644 s/example) |
| **VOLT LoRA** | **537 / 1,490** | **36.04%** | **125.6** | **~943 s (0.633 s/example)** |

These validation rows establish the comparison among base, GRPO, and VOLT;
they do **not** establish that VOLT beats CBRR or self-ranking, because those
headline results use a different split and serving pipeline.

On these paired validation predictions, VOLT beats GRPO by 39 answers, or
2.62 percentage points (121 VOLT-only wins, 82 GRPO-only wins; McNemar
`p = 0.0075`; task-stratified bootstrap 95% interval `+0.87` to `+4.36`
points). It uses 23.2% fewer completion tokens than GRPO. Against the base it
gains 18.26 points and uses 44.6% fewer completion tokens.

The wall-clock values are local, batched, unmerged-PEFT measurements, not the
same latency protocol as the API table. In this run VOLT was slightly faster
than GRPO, but the two unmerged adapters were roughly 49–52% slower than the
bare base in wall-clock throughput despite shorter answers. Merge the LoRA
into the base weights and benchmark the target serving stack before making a
production latency claim.

Training efficiency shows the intended VOLT advantage more directly. Both
runs generated 21,504 rollouts. GRPO produced only 5,432 rollouts with nonzero
group-relative advantage, while VOLT retained nonzero signal for all 21,504
(about 4x as many informative rollouts). VOLT generated 4,655,325 tokens versus
GRPO's 5,593,771, a 16.8% reduction. Observed active training time was roughly
4 h 38 min for VOLT and 4 h 49 min for GRPO, but shared-GPU OOM recovery makes
those wall times descriptive rather than a controlled speed benchmark.

### Worked input/output examples

The small logic question below is synthetic. It demonstrates the exact request
and response shapes; it is not a cherry-picked benchmark prediction and is not
used in any reported score.

> Every red object is square. Object K is red. Is object K square?

#### 1. Direct answer: cheapest one-call baseline

Input sent to the model:

```text
Every red object is square. Object K is red. Is object K square?

Return only the final answer. Do not include reasoning, explanation, or extra text.
```

Expected response shape:

```text
Yes
```

`canonical_short` is the safer variant when tasks mix booleans, option labels,
numbers, and lists: it explicitly tells the model which canonical output shape
to use. Both approaches are one request and require no calibration.

#### 2. Self-ranking: generate three, then let the model select

Each of three generation calls receives the `concise_cot` form:

```text
Every red object is square. Object K is red. Is object K square?

Think briefly and solve the problem. Keep the reasoning concise.
End with exactly one line in this format: The final answer is: <answer>
```

Illustrative candidate responses:

```text
Candidate 1: K is red and every red object is square.
The final answer is: Yes

Candidate 2: The rule does not name K directly.
The final answer is: No

Candidate 3: Applying red -> square to K gives square(K).
The final answer is: Yes
```

A fourth, deterministic call receives the original question and all raw
candidates:

```text
Question:
Every red object is square. Object K is red. Is object K square?

Candidate 1:
K is red and every red object is square.
The final answer is: Yes

Candidate 2:
The rule does not name K directly.
The final answer is: No

Candidate 3:
Applying red -> square to K gives square(K).
The final answer is: Yes

Compare the candidates against the exact question and every decisive constraint.
Do not vote by wording or length. Select or correct the answer that is best supported.
Return only the final answer, with no explanation.
```

Selector response:

```text
Yes
```

This is not majority voting: the selector may correct all three candidates.
That verification-and-format-repair pass explains the accuracy gain, while the
four calls and 681-token test average explain the latency cost.

#### 3. CBRR: calibrate once, route by task, call once

CBRR's input during calibration is `(task ID, strategy, binary correctness)`
for each candidate strategy on 25 labeled examples. Its output is a frozen JSON
policy. This excerpt is from the checked-in E2B policy:

```json
{
  "default_strategy": "direct_answer",
  "task_strategies": {
    "bbh/boolean_expressions": "concise_cot",
    "bbh/causal_judgement": "canonical_short"
  }
}
```

At inference, the application supplies a task key and question. For a
`bbh/boolean_expressions` request such as the synthetic example below, the
router looks up `concise_cot` and constructs one ordinary model request:

```text
Is the Boolean expression `True and not False` true?

Think briefly and solve the problem. Keep the reasoning concise.
End with exactly one line in this format: The final answer is: <answer>
```

Illustrative model output:

```text
not False is True, so the conjunction is True.
The final answer is: True
```

The task ID controls prompt construction but is not silently inserted into the
model prompt. There is no per-question search, no answer-dependent routing,
and no extra model call. Unknown tasks fall back to `direct_answer`.

#### 4. GRPO: the compute-matched RL baseline

**GRPO** means **Group Relative Policy Optimization**. For each training
prompt it samples a fixed group of responses, scores each response with the
binary exact-match reward, and normalizes rewards relative to that same group.
It therefore avoids training a separate value model.

For an illustrative four-rollout group with rewards `[1, 1, 0, 0]`, the mean
is `0.5`, the standard deviation is `0.5`, and the normalized advantages are
`[+1, +1, -1, -1]`. Those values train the LoRA. But rewards
`[1, 1, 1, 1]` or `[0, 0, 0, 0]` produce zero advantage for every response:
the model spent tokens, yet that group gives no policy-gradient direction.
The actual baseline uses groups of eight and the same total rollout budget as
VOLT.

At deployment, GRPO machinery is gone. The input is just the selected prompt
template and the output is one normal model response from the LoRA adapter.

#### 5. VOLT: history-based baseline and adaptive rollout allocation

VOLT receives the same training examples and binary rewards as GRPO, but its
state from earlier iterations changes both training outputs:

1. a frozen posterior produces `baseline_i` and an allocation score before
   any current response is sampled;
2. a deterministic allocator outputs 1–8 rollouts per selected prompt;
3. the scorer outputs reward `r` for each completion;
4. the trainer outputs the advantage `r - baseline_i` and updates only LoRA.

For example, suppose earlier results give this prompt a frozen baseline of
`0.2` and the allocator buys one rollout:

```text
training input  -> question + concise_cot instruction
model output    -> "The final answer is: Yes"
verifier output -> reward = 1
VOLT output     -> advantage = 1 - 0.2 = +0.8
optimizer       -> reinforce the sampled completion in the LoRA weights
```

If the answer were wrong, its advantage would be `-0.2`. Unlike GRPO, either
single outcome remains usable because the baseline was fixed from history,
not estimated from the current group. Prompts near posterior success
probability `0.5` receive more rollouts; almost-solved and almost-impossible
prompts receive fewer, with a 15% exploration floor preventing starvation.

Serving the resulting adapter is deliberately ordinary:

```text
application input -> one user prompt
base + VOLT LoRA  -> one generated response
application output -> parsed final answer
```

There is no posterior tracker, allocator, reward function, or multi-sample vote
at inference time. VOLT changes how the LoRA is trained, not the serving API.

### Run the methods

Use the common evaluator command in [Reproducing the evaluation](#reproducing-the-evaluation)
and choose one of these method-specific flag sets:

```bash
# Cheapest baseline: one deterministic call
--prompt-strategy direct_answer --temperature 0

# Universal self-ranking: 3 candidates + 1 selector
--prompt-strategy concise_cot --self-consistency-k 3 \
  --response-selection self_rank --temperature 0.7 \
  --max-tokens 256 --selection-max-tokens 64

# CBRR: one routed call; missing task IDs use the policy's default
--prompt-policy results/e2b-confirmatory-20260709_231405/selection/cbrr_policy.json \
  --temperature 0
```

Train the two adapters and run checkpoint selection/final evaluation with the
commands in the [VOLT section](#volt-variance-optimal-reinforcement-learning-under-a-rollout-budget).

### Can the LoRA be used outside these datasets?

Yes, technically: the VOLT artifact is a normal PEFT LoRA adapter, not a
benchmark-only lookup table. It can answer arbitrary application prompts when
loaded on the **exact compatible E2B base checkpoint and tokenizer**, and it
can be enabled, disabled, or merged into the base weights. It cannot be
attached directly to a different architecture or unrelated base revision.

Evidence for transfer is encouraging but limited. Across 16 whole tasks held
out from RL training (400 validation examples), base E2B scored 22.00%, GRPO
30.75%, and VOLT 33.25%. VOLT's +2.50-point lead over GRPO was not significant
on that subset (`p = 0.237`; bootstrap interval `-0.75` to `+6.00`), so this is
not yet proof of broad chat, coding, or domain-specific improvement. Evaluate
the adapter on the target application's prompts, safety constraints, output
format, and latency before deployment.

Prompt methods and weight tuning can also be combined, but a router calibrated
on the base model may become stale after LoRA tuning. Re-run CBRR calibration
against the tuned checkpoint instead of assuming the old per-task choices
remain optimal. For production, a standard PEFT load/merge looks like:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "<the exact E2B base checkpoint used for training>"
adapter_dir = "<selected-checkpoint>/adapter"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_dir)
model = model.merge_and_unload()  # optional; benchmark before/after merging
```

## The 26 prompt strategies, explained

A prompt strategy is a fixed instruction appended to the benchmark question
(the question always comes first, then the instruction). Percentages in
parentheses are overall accuracy on the matched validation split for
(E2B / E4B) — baseline `direct_answer` is (25.0% / 32.6%). The full per-dataset
matrix is in [docs/DETAILED_RESULTS.md](docs/DETAILED_RESULTS.md#3-all-29-universal-arms-by-model-and-dataset-matched-validation).

### Baselines

- **`raw`** — sends the dataset question with no instruction at all. The model
  tends to chat, explain, or hedge, which fails exact-match scoring. This is
  the floor (2.0% / 2.2%), and it is why every other strategy exists.
- **`direct_answer`** — "Return only the final answer. Do not include
  reasoning, explanation, or extra text." Cheap (about 15 tokens per answer)
  and surprisingly strong (25.0% / 32.6%). The reference point for everything
  else.

### Controlling the answer format

These change *how* the answer is written, not how the model thinks.

- **`canonical_short`** — spells out normalization rules: one option label for
  multiple choice, exactly `Yes`/`No` for booleans, digits for numbers, the
  list alone for lists. Best pure-formatting strategy (27.1% / 33.6%); it beat
  `direct_answer` on both models just by reducing format mismatches.
- **`native_format`** — "answer in exactly the format the question requests"
  (some items ask for `<answer>` tags or a fixed sentence). (22.8% / 31.5%)
- **`answer_type_router`** — first privately decide the answer *type* (label,
  boolean, number, phrase, list, tag), then output the most parseable form of
  that type. (22.0% / 28.1%)
- **`strict_json`** — answer as a JSON object `{"answer": ...}`. Brittle:
  wrappers get malformed or truncated, and scoring then fails even when the
  content is right (8.5% / 12.6%). A cautionary result for structured-output
  fans; the JSON failure modes are audited in
  [format_audit.md](paper/e2b-e4b-study/format-audit/format_audit.md).

### Think privately, answer only

These ask the model to reason or verify *silently* and print only the final
answer, keeping outputs short while (hopefully) improving the decision.

- **`careful_direct`** — "read carefully, including every condition and
  option label, then answer only." (19.7% / 31.7%)
- **`private_verify`** — solve, check once privately, then answer. Strongly
  model-dependent: it *hurt* E2B and *helped* E4B (19.7% / 34.2%), matching
  the literature finding that small models need external verifiers to
  self-correct reliably.
- **`selective_verify`** — keep the first answer unless a targeted check finds
  a concrete contradiction (missed negation, ignored constraint, arithmetic
  or label error). Designed to prevent needless self-revision. (22.6% / 31.5%)
- **`compare_then_commit`** — privately identify the two most plausible
  answers, compare both against the exact constraints, reject the weaker.
  Middling as a universal prompt (22.4% / 32.7%) but the single biggest
  contributor when *routed* to the tasks where it wins (+214 correct answers
  in the live routed run).
- **`fast_slow_gate`** — answer directly; only verify if genuinely uncertain.
  The gating language confused small models badly (8.5% / 19.1%).
- **`constraint_guard`** — extract the decisive constraints, derive a
  candidate, test it against each constraint once. (20.9% / 32.9%)
- **`negation_label_guard`** — pay special attention to NOT/EXCEPT/quantifiers
  and map content to the option label as a final step. (12.6% / 20.1%)
- **`draft_verify`** — very short private symbolic draft, then one check.
  (10.3% / 27.6%)
- **`option_elimination`** — for multiple choice, eliminate wrong options
  privately before deciding. (24.0% / 31.3%)
- **`condition_reconstruction`** — derive a candidate, hide the condition most
  able to falsify it, reconstruct that condition from the candidate, and
  compare with the actual text. (20.7% / 30.7%)
- **`counterexample_guard`** — try one targeted counterexample against the
  candidate before committing. (22.0% / 32.5%)
- **`rank_two_paths`** — build two genuinely different solution paths
  privately, rank them, use the stronger. (21.9% / 32.2%)

### Show brief reasoning, then a delimited answer

These allow visible reasoning but require a final line
`The final answer is: <answer>` that the scorer can extract.

- **`concise_cot`** — "think briefly," then the delimiter line. Excellent on
  BBH (35.4% / 46.7% there) but weak overall (17.4% / 24.3%) because on BBEH
  and USR the model often runs out of tokens or buries the answer.
- **`chain_of_draft`** — terse scratch notes instead of prose sentences, then
  the delimiter. Same profile, slightly better (17.7% / 28.7%).
- **`premise_conclusion`** — list the key premises, derive the conclusion.
  (16.5% / 23.8%)
- **`step_back`** — first state the general rule, then apply it. Small models
  spend the token budget philosophizing (4.1% / 8.9%).
- **`plan_and_solve`** — write a short plan, then solve. (4.1% / 6.0%)
- **`plan_and_solve_plus`** — detailed plan plus variable extraction plus
  verification. The longest template and the worst performer (1.3% / 2.1%):
  it reliably exhausts the token budget before answering.
- **`least_to_most`** — decompose into subproblems, solve simplest-to-hardest.
  (1.8% / 3.8%)
- **`symbolic_proof`** — translate to compact symbols or a proof sketch, then
  solve. (1.8% / 2.2%)

The pattern across this family: for small models under exact-match scoring,
verbose reasoning templates only pay off when a reliable extraction/selection
step follows. Which leads to:

### Sampling on top of a strategy (arms, not strategies)

Two inference-scaling variants wrap `concise_cot`:

- **Self-consistency (`concise_cot_sc_k3`)** — sample 3 answers at temperature
  0.7, take the majority-normalized answer. Barely helps (19.3% / 25.3%):
  majority voting cannot fix formatting failures.
- **Self-ranking (`concise_cot_self_rank_k3`)** — sample 3 answers, then ask
  the model to pick the best candidate and output only that answer. The best
  universal arm (34.0% / 39.1%): the selection pass both verifies and fixes
  formatting. Cost: about 681 tokens per answer.

## The CBRR prompt router (no weight changes)

Different tasks favor different strategies, so a single universal prompt
leaves accuracy on the table. CBRR ("conservative Bayesian reward router")
uses each task's 25 calibration examples to decide, per task, whether any
strategy beats `direct_answer` there — using a Beta-Bernoulli posterior on
paired wins/losses, switching only when the evidence is clear — and then
applies that one fixed strategy to every later example of the task. It never
selects per example.

Result on the frozen test split: 35.41% vs 26.39% for direct answering, at 66
tokens per answer (the best universal arm needs 681). Fit it yourself with
`scripts/calibrate_prompt_policy.py`; the evaluator accepts the resulting
policy file via `--prompt-policy`, and records the arm used for every
prediction. Protocol, amendments, and analysis:
[docs/E2B_CONFIRMATORY_PROTOCOL.md](docs/E2B_CONFIRMATORY_PROTOCOL.md),
[docs/PROMPT_OPTIMIZATION_RESEARCH.md](docs/PROMPT_OPTIMIZATION_RESEARCH.md).

## VOLT: variance-optimal reinforcement learning under a rollout budget

Prompt routing tops out near 35% because it never changes model weights. The
`rl/` package instead fine-tunes E2B through LoRA with **VOLT** — **V**ariance-
**O**ptimal a**L**location of **T**okens — an RL-with-verifiable-rewards (RLVR)
method designed for little training data, binary correctness rewards, one
shared GPU, and a fixed generation budget.

### Why fixed GRPO groups waste tokens

GRPO gives every selected prompt a fixed group of sampled solutions and
computes advantages from that group's reward mean and standard deviation. With
binary rewards, an eight-answer group that is all correct or all wrong has
zero advantage everywhere. Its generated tokens cannot update the policy.

That is common in this suite: many BBH prompts are nearly solved, while many
BBEH prompts are nearly impossible. Only prompts near the model's current
learning frontier reliably produce mixed groups.

| Property | GRPO | VOLT |
|---|---|---|
| Rollouts per selected prompt | Fixed at 8 | Adaptive, 1–8 |
| Advantage baseline | Current group mean | Frozen historical posterior mean |
| One-rollout update | Zero or undefined | Valid |
| All-correct/all-wrong samples | Zero advantage | Usually nonzero advantage |
| Uses historical difficulty | No | Yes, with task-level pooling |
| Exploration | Random prompt rotation | 15% least-recently-sampled floor |

### One posterior drives the method

For each training prompt `i`, VOLT tracks its current success probability
`p_i = P(correct | prompt_i)` with discounted Beta evidence. The prompt borrows
a prior mean from other examples in the same benchmark task:

```text
task_mean_i = (task_wins + 1) / (task_wins + task_losses + 2)
alpha_i     = discounted_prompt_wins   + m * task_mean_i
beta_i      = discounted_prompt_losses + m * (1 - task_mean_i)
```

The current configuration uses prior mass `m = 4` and discounts old evidence
by `gamma = 0.92` every iteration so the tracker follows a changing policy.
Before generating anything in iteration `k`, VOLT freezes a posterior snapshot.
That snapshot supplies two active quantities:

```text
baseline_i = E[p_i] = alpha_i / (alpha_i + beta_i)

score_i    = sqrt(E[p_i(1-p_i)])
           = sqrt(alpha_i * beta_i /
                  ((alpha_i + beta_i) * (alpha_i + beta_i + 1)))
```

The baseline estimates expected correctness. The score estimates where binary
reward variation—and therefore usable policy-gradient signal—is concentrated.
It is largest around `p = 0.5` and shrinks toward zero for almost-always-correct
or almost-always-wrong prompts.

### Variance-optimal rollout allocation

Suppose prompt `i` receives `n_i` rollouts, each costing about `l_i` tokens,
and its gradient estimator has variance `v_i`. Minimizing total estimator
variance under a generation budget gives the square-root allocation

```text
n_i ∝ sqrt(v_i / l_i).
```

VOLT models policy-score variance as growing roughly linearly with completion
length, `v_i ≈ kappa * p_i(1-p_i) * l_i`. Under that explicit assumption, the
length terms cancel:

```text
n_i ∝ sqrt(p_i(1-p_i)).
```

The implementation substitutes the posterior expectation, reserves 15% of
each iteration's budget for the least-recently-sampled prompts, distributes the
remainder proportionally to `score_i`, caps each prompt at eight rollouts, and
performs deterministic integer water-filling until the budget is exhausted.
This keeps stale prompts revisitable while concentrating most compute near the
learning frontier.

### Predictable baselines make adaptive sampling safe

Let `S_i(y) = grad log pi(y | prompt_i)` be the policy score and let `r` be
binary correctness. For any baseline fixed before sampling the current answer,

```text
E[(r - baseline_i) * S_i(y) | past] = grad P(correct | prompt_i).
```

VOLT therefore uses the simple advantage

```text
A = r - baseline_i.
```

The crucial word is **predictable**: both the baseline and rollout allocation
are functions only of earlier iterations. Conditioned on that history, they
are constants, so adaptive prompt selection introduces no additional
within-prompt policy-gradient bias. A single rollout is enough:

- if `baseline = 0.2`, success gets advantage `+0.8` and failure `-0.2`;
- if `baseline = 0.8`, success gets `+0.2` and failure `-0.8`.

Unexpected outcomes receive the strongest correction. Homogeneous samples no
longer collapse mechanically to zero, although a nonzero advantage does not
mean every rollout is equally valuable.

```mermaid
flowchart LR
    H["Past rollout outcomes"] --> P["Discounted hierarchical Beta state"]
    P --> S["Freeze snapshot at iteration start"]
    S --> B["Predictable baseline"]
    S --> A["Variance allocation score"]
    A --> W["Water-fill the rollout budget"]
    W --> G["Generate and score answers"]
    B --> V["Advantage = reward - baseline"]
    G --> V
    V --> U["One on-policy LoRA update"]
    G --> Q["Update posterior and length statistics"]
    Q --> P
```

### The implemented training step

VOLT broadcasts each sequence-level advantage across the generated completion
tokens and performs one strictly on-policy REINFORCE update:

```text
loss = -sum_rollouts(advantage * sum_completion_tokens(log_probability))
       / (number_of_rollouts * constant_length_normalizer)
```

There is no current-group standard-deviation division, no per-answer token
average that overweights short completions, no PPO replay, and no explicit KL
penalty. Gradient norm is clipped, and only a rank-32 LoRA adapter is updated;
the base checkpoint remains untouched.

The E2B experiment uses 48 iterations × 448 rollouts, temperature 0.9, up to
384 new tokens, and a fixed 300-example greedy validation probe every five
iterations. The training pool contains 1,040 usable calibration prompts, with
16 whole tasks excluded from training to measure unseen-task transfer.

### Optional length control

VOLT also implements a success-conditioned primal-dual length constraint. It
can penalize long **correct** answers while never rewarding a wrong answer for
giving up early. Its multiplier and shaped baseline use frozen historical
length statistics, preserving predictability. This component is implemented
but disabled in the current E2B configuration, so the active experiment tests
the posterior baseline and adaptive allocator cleanly.

### Scientific status and boundaries

- Each rollout is conditionally unbiased for its sampled prompt, but the
  current loss gives prompts weight proportional to their allocated rollout
  counts. The implemented update therefore optimizes an adaptive curriculum,
  not an exactly uniform average over all prompts. Exact uniform weighting
  would require per-prompt normalization or importance weights.
- Calling the allocation token-optimal relies on the testable assumption that
  policy-score variance grows linearly with sequence length. If it does not,
  the optimal rule should retain an explicit length-cost term.
- Posterior discounting handles policy drift only approximately, and
  task-level pooling can mislead when prompts within one task are heterogeneous.
- Training telemetry shows the intended mechanism—VOLT keeps nonzero
  advantages where GRPO discards many homogeneous groups—but the frozen-test
  comparison remains the standard for any final accuracy claim.

The full derivations, proofs, related-work positioning, and manuscript are in
[paper/volt/](paper/volt/). Train and evaluate on the GPU host with:

```bash
# compute-matched baseline and method
python rl/run_train.py --config experiments/rl/grpo_e2b.json
python rl/run_train.py --config experiments/rl/volt_e2b.json

# validation selection followed by frozen-test evaluation
./scripts/run_rl_evals.sh
```

Training uses calibration rows only, checkpoints are selected using validation
rows, and the frozen test split is reserved for the final selected models.

## Reproducing the evaluation

Download the datasets (every run records the dataset git revisions it used in
its `run_config.json`; the original study used BBH `9ee07bd`, BBEH `80d12ca`,
USR `39bc520`):

```bash
DATA_ROOT=/data/benwulab/gemma4-eval/datasets ./scripts/download_datasets.sh
```

Smoke test (2 examples per task):

```bash
python3 eval_benchmarks.py \
  --datasets-root /data/benwulab/gemma4-eval/datasets \
  --base-url http://127.0.0.1:8888/v1 \
  --model SubTokenLLM \
  --benchmarks bbh,bbeh \
  --limit-per-task 2 \
  --parallel 2 \
  --output-dir /data/benwulab/gemma4-eval/runs/smoke
```

Useful flags: `--prompt-strategy <name>` (any strategy above),
`--self-consistency-k N` plus `--response-selection majority_vote|self_rank`,
`--prompt-policy policy.json` for routed runs, `--benchmarks bbh,bbeh,usr`.
Every run directory gets `run_config.json`, `predictions.jsonl`, and
`summary.json`. `scripts/start_full_strategy_matrix.sh` runs the whole
strategy matrix; `scripts/summarize_strategy_runs.py` aggregates it.

## Deployment (ops)

E4B and E2B are served on `benwulab-remote` ports 8888/8889; a body-preserving
router on 8890 dispatches by model ID; a reverse SSH tunnel plus an Nginx
rewrite exposes it at `https://llm.agaii.org/llm/v1`. The router never injects
prompts and forwards request bodies unchanged. Systemd units and configs are
in [ops/](ops/); install with `./ops/install_systemd_services.sh`.

## Repository map

```
eval_benchmarks.py       evaluation harness and scorer (single file)
rl/                      VOLT RL training package (LoRA, allocation, eval)
scripts/                 strategy matrix, router calibration, RL evals, analysis
experiments/             frozen protocols, arm manifests, RL run configs
docs/                    protocol, research notes, detailed result tables
paper/e2b-e4b-study/     prompt-study manuscript and audits
paper/volt/              VOLT method, theory (proofs), manuscript draft
results/                 archived run bundles (predictions, summaries, logs)
ops/                     serving, router, tunnels, systemd units
tests/                   unit tests (scorer, router, protocol, RL math)
```
