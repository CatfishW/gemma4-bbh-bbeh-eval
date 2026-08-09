# Gemma 4 E2B/E4B Reasoning Evaluation

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
the *test* split (9,550 examples) that nothing is ever tuned on. All headline
numbers below are from that untouched test split.

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

| Approach | Weights changed? | Accuracy | Avg tokens per answer |
|---|---|---:|---:|
| Ask for the answer directly (`direct_answer`) | No | 26.39% | 15 |
| Best universal prompt (`concise_cot_self_rank_k3`) | No | 35.06% | 681 |
| CBRR per-task prompt router | No | **35.41%** | 66 |
| VOLT RL fine-tuning (this branch) | Yes (LoRA) | *training in progress* | - |

How to read this: just asking better questions moves this small model from
26% to 35% — and the router gets the same accuracy as the best prompt while
generating one tenth as much text. The gain is statistically solid (McNemar
p ≈ 1e-132, bootstrap +8.4 to +9.6 points) and concentrated in BBH; BBEH
barely moves with any prompt. On the larger E4B model the router reaches
41.6% on the matched validation set.

All other tables (per-dataset breakdowns, both models, all 29 arms, robustness
checks) are in [docs/DETAILED_RESULTS.md](docs/DETAILED_RESULTS.md).

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

## VOLT: reinforcement learning under a token budget (this branch)

Prompt routing tops out near 35% because the weights never change. The `rl/`
package trains the E2B weights (LoRA adapters, base model untouched on disk)
with **VOLT** — a new RL algorithm built for exactly this setting: little
data (the 1,040 usable calibration prompts), one shared GPU, and a hard
generation budget.

The idea in one paragraph: standard group-based RL (GRPO) spends the same
number of sampled solutions on every prompt, but prompts the model always
solves — or never solves — provably teach it nothing. VOLT keeps a small
Bayesian difficulty estimate for every training prompt (pooled across tasks,
decaying over time) and uses that *one* estimate three ways: to decide how
many rollouts each prompt deserves (more where the model is 50/50), as the
baseline that keeps the learning signal unbiased even though the sampling is
adaptive (the martingale argument GRPO-family methods are missing), and to
throttle answer length toward a deployment budget. Degenerate all-correct or
all-wrong sample groups — dead weight in GRPO — still carry signal here.

Details, proofs, and positioning against 2025-26 work (DAPO, Dr. GRPO, SPO,
DynaMO, DUET, Knapsack-RL, Reinforce-Ada): [paper/volt/](paper/volt/).

Train and evaluate (on the GPU host):

```bash
# baseline GRPO and VOLT, same budgets, same protocol
python rl/run_train.py --config experiments/rl/grpo_e2b.json
python rl/run_train.py --config experiments/rl/volt_e2b.json

# frozen-split evaluation of base model and both adapters
./scripts/run_rl_evals.sh
```

Training uses only calibration rows (16 of 60 tasks additionally held out to
measure transfer), checkpoints are selected on validation rows, and the test
split is touched exactly once per final model. Results will be added to the
headline table when the runs complete.

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
