# Answer-stable frozen inference — experimental v0.1

This is an opt-in **system-level** experiment, not a new Gemma checkpoint or a
replacement for the registered CBRR/LoRA study. No existing evaluator, scorer,
manifest, result bundle, model weights, adapter, or serving script is modified.
The main integration point is `python -m frozen_inference` from the repository root.

## Implemented

| Component | Implementation | Boundary |
|---|---|---|
| Exact fast path | Full-input Boolean and rational-arithmetic parser; bounded AST execution | No arbitrary Python, substring extraction, floats, calls, attributes, or coercing bools to integers |
| Answer-stable interpretation | Typed expression, swap-tracking, and strict-ordering IR | Natural-language extraction is a model prediction, not a proof |
| Backward slicing/state merging | Propagate possible source positions backward from one tracking query | One queried final value; correlated alternatives may be overapproximated |
| Local ambiguity repair | Resolve only a source clause whose alternatives can change the answer | Greedy one-clause selection can miss ambiguity interactions |
| Full-resolution control | `--mode full` resolves all ambiguous clauses before accepting | Same compiler/executor, explicit call/repair limits |
| Microprogram memory | Read-only literal templates, bounded integer slots, safe arithmetic, positive and negative development tests | Included library is a small synthetic example, not a learned benchmark solver bank |
| Repairability policy | Paired win/loss lower-bound gate penalized by incremental elapsed cost | Calibration only, unseen feature buckets fall back; no online self-rewards |
| Counterfactual scoring | Cyclic independent-option views, mapped multi-token sequence likelihoods | Optional local-model utility; agreement is not a correctness certificate |
| Activation transport | Paired activation contrast and temporary named-module hooks | Optional local-model utility; no architecture-specific vectors are supplied |

**There are no new Gemma accuracy or speedup measurements in this branch.** Unit
and synthetic tests validate implementation properties, not reasoning gains on
BBH, BBEH, or USR. The existing published test informed these hypotheses; use a
newly declared holdout for confirmatory claims. A `--study-id` is a provenance
label, not evidence that the data have never been seen.

## Quick offline check (no model or extra packages)

```bash
python -m unittest discover -s tests -p 'test_frozen_*.py' -v
python -m frozen_inference validate-skills experiments/frozen_inference/skills.json
python -m frozen_inference run \
  --examples experiments/frozen_inference/smoke.jsonl \
  --skills experiments/frozen_inference/skills.json \
  --offline --unscored --split external \
  --study-id synthetic-smoke-v1 \
  --output-dir /tmp/answer-stable-smoke-v1
```

Use a new output directory each time. The smoke run returns `True`, `42`, and
`143` with zero model calls, and **intentionally leaves one unsupported question
unanswered** because the fallback model is disabled. It is unscored and is not a
benchmark result. The default GitHub test workflow discovers the new tests;
PyTorch-specific tests are skipped when the optional dependency is absent.

## Live calibration using the existing local API

The core runtime needs only Python's standard library. It sends one user message
per call and uses the same OpenAI-compatible endpoint shape as the legacy
harness. The model remains frozen. No system message, adapter, arbitrary code
execution, forced server-side schema, or hidden teacher is introduced.

```bash
DATA_ROOT=/data/benwulab/gemma4-eval/datasets
BASE_URL=http://127.0.0.1:8889/v1

# Direct control under this runner, including the same cost instrumentation.
python -m frozen_inference run \
  --datasets-root "$DATA_ROOT" --base-url "$BASE_URL" \
  --model SubTokenLLM-E2B --deployment-id my-pinned-e2b-bf16-stack \
  --mode direct --split calibration --study-id answer-stable-dev-v1 \
  --output-dir /tmp/as-direct-cal

# Conditional model compiler + answer-stable executor + bounded local repairs.
python -m frozen_inference run \
  --datasets-root "$DATA_ROOT" --base-url "$BASE_URL" \
  --model SubTokenLLM-E2B --deployment-id my-pinned-e2b-bf16-stack \
  --mode stable --compile --split calibration --study-id answer-stable-dev-v1 \
  --max-calls 4 --max-repairs 2 --compile-tokens 512 --answer-tokens 64 \
  --output-dir /tmp/as-compile-cal

python -m frozen_inference fit-policy \
  --baseline /tmp/as-direct-cal --challenger /tmp/as-compile-cal \
  --min-examples 20 --penalty-per-second 0.01 \
  --output /tmp/as-policy.json

python -m frozen_inference run \
  --datasets-root "$DATA_ROOT" --base-url "$BASE_URL" \
  --model SubTokenLLM-E2B --deployment-id my-pinned-e2b-bf16-stack \
  --mode stable --compile --policy /tmp/as-policy.json \
  --split validation --study-id answer-stable-dev-v1 \
  --output-dir /tmp/as-policy-validation
```

`--compile` is deliberately off by default. Without it, the system tries exact
recognition and an optional `--skills` library, then direct answering. With it
but without a policy, all remaining examples are eligible for compilation; this
is an explicitly exploratory development arm and may be slower or less accurate.
A supplied policy is bound to the model/deployment identifier and inference caps
used by its challenger. Deployment identifiers are user assertions: the client
cannot remotely prove which weight bytes or kernels a server is using.

For a full-disambiguation ablation, use `--mode full --compile` in a separate run.
For exact/template-only ablation, omit `--compile`. Apply the same engineering
optimizations, precision, concurrency, and reasonable budget sweep to all arms.
Do not compare accuracy against one baseline and latency against another.

`--benchmarks bbh`, `--task bbh/boolean_expressions` (repeatable), `--limit N`, and
`--parallel N` support development runs. A limited/task-filtered run is not a
complete benchmark evaluation. This runner's seeds are question/action-derived;
its direct control is not claimed to be byte-identical to an archived legacy run.

## Data and scoring isolation

The dataset-root route reuses `eval_benchmarks.load_bbh`, `load_bbeh`, and
`load_unpuzzles_simple_reasoning`. Scored runs reuse the unchanged
`eval_benchmarks.evaluate_correctness`. A JSONL input uses records such as:

```json
{"benchmark":"my_suite","task":"tracking","index":0,"input":"...","target":"(B)"}
```

Targets are required for scoring; omit them and select `--unscored` for deployment
or smoke inputs. `Pipeline.predict` accepts **only the question**, never targets,
benchmark task IDs, or rewards. The orchestration layer scores afterward.

Splits match the existing index convention: calibration 0–24, validation 25–49,
test >=50. Selecting `test` additionally requires `--allow-test`. Policy fitting
rejects test/validation rows, duplicate identities, unequal coverage, changed
input hashes, unscored runs, or model/deployment mismatches. Only examples where
the challenger actually attempted compilation contribute to the compilation gate;
exact/template wins cannot inflate its value. Use a fresh validation set for gate
selection. The paired confidence bounds are not a global statistical guarantee.
`external` makes no assertion about historical benchmark split membership.

## Typed interpretation contract

The model can produce an expression, swap-tracking state, or bounded ordering
problem. A tracking example:

```json
{
  "kind": "tracking",
  "complete": true,
  "initial": {"Alice":"red ball","Bob":"blue ball","Claire":"green ball"},
  "query": "Alice",
  "steps": [
    {"source":"Alice trades with another player.",
     "alternatives":[["Alice","Bob"],["Alice","Claire"]]}
  ]
}
```

The quoted source must occur literally in the original question. Each alternative
is one swap; steps are chronological. Ordering pairs mean strict `before`, and
its query is a zero-based rank. Arbitrary moves, copies, historical queries,
adjacency constraints, distances, ties, defeasible rules and BoardgameQA-style
semantics are **not** silently reduced to these forms. Unsupported inputs must
fall back. Up to seven ordering entities are exhaustively checked under a shared
time/work budget; reaching a limit never yields a stability certificate.

Backward tracking computes the possible original sources of the requested final
position. Distinct histories with the same projected source state can be merged.
A late swap can make an earlier ambiguity relevant; the implementation analyzes
backward through the entire remaining program rather than discarding it based on
an intermediate answer. Tests compare this analysis with exhaustive execution on
1,000 seeded random programs.

If every represented interpretation gives the same answer, stable mode stops. If
not, it chooses a clause maximizing worst-case answer-set reduction and asks the
frozen model to resolve that clause using the **full original context**. It never
asks the human user. Invalid/null choices, inconsistent interpretations, failed
requests, truncation, unsupported outputs and exhausted repair budgets trigger
fallback. One logical call is reserved for fallback; total deadlines are
best-effort and can leave an example unanswered if no time remains.

The certificate is **conditional on the interpretation including the true
semantics**. An exact source quote and `complete:true` do not prove completeness.
The current compiler emits the whole bounded IR with alternatives; this version
does not yet implement token-by-token streaming partial extraction. It avoids
unnecessary disambiguation and execution state, not necessarily all initial
extraction tokens. It also does not solve arbitrary ambiguous semantics.

## Microprogram memory

`experiments/frozen_inference/skills.json` demonstrates the format. Templates are
literal strings with unique `{slot:int}` placeholders and explicit integer
bounds. There are no caller-provided regexes, generated Python programs, dynamic
imports or per-example answer lookups. Two positive development checks and a
negative applicability check are the minimum, not sufficient evidence of broad
semantic validity. Conflicting matching programs cause abstention. Build richer
libraries only from permitted development material; record construction cost and
evaluate held-out template compositions. This version validates user-authored
libraries; it does not automatically synthesize or optimize them.

## Optional tool-free local-model experiments

These utilities require PyTorch and an already loaded, pinned model/tokenizer.
They perform no downloads. Optional Transformers loading is the caller's
responsibility; select a local snapshot and the same tokenizer/template used for
the baseline. No particular Gemma architecture path is assumed.

```python
from frozen_inference.neural import FrozenSequenceScorer, counterfactual_score

# model and tokenizer are already loaded from the chosen local snapshot.
scorer = FrozenSequenceScorer(model, tokenizer, max_context=4096)
result = counterfactual_score(
    "Which color?\n(A) red\n(B) blue\n(C) green", scorer, views=3
)
print(result)
print(scorer.telemetry)  # Include every repeated prefill and forward pass.
```

The scorer handles multi-token labels by exact teacher forcing, not incomplete
API top-logprobs. The reference batches full sequences and **does not implement
shared-prefix KV caching**. It may be slower than direct answering. The option
guard rejects common dependent-option forms but does not prove arbitrary natural
language invariance. Use only independent choices. Mapped agreement is evidence
for a separately calibrated gate, never a correctness certificate. Counterfactual
scoring is a library experiment, not enabled in the API CLI pipeline.

```python
from frozen_inference.neural import contrast_vector, activation_transport

# Paired [examples, hidden] activations from answer-balanced DEVELOPMENT cases.
vector = contrast_vector(positive_activations, negative_activations)
# Replace module_path with an exact path verified in your loaded model.
with activation_transport(model, {module_path: vector}, scale=0.1):
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
```

Transport hooks act only on the final position of a single, unpadded example and
are removed even after exceptions. They do not mutate parameters. Do not share
that model instance with concurrent requests while hooks are installed. Layer
choice, vector selection, norm/scale and gates require development calibration;
random and wrong-task vectors are mandatory controls. No meaningful Gemma vector,
causal claim, or quality gain is supplied by this reference utility.

## Audit artifacts and cost interpretation

Each new output directory contains `run_config.json`, its SHA-256,
`predictions.jsonl`, and `summary.json`. Configuration is written before inference.
Outputs refuse overwrite and do not support resuming/mixing protocols. The config
hash excludes the trailing newline in its pretty-printed JSON serialization.
Records include source-code/input/policy/library hashes, user-provided deployment
ID, all logical calls and HTTP attempts, full generated message payloads, finish
reasons, nested reasoning-token usage when supplied, repair events, IR, final
analysis, errors and elapsed time. **These records contain question/model text and
scored targets; handle them as dataset artifacts, not public telemetry by default.**
API keys and Authorization headers are not persisted.

Retries default to zero; enabling them increases HTTP attempts beyond the logical
call budget. If a request fails, remote compute may already have happened. Usage
is marked incomplete, and unknown token cost is not reported as zero. End-to-end
means/medians/nearest-rank p95 include pipeline calls and analysis, but not dataset
loading, server queue telemetry, or offline library construction. `wall_seconds`
adds runner overhead and concurrency effects. There is no GPU-kernel timing from
the HTTP API. Capture server-side prefill, decode, cache, memory and GPU-time
measurements separately before making speedup claims. Record the actual rendered
chat template: one user message does not prove the server adds no control turns.

For a new paper, compare live direct, optimized CBRR, budgeted native thinking,
full compilation, stable compilation, memory, and the calibrated gate. Measure
joint-failure recovery over direct and CBRR, conditional-certificate error rate,
per-task effects, fallback rates and the accuracy–latency frontier. Do not relabel
these system experiments as prompt-only or merge them into the original frozen
confirmatory result table.
