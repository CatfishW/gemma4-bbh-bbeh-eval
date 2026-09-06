# Running the comparison suite

Read [the research/novelty/protocol plan](SOTA_COMPARISON_PLAN.md) first. Commands
below run only when explicitly invoked; no models, datasets or packages are
downloaded at import. Run from the repository root. Use fresh output directories.

## Validate code and environment

```bash
python -m unittest discover -s tests -p 'test_comparison_*.py' -v
python comparison_suite.py doctor --output /tmp/comparison-environment.json
```

Core dataset, prompt, artifact and statistics functionality uses the standard
library. CPU training tests additionally use torch. The local-model baseline and
training bridges need the existing Transformers/PEFT environment. Symbolic MATH
grading requires the separate optional math dependency file. There is no unified
safe `pip install` command for every historical upstream environment.

## Prepare real official sources without changing their roles

For GSM8K, use local official files described by the authors' repository:

```bash
python comparison_suite.py prepare \
  --input /path/to/grade_school_math/data/train.jsonl --format gsm8k \
  --dataset gsm8k-train --split train --source-split train \
  --revision ACTUAL_DATA_REVISION --expected-count 7473 --dev-fraction 0.1 \
  --output /tmp/gsm8k-train-dev.jsonl

python comparison_suite.py select --data /tmp/gsm8k-train-dev.jsonl \
  --split train --output /tmp/gsm8k-train-only.jsonl
python comparison_suite.py select --data /tmp/gsm8k-train-dev.jsonl \
  --split dev --output /tmp/gsm8k-dev.jsonl

python comparison_suite.py prepare \
  --input /path/to/grade_school_math/data/test.jsonl --format gsm8k \
  --dataset gsm8k-test --split test --source-split test \
  --revision ACTUAL_DATA_REVISION --expected-count 1319 \
  --output /tmp/gsm8k-test.jsonl
```

These are standard expected GSM8K counts; the commands deliberately fail rather
than silently accept a different local export. Supply the actual content revision,
not the literal placeholder. Selection keeps complete declared groups together.
For MATH/MATH-500 use `--format math`/`math500`; a local JSON list or JSONL must
include the problem and final answer or a boxed solution. CSV is supported where
columns match the corresponding adapter. The tool does not parse arbitrary
Parquet/HF Arrow caches: use the benchmark's local JSON export.

Supported converters: `canonical`, `gsm8k`, `math`, `math500`, `aime`, `gpqa`,
`mmlu_pro`, `bbh`, `bbeh`, `svamp`, `deepscaler`. `--group-field` requires that
field on every input. `merge` validates duplicate IDs and cross-role group/question
leakage; no near-duplicate semantic detector is claimed.

Legacy full-validation export uses the unchanged existing loaders and keeps
source puzzle families together:

```bash
python comparison_suite.py export-legacy \
  --datasets-root /data/benwulab/gemma4-eval/datasets \
  --benchmarks bbh,bbeh,usr --legacy-split validation \
  --expected-count 1490 --output /tmp/legacy-validation.jsonl
```

The expected count applies to the repository's recorded revisions. The exporter
labels these historically exposed data evaluation-only and will not turn them
into new training sources. Legacy BBEH records use `bbeh_official`: provide the
pinned official `bbeh/bbeh/evaluate.py` file to scored runs. Legacy scorer results
remain separate from new officially graded results.

## Run frozen prompting baselines

Copy `experiments/comparison/identity.example.json` and fill in every field. API
model hashes are assertions made by the operator; the client cannot prove which
remote weights are served. Local-model runs recompute base/adapter file identities.
Run equal serving/load profiles for latency comparisons.

```bash
python comparison_suite.py run \
  --data /tmp/gsm8k-dev.jsonl \
  --method experiments/comparison/methods/cot.json \
  --identity /path/to/filled-identity.json \
  --base-url http://127.0.0.1:8889/v1 --model SubTokenLLM-E2B \
  --seed 7 --output /tmp/cot-dev

python comparison_suite.py run \
  --data /tmp/gsm8k-dev.jsonl \
  --method experiments/comparison/methods/cod.json \
  --identity /path/to/filled-identity.json \
  --base-url http://127.0.0.1:8889/v1 --model SubTokenLLM-E2B \
  --seed 7 --output /tmp/cod-dev

python comparison_suite.py compare --baseline /tmp/cot-dev \
  --candidate /tmp/cod-dev --replicates 10000 --output /tmp/paired-cod.json
```

Replace `--base-url` with `--model-path /path/to/local/checkpoint` for direct local
inference. Use `--loader causal` for a compatible local causal-model family;
default `legacy_gemma` uses the repository's proven multimodal loader. `--adapter`
loads a trained LoRA. `--max-context` is explicit and never silently truncates
questions. Test-role data additionally require `--allow-test` after freezing the
study. No endpoint is contacted by `prepare`, `compare` or tests.

Method JSON selects `direct`, `cot`, `cod`, `plan_solve`, `self_verify`,
`self_consistency`, `self_rank`, `self_refine`, or `native_thinking`. `budget` is the
**total** completion budget across all inference calls. `k`, `rounds`, sampling
`temperature` and a ranking `final_budget` are explicit. Sweep copied configs
on development data; supplied budgets are examples, not an already optimized
frontier. All presets default to zero demonstrations. Any added demonstrations
must be frozen from allowed training material and disclosed; it is not the
paper's few-shot setup merely because its method name matches.

Native thinking passes an explicit tokenizer/API template flag. A remote service
may ignore unsupported extras: audit its rendered template. Local native thinking
requires an explicit response-channel parser; the code refuses heuristic mixing
of private-thinking and final channels. The decoder records full output tokens,
not just visible final text. HTTP failures are not retried based on labels.

## Import genuine upstream modules

Run the original baseline in its isolated reviewed environment. Export one row
per canonical case, retaining all failed cases, using this contract:

```json
{"id":"gsm8k-test/0","case_digest":"SHA256_OF_CANONICAL_CASE","prediction":"...","elapsed_seconds":1.2,"input_tokens":300,"output_tokens":150,"error":null,"calls":[]}
```

`case_digest` is computed with `comparison_suite.digest(dataclasses.asdict(case))`.
Do not map by row number alone after shuffling/filtering. The supplied `correct`
field, if any, is ignored. Metadata follows `upstream-run.example.json`: exact
40-character source commit, executed recipe/modifications, same model/hardware
identity fields, seed, and whether **all** calls were accounted. Incomplete cost
accounting makes aggregate token cost unknown; it does not become zero.

```bash
python comparison_suite.py import-run --data /tmp/gsm8k-dev.jsonl \
  --predictions /path/to/upstream-export.jsonl \
  --identity /path/to/upstream-run-metadata.json \
  --output /tmp/thinkless-imported-dev
```

No `exec` of generated PAL/PoT code is added to this evaluator. Run full program
baselines only in a proper isolated execution environment, with no secrets or
network and bounded resources. Game24 cannot be truthfully scored by numeric 24:
its task verifier and full state trace must be provided by the upstream run; the
current native suite does not implement that verifier. Do not import it as a
plain numeric task and call the result an official ToT comparison.

## Train quality controls and CRAFT-Q

```bash
python comparison_training.py --data /tmp/gsm8k-train-only.jsonl \
  --config experiments/comparison/training/rloo.json \
  --model-path /data/models/gemma-4-E2B-it --device cuda:0 \
  --output /data/benwulab/gemma4-rl/comparison-rloo-seed7

python comparison_training.py --data /tmp/gsm8k-train-only.jsonl \
  --config experiments/comparison/training/craft_quality.json \
  --model-path /data/models/gemma-4-E2B-it --device cuda:0 \
  --output /data/benwulab/gemma4-rl/comparison-craftq-seed7
```

`grpo_reference.json`, `drgrpo_reference.json`, and `sft_answers.json` define other
controls. The SFT-only control defaults to ordinary temperature-one cross-entropy.
Optional `--initial-adapter /path/to/warmup/adapter-final` starts a new declared
run and records the warmup identity; it is not optimizer resume. Apply the same
warmup to every arm in the warmup-matched comparison and keep a no-warmup ablation.
The reference trainer has no automatic distributed scheduler and does not alter
running services. Multiple seeds require independently launched configurations;
changing a config after a run does not retroactively alter its protocol.

For CRAFT-Q checkpoint evaluation through the exact same scorer and paired output
contract:

```bash
python comparison_craft.py --data /tmp/gsm8k-dev.jsonl \
  --model-path /data/models/gemma-4-E2B-it \
  --training-run /data/benwulab/gemma4-rl/comparison-craftq-seed7 \
  --identity /path/to/filled-identity.json --seed 7 --gate sample \
  --output /tmp/craftq-dev
```

Use `--checkpoint /path/to/original/checkpoint-00012` instead of `--training-run`
for original CRAFT checkpoints. Original artifact integrity and base-model bytes
are checked. `--gate always-continue` is a required content-quality control;
`greedy` is a distinct deployment policy. No gold enters `predict` or its gate.
The common post-hoc scorer does not automatically require EOS to call a visible
answer correct; termination is separately reported. This is intentional matched
scoring, not a rewrite of CRAFT's historical EOS-gated training/pilot results.

For single-path evaluation of any trained content adapter, use the local baseline
runner with `--adapter` and the same CoT/native prompt profile as the base. A
CRAFT adapter under ordinary CoT is a **prompt-transfer experiment**, not the
same policy as staged CRAFT.

## Output integrity and interpretation

Every run writes immutable `protocol.json`, `predictions.jsonl`, `summary.json`.
Missing rows, mismatched case/scorer/seed hashes, tampered predictions and partial
runs fail comparison. Inference errors score incorrect; grader/library errors
abort rather than silently become a different scoring rule. Time spent grading
is separate. Secrets from environment variables are not serialized. Raw response
payloads may contain benchmark text and reasoning: keep them with permitted
research artifacts rather than automatically publishing them.

The comparison output contains paired accuracy changes, group/task sensitivity
intervals, latency ratios, exact McNemar and Holm results. It marks different-base
comparisons descriptive and always leaves `sota_claim=false`. Actual training-seed
uncertainty and fully matched GPU performance still require the planned experiments.
