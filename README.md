# Gemma 4 E2B/E4B Reasoning Evaluation

Small, auditable harness for evaluating the Gemma 4 E2B and E4B deployments against:

- BBH from `suzgunmirac/BIG-Bench-Hard`
- BBEH from `google-deepmind/bbeh`
- USR from `google-deepmind/unpuzzles_and_simple_reasoning`

The evaluator uses the OpenAI-compatible chat completions API and intentionally sends no system message. Each request contains exactly one `user` message.

## Access

Use:

```bash
curl https://llm.agaii.org/llm/v1/models
```

OpenAI SDK shape:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://llm.agaii.org/llm/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="SubTokenLLM-E2B",  # Use SubTokenLLM for E4B.
    messages=[{"role": "user", "content": "Reply with exactly: online"}],
    max_completion_tokens=16,
    temperature=0,
)
print(response.choices[0].message.content)
```

Available public model IDs:

- `SubTokenLLM`: Gemma 4 E4B.
- `SubTokenLLM-E2B`: Gemma 4 E2B.

The gateway dispatches by `model` while forwarding the original request body unchanged.
It does not inject a system prompt. Direct remote endpoints:

```bash
curl http://127.0.0.1:8888/v1/models
curl http://127.0.0.1:8889/v1/models
```

Tang tunnel endpoint after the shared model router and `ops/tunnel_tang_25570.sh` are running:

```bash
curl http://127.0.0.1:25570/v1/models
```

## Download Datasets

On `benwulab-remote`:

```bash
DATA_ROOT=/data/benwulab/gemma4-eval/datasets ./scripts/download_datasets.sh
```

Known downloaded revisions for the first run:

- BBH: `9ee07bd481feebf959a6b59d61ea57bdcf30964d`
- BBEH: `80d12ca916b7158f22293fcf3144f4d3d854d4be`

## Unpuzzles And Simple Reasoning

The `usr` benchmark name loads all auto-scorable examples from `google-deepmind/unpuzzles_and_simple_reasoning`:

- `simple_reasoning/*` from `simple_reasoning.json`
- non-empty `unpuzzles/original` and `unpuzzles/unpuzzle` examples from `unpuzzles.json`
- non-empty `shifted_unpuzzles/original`, `shifted_unpuzzles/unpuzzle`, and `shifted_unpuzzles/shifted` examples from `shifted_unpuzzles.json`

You can also run only `simple_reasoning`, `unpuzzles`, or `shifted_unpuzzles`.

```bash
python3 eval_benchmarks.py \
  --datasets-root /data/benwulab/gemma4-eval/datasets \
  --base-url http://127.0.0.1:8888/v1 \
  --model SubTokenLLM \
  --benchmarks usr \
  --prompt-strategy direct_answer \
  --parallel 2 \
  --output-dir /data/benwulab/gemma4-eval/runs/usr-direct
```

## Run Eval

Smoke test:

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

Full run:

```bash
python3 eval_benchmarks.py \
  --datasets-root /data/benwulab/gemma4-eval/datasets \
  --base-url https://llm.agaii.org/llm/v1 \
  --model SubTokenLLM \
  --benchmarks bbh,bbeh \
  --parallel 2 \
  --output-dir /data/benwulab/gemma4-eval/runs/full
```

Use `--prompt-strategy raw` if you want to send the dataset input exactly as stored. The default `direct_answer` strategy appends a user-level instruction asking the model to output only the final answer. `--prompt-mode raw` and `--prompt-mode answer_only` remain as backward-compatible aliases.

Available prompt strategies:

- `direct_answer`: final answer only baseline.
- `strict_json`: JSON object with one `answer` field.
- `native_format`: preserve the answer format requested by each item.
- `canonical_short`: normalize labels, booleans, numbers, and lists.
- `private_verify`: solve and check once privately, then answer only.
- `selective_verify`: revise only when a targeted check finds a concrete contradiction.
- `compare_then_commit`: compare the two strongest candidates privately.
- `fast_slow_gate`: verify once only when the direct answer is uncertain.
- `constraint_guard`: replay the decisive constraints before committing.
- `negation_label_guard`: protect negations, quantifiers, and option-label mapping.
- `draft_verify`: use a terse private draft followed by one check.
- `concise_cot`: concise reasoning with a final-answer delimiter.
- `chain_of_draft`: terse scratch notes with a final-answer delimiter.
- `plan_and_solve`: short plan, solve, final-answer delimiter.
- `step_back`: identify the general principle, then solve.
- `premise_conclusion`: explicit premise-to-conclusion template.
- `symbolic_proof`: compact symbolic translation or proof sketch, then solve.
- `plan_and_solve_plus`: detailed plan, variable extraction, solve, and verification.
- `least_to_most`: solve dependency-ordered subproblems from simplest to hardest.
- `condition_reconstruction`: reconstruct the decisive condition before correction.
- `counterexample_guard`: try one targeted counterexample before committing.
- `rank_two_paths`: privately compare two distinct compact solution paths.
- `raw`: dataset input only.

Self-consistency is enabled with `--self-consistency-k`. The evaluator records every generation and chooses the majority normalized final answer. It still sends no system message.

## Reward-Routed Prompt Policy

`scripts/calibrate_prompt_policy.py` treats each prompt strategy as a Beta-Bernoulli bandit arm. It selects one fixed arm per benchmark task from the first 25 examples, preserves `direct_answer` on ties, and scores only examples with index 25 or greater. The resulting `policy.json` can be passed directly to the evaluator with `--prompt-policy`; every prediction records the arm that produced it.

```bash
python3 scripts/calibrate_prompt_policy.py \
  --runs-root /data/benwulab/gemma4-eval/runs/full-strategy-matrix-20260706_025955 \
  --runs-root /data/benwulab/gemma4-eval/runs/usr-strategy-matrix-20260707_022230 \
  --runs-root /data/benwulab/gemma4-eval/runs/full-challenger-winners-20260709_120053 \
  --strategies direct_answer,strict_json,concise_cot,chain_of_draft,plan_and_solve,step_back,premise_conclusion,symbolic_proof,canonical_short,private_verify \
  --calibration-size 25 \
  --datasets-root /data/benwulab/gemma4-eval/datasets \
  --output-dir /data/benwulab/gemma4-eval/runs/reward-routed-policy/offline
```

Run the complete calibration and online held-out confirmation with:

```bash
./scripts/run_reward_routed_policy.sh
```

To fit a conservative policy from a calibration-only strategy sweep and confirm it on the held-out suffix:

```bash
SAMPLE_ROOT=/data/benwulab/gemma4-eval/runs/research-strategy-sweep-20260709_183952 \
./scripts/run_sampled_policy_confirmation.sh
```

The fully live `reward_routed_v2` run scores `4,020/11,040` (36.41%), compared with `3,632/11,040` (32.90%) for direct answer: +388 correct and +3.51 accuracy points with zero request errors. The higher-ceiling v1 archived replay scores `4,738/11,040` (42.92%) and is recorded separately from live confirmation. See [docs/PROMPT_OPTIMIZATION_RESEARCH.md](docs/PROMPT_OPTIMIZATION_RESEARCH.md) for the protocol, research basis, examples, negative results, and RL decision.

## Prompt Strategy Matrix

The full strategy runner executes the prompt strategies from the prompt-strategy table that are meaningful for BBH/BBEH without external tools or custom verifiers. It records one folder per strategy with `run_config.json`, `command.txt`, `stdout.log`, `stderr.log`, `predictions.jsonl`, and `summary.json`.

```bash
cd /data/benwulab/gemma4-eval/repo
RUNS_ROOT=/data/benwulab/gemma4-eval/runs/full-strategy-matrix-$(date +%Y%m%d_%H%M%S) \
PARALLEL=2 \
BASE_URL=http://127.0.0.1:8888/v1 \
MODEL=SubTokenLLM \
./scripts/start_full_strategy_matrix.sh
```

Monitor the active run:

```bash
tail -f "$RUNS_ROOT/matrix.log"
tail -f "$RUNS_ROOT/direct_answer/stdout.log"
```

After completion, `aggregate_summary.json` contains a compact summary across all strategies.

To automatically upload a completed matrix run back to GitHub:

```bash
REMOTE_RUN_ROOT=/data/benwulab/gemma4-eval/runs/full-strategy-matrix-YYYYMMDD_HHMMSS \
REMOTE_PID=<pid> \
./scripts/wait_collect_push_results.sh
```

The uploader waits for `aggregate_summary.json`, rsyncs the completed remote run into `results/`, compresses `predictions.jsonl` files, writes `upload_manifest.json`, commits, and pushes to `main`.

## Public HTTPS Route

E4B and E2B run on `benwulab-remote:8888` and `:8889`. The body-preserving router on `:8890` dispatches by model ID. A reverse SSH tunnel exposes the router at `127.0.0.1:25570` on Tang.

Install `ops/nginx_llm_agaii_org.conf` on Tang in the active BT-panel Nginx vhost tree, then reload Nginx. The Nginx route only rewrites `/llm/v1/*` to `/v1/*` and does not alter request bodies or inject prompts.
