# Gemma 4 E4B BBH/BBEH Evaluation

Small, auditable harness for evaluating the `SubTokenLLM` Gemma 4 E4B deployment against:

- BBH from `suzgunmirac/BIG-Bench-Hard`
- BBEH from `google-deepmind/bbeh`
- USR from `google-deepmind/unpuzzles_and_simple_reasoning`

The evaluator uses the OpenAI-compatible chat completions API and intentionally sends no system message. Each request contains exactly one `user` message.

## Access

Once the Tang Nginx route is installed, use:

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
    model="SubTokenLLM",
    messages=[{"role": "user", "content": "Reply with exactly: online"}],
    max_completion_tokens=16,
    temperature=0,
)
print(response.choices[0].message.content)
```

Current direct remote endpoint during setup:

```bash
curl http://127.0.0.1:8888/v1/models
```

Tang tunnel endpoint after `ops/tunnel_tang_25570.sh` is running on `benwulab-remote`:

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
- `concise_cot`: concise reasoning with a final-answer delimiter.
- `chain_of_draft`: terse scratch notes with a final-answer delimiter.
- `plan_and_solve`: short plan, solve, final-answer delimiter.
- `step_back`: identify the general principle, then solve.
- `premise_conclusion`: explicit premise-to-conclusion template.
- `symbolic_proof`: compact symbolic translation or proof sketch, then solve.
- `raw`: dataset input only.

Self-consistency is enabled with `--self-consistency-k`. The evaluator records every generation and chooses the majority normalized final answer. It still sends no system message.

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

The Gemma service runs on `benwulab-remote:8888`. A reverse SSH tunnel from `benwulab-remote` to `tang-server-org` exposes it at `127.0.0.1:25570` on Tang.

Install `ops/nginx_llm_agaii_org.conf` on Tang in the active BT-panel Nginx vhost tree, then reload Nginx. The Nginx route only rewrites `/llm/v1/*` to `/v1/*` and does not alter request bodies or inject prompts.
