# Gemma 4 E4B BBH/BBEH Evaluation

Small, auditable harness for evaluating the `SubTokenLLM` Gemma 4 E4B deployment against:

- BBH from `suzgunmirac/BIG-Bench-Hard`
- BBEH from `google-deepmind/bbeh`

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

Use `--prompt-mode raw` if you want to send the dataset input exactly as stored. The default `answer_only` mode appends a user-level instruction asking the model to output only the final answer.

## Public HTTPS Route

The Gemma service runs on `benwulab-remote:8888`. A reverse SSH tunnel from `benwulab-remote` to `tang-server-org` exposes it at `127.0.0.1:25570` on Tang.

Install `ops/nginx_llm_agaii_org.conf` on Tang in the active BT-panel Nginx vhost tree, then reload Nginx. The Nginx route only rewrites `/llm/v1/*` to `/v1/*` and does not alter request bodies or inject prompts.

