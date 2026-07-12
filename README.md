# Gemma 4 E2B/E4B Reasoning Evaluation

Small, auditable harness for evaluating the Gemma 4 E2B and E4B deployments against:

- BBH from `suzgunmirac/BIG-Bench-Hard`
- BBEH from `google-deepmind/bbeh`
- USR from `google-deepmind/unpuzzles_and_simple_reasoning`

The evaluator uses the OpenAI-compatible chat completions API and intentionally sends no system message. Each request contains exactly one `user` message.

Confirmatory-study materials:

- [E2B preregistered protocol](docs/E2B_CONFIRMATORY_PROTOCOL.md)
- [Machine-readable protocol](experiments/e2b_confirmatory_protocol.json)
- [Frozen 29-arm manifest](experiments/e2b_arm_manifest.jsonl)
- [Data, model, scoring, and validity statement](docs/DATA_MODEL_AND_EVALUATION_STATEMENT.md)
- [Paper-oriented study summary](paper/e2b-e4b-study/manuscript.md)
- [E2B confirmatory result bundle](results/e2b-confirmatory-20260709_231405)
- [Question, ground truth, direct, and CBRR examples](results/e2b-confirmatory-20260709_231405/analysis/examples.md)
- [Paper bibliography](paper/references.bib)

## Confirmatory Result

The frozen E2B test contains 9,550 examples. The conservative Bayesian reward
router (CBRR) is an offline task-conditioned contextual-bandit policy fitted on
calibration rows; model weights are unchanged.

| E2B arm | Correct | Accuracy | Delta vs direct | Mean completion tokens |
|---|---:|---:|---:|---:|
| `direct_answer` | 2,520/9,550 | 26.39% | - | 14.68 |
| `concise_cot_self_rank_k3` | 3,348/9,550 | 35.06% | +8.67 pp | 681.26 |
| `cbrr_policy` | 3,382/9,550 | 35.41% | +9.03 pp | 65.60 |

CBRR produced 1,100 paired wins and 238 losses versus direct answer. The exact
two-sided McNemar p-value is `1.33e-132`, the Holm-adjusted p-value is
`6.66e-132`, and the task-stratified bootstrap interval is +8.41 to +9.64
percentage points. A task-cluster bootstrap gives +4.19 to +14.59 points; both
additional fixed-seed repeats retain +8.89 to +8.96 points.

### Untouched E2B test by dataset

The test split contains 5,161 BBH, 3,370 BBEH, and 1,019 USR examples.

| Finalist | BBH | BBEH | USR | Overall |
|---|---:|---:|---:|---:|
| `direct_answer` | 2,102/5,161 (40.73%) | 364/3,370 (10.80%) | 54/1,019 (5.30%) | 2,520/9,550 (26.39%) |
| `private_verify` | 1,585/5,161 (30.71%) | 315/3,370 (9.35%) | 86/1,019 (8.44%) | 1,986/9,550 (20.80%) |
| `condition_reconstruction` | 1,689/5,161 (32.73%) | 348/3,370 (10.33%) | 32/1,019 (3.14%) | 2,069/9,550 (21.66%) |
| `concise_cot_self_rank_k3` | 2,958/5,161 (57.31%) | 347/3,370 (10.30%) | 43/1,019 (4.22%) | 3,348/9,550 (35.06%) |
| `cbrr_policy` | **2,959/5,161 (57.33%)** | 360/3,370 (10.68%) | 63/1,019 (6.18%) | **3,382/9,550 (35.41%)** |
| `e4b_policy_transfer` | 2,095/5,161 (40.59%) | 347/3,370 (10.30%) | 54/1,019 (5.30%) | 2,496/9,550 (26.14%) |

CBRR's effect is concentrated in BBH (+16.61 points). BBEH is flat to slightly
negative (-0.12 points), and USR improves modestly (+0.88 points). CBRR uses
about 4.5 times the completion tokens of direct answer, although it slightly
outperforms self-ranking with about one tenth of the completion tokens.

### Matched validation by model and dataset

The matched validation split contains 675 BBH, 575 BBEH, and 240 USR examples.
CBRR is task-conditioned, so its rows are shown separately from the universal
prompt matrix. E4B results are exploratory because E4B informed strategy
development.

| Model and strategy | BBH | BBEH | USR | Overall |
|---|---:|---:|---:|---:|
| E2B `direct_answer` | 40.74% | 10.43% | 15.83% | 25.03% |
| E2B `concise_cot_self_rank_k3` | 59.11% | 11.30% | 17.50% | 33.96% |
| E2B `cbrr_policy` | 57.78% | 11.48% | 16.67% | 33.29% |
| E4B `direct_answer` | 50.52% | 14.61% | 25.00% | 32.55% |
| E4B `concise_cot_self_rank_k3` | 65.93% | 15.30% | 20.83% | 39.13% |
| E4B `cbrr_policy` | **67.56%** | **16.35%** | **29.17%** | **41.61%** |

<details>
<summary>All 29 universal arms by model and dataset</summary>

| Strategy | E2B BBH | E2B BBEH | E2B USR | E2B all | E4B BBH | E4B BBEH | E4B USR | E4B all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `concise_cot_self_rank_k3` | **59.11%** | 11.30% | **17.50%** | **33.96%** | **65.93%** | **15.30%** | 20.83% | **39.13%** |
| `canonical_short` | 45.33% | **11.48%** | 13.33% | 27.11% | 55.26% | 13.74% | 20.00% | 33.56% |
| `direct_answer` | 40.74% | 10.43% | 15.83% | 25.03% | 50.52% | 14.61% | **25.00%** | 32.55% |
| `option_elimination` | 41.63% | 9.22% | 9.58% | 23.96% | 52.74% | 11.30% | 19.17% | 31.34% |
| `compare_then_commit` | 35.26% | 10.96% | 13.33% | 22.35% | 53.48% | 14.09% | 18.75% | 32.68% |
| `counterexample_guard` | 35.85% | 9.39% | 13.33% | 22.01% | 52.59% | 14.78% | 18.33% | 32.48% |
| `native_format` | 38.37% | 7.83% | 15.00% | 22.82% | 52.59% | 11.30% | 20.42% | 31.48% |
| `rank_two_paths` | 34.67% | 10.43% | 13.33% | 21.88% | 52.44% | 14.09% | 18.75% | 32.21% |
| `selective_verify` | 37.48% | 8.35% | 14.58% | 22.55% | 51.11% | 14.09% | 17.92% | 31.48% |
| `private_verify` | 29.93% | 10.61% | 12.92% | 19.73% | 55.70% | 13.91% | 22.50% | 34.23% |
| `constraint_guard` | 33.04% | 10.61% | 11.67% | 20.94% | 54.07% | 14.09% | 18.33% | 32.89% |
| `careful_direct` | 31.11% | 8.70% | 14.17% | 19.73% | 51.85% | 14.09% | 17.50% | 31.74% |
| `condition_reconstruction` | 32.59% | 9.91% | 13.33% | 20.74% | 50.96% | 12.52% | 17.50% | 30.74% |
| `answer_type_router` | 36.15% | 9.91% | 10.83% | 21.95% | 46.67% | 13.74% | 10.42% | 28.12% |
| `chain_of_draft` | 34.22% | 2.78% | 6.67% | 17.65% | 54.22% | 3.48% | 17.50% | 28.72% |
| `concise_cot_sc_k3` | 38.81% | 2.26% | 5.00% | 19.26% | 50.22% | 3.30% | 7.92% | 25.30% |
| `concise_cot` | 35.41% | 1.57% | 4.58% | 17.38% | 46.67% | 3.83% | 10.42% | 24.30% |
| `premise_conclusion` | 32.74% | 2.09% | 5.42% | 16.51% | 47.56% | 2.09% | 8.75% | 23.76% |
| `draft_verify` | 15.26% | 6.61% | 5.42% | 10.34% | 42.81% | 14.09% | 17.08% | 27.58% |
| `negation_label_guard` | 23.56% | 4.00% | 2.50% | 12.62% | 33.78% | 8.70% | 9.17% | 20.13% |
| `direct_key_condition_refine` | 31.70% | 5.91% | 14.58% | 18.99% | 19.26% | 5.04% | 13.33% | 12.82% |
| `fast_slow_gate` | 13.04% | 4.17% | 6.25% | 8.52% | 28.74% | 8.52% | 17.08% | 19.06% |
| `strict_json` | 14.07% | 4.87% | 1.67% | 8.52% | 20.44% | 7.13% | 3.75% | 12.62% |
| `step_back` | 8.15% | 0.35% | 1.67% | 4.09% | 18.81% | 0.35% | 1.67% | 8.93% |
| `plan_and_solve` | 8.89% | 0.00% | 0.42% | 4.09% | 12.74% | 0.00% | 1.25% | 5.97% |
| `least_to_most` | 4.00% | 0.00% | 0.00% | 1.81% | 7.70% | 0.17% | 1.25% | 3.76% |
| `raw` | 1.48% | 3.13% | 0.83% | 2.01% | 0.74% | 4.00% | 2.08% | 2.21% |
| `symbolic_proof` | 4.00% | 0.00% | 0.00% | 1.81% | 4.44% | 0.17% | 0.83% | 2.21% |
| `plan_and_solve_plus` | 2.81% | 0.00% | 0.00% | 1.28% | 4.15% | 0.17% | 0.83% | 2.08% |

</details>

See the
[confirmatory report](results/e2b-confirmatory-20260709_231405/analysis/report.md),
[cluster sensitivity](paper/e2b-e4b-study/cluster-robustness/cluster_robustness.md),
[strict JSON audit](paper/e2b-e4b-study/format-audit/format_audit.md),
[cross-model CSV](paper/e2b-e4b-study/cross_model_screening.csv), and
[post-hoc direct-fallback replay](paper/e2b-e4b-study/budget-sensitivity/fallback_replay_sensitivity.md).

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

Deployment verification on 2026-07-10 found both model services, the router, and
the public tunnel enabled and active under user systemd with zero restarts. See
the [deployment snapshot](paper/e2b-e4b-study/deployment-verification-20260710.md).

The gateway dispatches by `model` while forwarding the original request body unchanged.
It does not inject a system prompt. Direct remote endpoints:

```bash
curl http://127.0.0.1:8888/v1/models
curl http://127.0.0.1:8889/v1/models
```

The production model servers, router, and public tunnel have tracked user-systemd
units under `ops/systemd/`. Install and enable them without interrupting a live
deployment with:

```bash
./ops/install_systemd_services.sh
```

The units are started only after any legacy port-owning processes have been stopped.
User lingering must remain enabled so the deployment starts without an interactive
login.

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
- USR: `39bc520a2f4c243eb04ce1cc27f28c7c61d12e42`

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
