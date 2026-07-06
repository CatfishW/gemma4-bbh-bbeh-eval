#!/usr/bin/env bash
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-/data/benwulab/gemma4-eval/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/data/benwulab/gemma4-eval/runs/full-strategy-matrix-$(date +%Y%m%d_%H%M%S)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${MODEL:-SubTokenLLM}"
BENCHMARKS="${BENCHMARKS:-bbh,bbeh}"
PARALLEL="${PARALLEL:-2}"
TIMEOUT="${TIMEOUT:-300}"
RETRIES="${RETRIES:-2}"
REPO_REVISION="${REPO_REVISION:-$(git rev-parse HEAD 2>/dev/null || true)}"

mkdir -p "$RUNS_ROOT"

cat > "$RUNS_ROOT/matrix_manifest.jsonl" <<EOF
{"name":"direct_answer","prompt_strategy":"direct_answer","self_consistency_k":1,"temperature":0.0,"max_tokens":64,"source":"direct answer baseline / option-only answer"}
{"name":"strict_json","prompt_strategy":"strict_json","self_consistency_k":1,"temperature":0.0,"max_tokens":96,"source":"strict JSON schema with answer field"}
{"name":"concise_cot","prompt_strategy":"concise_cot","self_consistency_k":1,"temperature":0.0,"max_tokens":256,"source":"concise CoT"}
{"name":"chain_of_draft","prompt_strategy":"chain_of_draft","self_consistency_k":1,"temperature":0.0,"max_tokens":192,"source":"Chain-of-Draft"}
{"name":"plan_and_solve","prompt_strategy":"plan_and_solve","self_consistency_k":1,"temperature":0.0,"max_tokens":256,"source":"Plan-and-Solve"}
{"name":"step_back","prompt_strategy":"step_back","self_consistency_k":1,"temperature":0.0,"max_tokens":256,"source":"Step-Back"}
{"name":"premise_conclusion","prompt_strategy":"premise_conclusion","self_consistency_k":1,"temperature":0.0,"max_tokens":256,"source":"premise-to-conclusion"}
{"name":"symbolic_proof","prompt_strategy":"symbolic_proof","self_consistency_k":1,"temperature":0.0,"max_tokens":256,"source":"symbolic translation / proof sketch"}
{"name":"concise_cot_sc_k3","prompt_strategy":"concise_cot","self_consistency_k":3,"temperature":0.7,"max_tokens":256,"source":"self-consistency k=3 with normalized majority vote"}
EOF

cat > "$RUNS_ROOT/environment.json" <<EOF
{
  "created_at": "$(date -Is)",
  "hostname": "$(hostname)",
  "repo_dir": "$(pwd)",
  "repo_revision": "$REPO_REVISION",
  "datasets_root": "$DATASETS_ROOT",
  "base_url": "$BASE_URL",
  "model": "$MODEL",
  "benchmarks": "$BENCHMARKS",
  "parallel": "$PARALLEL",
  "timeout": "$TIMEOUT",
  "retries": "$RETRIES"
}
EOF

run_strategy() {
  local name="$1"
  local prompt_strategy="$2"
  local self_consistency_k="$3"
  local temperature="$4"
  local max_tokens="$5"
  local out_dir="$RUNS_ROOT/$name"
  mkdir -p "$out_dir"
  cat > "$out_dir/command.txt" <<EOF
python3 eval_benchmarks.py \\
  --datasets-root "$DATASETS_ROOT" \\
  --base-url "$BASE_URL" \\
  --model "$MODEL" \\
  --benchmarks "$BENCHMARKS" \\
  --prompt-strategy "$prompt_strategy" \\
  --self-consistency-k "$self_consistency_k" \\
  --temperature "$temperature" \\
  --max-tokens "$max_tokens" \\
  --parallel "$PARALLEL" \\
  --timeout "$TIMEOUT" \\
  --retries "$RETRIES" \\
  --output-dir "$out_dir"
EOF
  echo "[$(date -Is)] START $name" | tee -a "$RUNS_ROOT/matrix.log"
  bash "$out_dir/command.txt" > "$out_dir/stdout.log" 2> "$out_dir/stderr.log"
  echo "[$(date -Is)] DONE $name" | tee -a "$RUNS_ROOT/matrix.log"
}

while IFS= read -r row; do
  name="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["name"])' <<<"$row")"
  prompt_strategy="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["prompt_strategy"])' <<<"$row")"
  self_consistency_k="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["self_consistency_k"])' <<<"$row")"
  temperature="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["temperature"])' <<<"$row")"
  max_tokens="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["max_tokens"])' <<<"$row")"
  run_strategy "$name" "$prompt_strategy" "$self_consistency_k" "$temperature" "$max_tokens"
done < "$RUNS_ROOT/matrix_manifest.jsonl"

python3 scripts/summarize_strategy_runs.py "$RUNS_ROOT" > "$RUNS_ROOT/aggregate_summary.json"
echo "[$(date -Is)] all strategies complete: $RUNS_ROOT" | tee -a "$RUNS_ROOT/matrix.log"
