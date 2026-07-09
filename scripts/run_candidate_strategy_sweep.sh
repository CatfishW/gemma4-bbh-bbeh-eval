#!/usr/bin/env bash
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-/data/benwulab/gemma4-eval/datasets}"
RUNS_ROOT="${RUNS_ROOT:-/data/benwulab/gemma4-eval/runs/candidate-strategy-sweep-$(date +%Y%m%d_%H%M%S)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${MODEL:-SubTokenLLM}"
BENCHMARKS="${BENCHMARKS:-bbh,bbeh,usr}"
LIMIT_PER_TASK="${LIMIT_PER_TASK:-25}"
PARALLEL="${PARALLEL:-2}"
TIMEOUT="${TIMEOUT:-300}"
RETRIES="${RETRIES:-2}"
REPO_REVISION="${REPO_REVISION:-$(git rev-parse HEAD 2>/dev/null || true)}"

mkdir -p "$RUNS_ROOT"

cat > "$RUNS_ROOT/candidate_manifest.jsonl" <<EOF
{"name":"direct_answer","prompt_strategy":"direct_answer","self_consistency_k":1,"temperature":0.0,"max_tokens":64}
{"name":"native_format","prompt_strategy":"native_format","self_consistency_k":1,"temperature":0.0,"max_tokens":96}
{"name":"canonical_short","prompt_strategy":"canonical_short","self_consistency_k":1,"temperature":0.0,"max_tokens":96}
{"name":"private_verify","prompt_strategy":"private_verify","self_consistency_k":1,"temperature":0.0,"max_tokens":96}
{"name":"option_elimination","prompt_strategy":"option_elimination","self_consistency_k":1,"temperature":0.0,"max_tokens":96}
{"name":"answer_type_router","prompt_strategy":"answer_type_router","self_consistency_k":1,"temperature":0.0,"max_tokens":96}
{"name":"careful_direct","prompt_strategy":"careful_direct","self_consistency_k":1,"temperature":0.0,"max_tokens":96}
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
  "limit_per_task": "$LIMIT_PER_TASK",
  "parallel": "$PARALLEL",
  "timeout": "$TIMEOUT",
  "retries": "$RETRIES"
}
EOF

run_candidate() {
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
  --limit-per-task "$LIMIT_PER_TASK" \\
  --prompt-strategy "$prompt_strategy" \\
  --self-consistency-k "$self_consistency_k" \\
  --temperature "$temperature" \\
  --max-tokens "$max_tokens" \\
  --parallel "$PARALLEL" \\
  --timeout "$TIMEOUT" \\
  --retries "$RETRIES" \\
  --output-dir "$out_dir"
EOF
  echo "[$(date -Is)] START $name" | tee -a "$RUNS_ROOT/sweep.log"
  bash "$out_dir/command.txt" > "$out_dir/stdout.log" 2> "$out_dir/stderr.log"
  echo "[$(date -Is)] DONE $name" | tee -a "$RUNS_ROOT/sweep.log"
}

while IFS= read -r row; do
  name="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["name"])' <<<"$row")"
  prompt_strategy="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["prompt_strategy"])' <<<"$row")"
  self_consistency_k="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["self_consistency_k"])' <<<"$row")"
  temperature="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["temperature"])' <<<"$row")"
  max_tokens="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["max_tokens"])' <<<"$row")"
  run_candidate "$name" "$prompt_strategy" "$self_consistency_k" "$temperature" "$max_tokens"
done < "$RUNS_ROOT/candidate_manifest.jsonl"

python3 scripts/summarize_strategy_runs.py "$RUNS_ROOT" > "$RUNS_ROOT/aggregate_summary.json"
echo "[$(date -Is)] candidate sweep complete: $RUNS_ROOT" | tee -a "$RUNS_ROOT/sweep.log"
