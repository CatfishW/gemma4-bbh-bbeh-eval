#!/bin/bash
# Separate Gemma 4 native-thinking BBEH reproduction and frozen adapter eval.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/data/benwulab/gemma4-rl/env/bin/python}
DATA=${DATA:-/data/benwulab/gemma4-eval/datasets}
RUNS=${RUNS:-/data/benwulab/gemma4-rl/runs}
OUT=${OUT:-$RUNS/evals-official-thinking/v3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

GRPO_ADAPTER=${GRPO_ADAPTER:-$(tr -d '\r\n' < "$RUNS/evals/grpo_selected_adapter.txt")}
VOLT_ADAPTER=${VOLT_ADAPTER:-$(tr -d '\r\n' < "$RUNS/evals/volt_selected_adapter.txt")}

evaluate() {
  local label="$1" scope="$2" output="$3" adapter="$4"
  if [ -f "$output/summary.json" ]; then
    echo "skip $output (complete summary exists)"
    return 0
  fi
  local resume_args=()
  local adapter_args=()
  if [ -f "$output/run_config.json" ]; then
    resume_args=(--resume)
  fi
  if [ -n "$adapter" ]; then
    adapter_args=(--adapter "$adapter")
  fi
  echo "=== native-thinking eval $label $scope start $(date --iso-8601=seconds) ==="
  "$PY" rl/eval_official_thinking.py \
    --datasets-root "$DATA" \
    --model-label "$label" \
    --scope "$scope" \
    --batch-size 4 \
    --max-batch-tokens 49152 \
    --max-new-tokens 8192 \
    --output-dir "$output" \
    "${adapter_args[@]}" \
    "${resume_args[@]}"
  echo "=== native-thinking eval $label $scope done $(date --iso-8601=seconds) ==="
}

mkdir -p "$OUT"
evaluate base all "$OUT/base-all" ""
evaluate grpo frozen_test "$OUT/grpo-frozen-test" "$GRPO_ADAPTER"
evaluate volt frozen_test "$OUT/volt-frozen-test" "$VOLT_ADAPTER"

"$PY" scripts/compare_official_thinking_evals.py \
  --base "$OUT/base-all/predictions.jsonl" \
  --grpo "$OUT/grpo-frozen-test/predictions.jsonl" \
  --volt "$OUT/volt-frozen-test/predictions.jsonl" \
  --output-dir "$OUT"

echo "=== ALL NATIVE-THINKING EVALS DONE $(date --iso-8601=seconds) ==="
