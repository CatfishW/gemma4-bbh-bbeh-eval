#!/bin/bash
# Post-training evaluation sequence for the RL study.
# Selects each run's checkpoint by best validation probe, confirms on the full
# validation split, then evaluates base/GRPO/VOLT on the frozen test split with
# the frozen per-arm decoding limits. Runs sequentially on one GPU.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/data/benwulab/gemma4-rl/env/bin/python}
DATA=${DATA:-/data/benwulab/gemma4-eval/datasets}
RUNS=${RUNS:-/data/benwulab/gemma4-rl/runs}
EVALS="$RUNS/evals"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$EVALS"

best_checkpoint() {
  $PY - "$1" <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
best_iteration, best_accuracy = None, -1.0
for line in (run / "metrics.jsonl").read_text().splitlines():
    row = json.loads(line)
    if "val_probe_accuracy" in row and row["val_probe_accuracy"] > best_accuracy:
        best_accuracy = row["val_probe_accuracy"]
        best_iteration = row["iteration"] + 1
candidates = sorted(run.glob("checkpoint-*/adapter"))
if best_iteration is not None:
    exact = run / f"checkpoint-{best_iteration:04d}" / "adapter"
    if exact.exists():
        print(exact)
        raise SystemExit(0)
print(candidates[-1] if candidates else "")
EOF
}

evaluate() {
  local name="$1" split="$2" strategy="$3" adapter="$4"
  local out="$EVALS/$name-$split-$strategy"
  if [ -f "$out/summary.json" ]; then
    echo "skip $out (exists)"
    return 0
  fi
  local adapter_args=()
  if [ -n "$adapter" ]; then
    adapter_args=(--adapter "$adapter")
  fi
  echo "=== eval $name $split $strategy start $(date) ==="
  $PY rl/eval_policy.py \
    --datasets-root "$DATA" \
    --split "$split" \
    --prompt-strategy "$strategy" \
    --batch-size 32 \
    --output-dir "$out" \
    "${adapter_args[@]}"
  echo "=== eval $name $split $strategy done $(date) ==="
}

GRPO_ADAPTER=$(best_checkpoint "$RUNS/grpo-e2b")
VOLT_ADAPTER=$(best_checkpoint "$RUNS/volt-e2b")
echo "grpo adapter: $GRPO_ADAPTER"
echo "volt adapter: $VOLT_ADAPTER"
echo "$GRPO_ADAPTER" > "$EVALS/grpo_selected_adapter.txt"
echo "$VOLT_ADAPTER" > "$EVALS/volt_selected_adapter.txt"

# Full-validation confirmation of the probe-selected checkpoints.
evaluate grpo validation concise_cot "$GRPO_ADAPTER"
evaluate volt validation concise_cot "$VOLT_ADAPTER"
evaluate base validation concise_cot ""

# Frozen test (single shot per cell). Evaluate the training-matched
# concise-CoT condition first so the primary GRPO/VOLT comparison lands before
# the prompt-transfer direct-answer checks.
evaluate base test concise_cot ""
evaluate grpo test concise_cot "$GRPO_ADAPTER"
evaluate volt test concise_cot "$VOLT_ADAPTER"
evaluate base test direct_answer ""
evaluate grpo test direct_answer "$GRPO_ADAPTER"
evaluate volt test direct_answer "$VOLT_ADAPTER"

echo "=== ALL EVALS DONE $(date) ==="
