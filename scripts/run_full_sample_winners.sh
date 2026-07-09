#!/usr/bin/env bash
set -euo pipefail

SAMPLE_ROOT="${SAMPLE_ROOT:?set SAMPLE_ROOT to the completed candidate sweep root}"
FULL_ROOT="${FULL_ROOT:-/data/benwulab/gemma4-eval/runs/full-challenger-winners-$(date +%Y%m%d_%H%M%S)}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/benwulab/gemma4-eval/datasets}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${MODEL:-SubTokenLLM}"
BENCHMARKS="${BENCHMARKS:-bbh,bbeh,usr}"
PARALLEL="${PARALLEL:-2}"
TIMEOUT="${TIMEOUT:-300}"
RETRIES="${RETRIES:-2}"
REPO_REVISION="${REPO_REVISION:-$(git rev-parse HEAD 2>/dev/null || true)}"

while [ ! -f "$SAMPLE_ROOT/aggregate_summary.json" ]; do
  echo "[$(date -Is)] waiting for sample sweep aggregate: $SAMPLE_ROOT/aggregate_summary.json"
  sleep 120
done

mkdir -p "$FULL_ROOT"

python3 - "$SAMPLE_ROOT" "$FULL_ROOT/selected_strategies.json" <<'PY'
import json
from pathlib import Path
import sys

sample = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []

for summary_path in sorted(sample.glob("*/summary.json")):
    summary = json.loads(summary_path.read_text())
    correct = sum(item["correct"] for item in summary["benchmarks"])
    total = sum(item["total"] for item in summary["benchmarks"])
    rows.append(
        {
            "name": summary_path.parent.name,
            "prompt_strategy": summary["prompt_strategy"],
            "max_tokens": summary["max_tokens"],
            "temperature": summary["temperature"],
            "self_consistency_k": summary["self_consistency_k"],
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }
    )

baseline = next(item for item in rows if item["name"] == "direct_answer")
winners = [
    item
    for item in rows
    if item["name"] != "direct_answer" and item["accuracy"] > baseline["accuracy"]
]
if not winners:
    winners = sorted(
        (item for item in rows if item["name"] != "direct_answer"),
        key=lambda item: item["accuracy"],
        reverse=True,
    )[:2]

payload = {
    "baseline": baseline,
    "selected": winners,
    "all": sorted(rows, key=lambda item: item["accuracy"], reverse=True),
}
out.write_text(json.dumps(payload, indent=2) + "\n")
PY

cat > "$FULL_ROOT/environment.json" <<EOF
{
  "created_at": "$(date -Is)",
  "hostname": "$(hostname)",
  "repo_dir": "$(pwd)",
  "repo_revision": "$REPO_REVISION",
  "sample_root": "$SAMPLE_ROOT",
  "datasets_root": "$DATASETS_ROOT",
  "base_url": "$BASE_URL",
  "model": "$MODEL",
  "benchmarks": "$BENCHMARKS",
  "parallel": "$PARALLEL",
  "timeout": "$TIMEOUT",
  "retries": "$RETRIES"
}
EOF

python3 - "$FULL_ROOT/selected_strategies.json" <<'PY' |
import json
import sys

payload = json.load(open(sys.argv[1]))
for item in payload["selected"]:
    print(json.dumps(item))
PY
while IFS= read -r row; do
  name="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["name"])' <<<"$row")"
  prompt_strategy="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["prompt_strategy"])' <<<"$row")"
  max_tokens="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["max_tokens"])' <<<"$row")"
  temperature="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["temperature"])' <<<"$row")"
  self_consistency_k="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["self_consistency_k"])' <<<"$row")"
  out_dir="$FULL_ROOT/$name"
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
  echo "[$(date -Is)] START $name" | tee -a "$FULL_ROOT/full_challenger.log"
  bash "$out_dir/command.txt" > "$out_dir/stdout.log" 2> "$out_dir/stderr.log"
  echo "[$(date -Is)] DONE $name" | tee -a "$FULL_ROOT/full_challenger.log"
done

python3 scripts/summarize_strategy_runs.py "$FULL_ROOT" > "$FULL_ROOT/aggregate_summary.json"
echo "[$(date -Is)] full challenger winners complete: $FULL_ROOT" | tee -a "$FULL_ROOT/full_challenger.log"
