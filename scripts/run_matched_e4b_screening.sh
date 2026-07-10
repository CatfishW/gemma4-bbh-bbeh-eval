#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/benwulab/gemma4-eval/datasets}"
RUNS_BASE="${RUNS_BASE:-/data/benwulab/gemma4-eval/runs}"
RUN_ROOT="${RUN_ROOT:-$RUNS_BASE/e4b-matched-screening-$(date +%Y%m%d_%H%M%S)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${MODEL:-SubTokenLLM}"
PARALLEL="${PARALLEL:-8}"
TIMEOUT="${TIMEOUT:-300}"
RETRIES="${RETRIES:-2}"
SEED="${SEED:-20260709}"
MANIFEST="$REPO_DIR/experiments/e2b_arm_manifest.jsonl"
E4B_POLICY="$REPO_DIR/experiments/e4b_reward_routed_v2_policy.json"
LOG="$RUN_ROOT/screening.log"

mkdir -p "$RUN_ROOT/arms" "$RUN_ROOT/selection"

status() {
  python3 - "$RUN_ROOT/status.json" "$1" "$2" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
Path(sys.argv[1]).write_text(json.dumps({
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "state": sys.argv[2],
    "detail": sys.argv[3],
}, indent=2) + "\n")
PY
}

trap 'rc=$?; if [[ $rc -ne 0 ]]; then status failed "exit code $rc"; fi' EXIT

run_arm() {
  local name="$1" prompt="$2" k="$3" temperature="$4" max_tokens="$5" \
    selection="$6" selection_tokens="$7"
  local out="$RUN_ROOT/arms/$name"
  mkdir -p "$out"
  if [[ -f "$out/summary.json" && ( -f "$out/predictions.jsonl" || -f "$out/predictions.jsonl.gz" ) ]]; then
    printf '[%s] SKIP %s\n' "$(date -Is)" "$name" | tee -a "$LOG"
    return
  fi
  local cmd=(
    python3 "$REPO_DIR/eval_benchmarks.py"
    --datasets-root "$DATASETS_ROOT"
    --base-url "$BASE_URL"
    --model "$MODEL"
    --benchmarks bbh,bbeh,usr
    --limit-per-task 50
    --prompt-strategy "$prompt"
    --self-consistency-k "$k"
    --temperature "$temperature"
    --max-tokens "$max_tokens"
    --response-selection "$selection"
    --selection-max-tokens "$selection_tokens"
    --seed "$SEED"
    --parallel "$PARALLEL"
    --timeout "$TIMEOUT"
    --retries "$RETRIES"
    --output-dir "$out"
  )
  printf '%q ' "${cmd[@]}" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  printf '[%s] START %s\n' "$(date -Is)" "$name" | tee -a "$LOG"
  "${cmd[@]}" > "$out/stdout.log" 2> "$out/stderr.log"
  printf '[%s] DONE %s\n' "$(date -Is)" "$name" | tee -a "$LOG"
}

cd "$REPO_DIR"
status screening "matched exploratory E4B screening over all registered arms"
curl -fsS "$BASE_URL/models" > "$RUN_ROOT/models.json"
cp "$MANIFEST" "$RUN_ROOT/arm_manifest.jsonl"
cat > "$RUN_ROOT/environment.json" <<EOF
{
  "created_at": "$(date -Is)",
  "study_role": "exploratory_matched_model_scale_replication",
  "hostname": "$(hostname)",
  "repo_revision": "$(git rev-parse HEAD)",
  "model": "$MODEL",
  "base_url": "$BASE_URL",
  "seed": $SEED,
  "parallel": $PARALLEL,
  "split": "indices below 50 only",
  "system_messages_sent": 0
}
EOF
nvidia-smi -q > "$RUN_ROOT/nvidia-smi-q.txt"

python3 - "$MANIFEST" <<'PY' |
import json
from pathlib import Path
import sys
for line in Path(sys.argv[1]).read_text().splitlines():
    row = json.loads(line)
    print("\t".join(str(row[key]) for key in (
        "name", "prompt_strategy", "self_consistency_k", "temperature", "max_tokens",
        "response_selection", "selection_max_tokens",
    )))
PY
while IFS=$'\t' read -r name prompt k temperature max_tokens selection selection_tokens; do
  run_arm "$name" "$prompt" "$k" "$temperature" "$max_tokens" "$selection" "$selection_tokens"
done

python3 "$REPO_DIR/scripts/summarize_strategy_runs.py" "$RUN_ROOT/arms" \
  > "$RUN_ROOT/aggregate_summary.json"
status selecting "fit exploratory E4B CBRR using the matched calibration and validation split"
python3 "$REPO_DIR/scripts/select_e2b_finalists.py" \
  --runs-root "$RUN_ROOT/arms" \
  --manifest "$MANIFEST" \
  --e4b-policy "$E4B_POLICY" \
  --policy-name e4b_matched_cbrr_exploratory \
  --study-label E4B \
  --output-dir "$RUN_ROOT/selection" \
  > "$RUN_ROOT/selection/stdout.log" 2> "$RUN_ROOT/selection/stderr.log"

find "$RUN_ROOT" -type f -name predictions.jsonl -print0 |
  while IFS= read -r -d '' path; do gzip -f "$path"; done
python3 - "$RUN_ROOT" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
root = Path(sys.argv[1])
files = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path.name in {"artifact_manifest.json", "status.json"}:
        continue
    files.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
(root / "artifact_manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
PY
status complete "matched E4B screening and validation selection complete"
touch "$RUN_ROOT/COMPLETE"
trap - EXIT
