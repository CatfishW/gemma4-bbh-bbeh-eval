#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/benwulab/gemma4-eval/datasets}"
RUNS_BASE="${RUNS_BASE:-/data/benwulab/gemma4-eval/runs}"
RUN_ROOT="${RUN_ROOT:-$RUNS_BASE/e2b-confirmatory-$(date +%Y%m%d_%H%M%S)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8889/v1}"
MODEL="${MODEL:-SubTokenLLM-E2B}"
BENCHMARKS="${BENCHMARKS:-bbh,bbeh,usr}"
PARALLEL="${PARALLEL:-4}"
TIMEOUT="${TIMEOUT:-300}"
RETRIES="${RETRIES:-2}"
PRIMARY_SEED="${PRIMARY_SEED:-20260709}"
MANIFEST="$REPO_DIR/experiments/e2b_arm_manifest.jsonl"
PROTOCOL="$REPO_DIR/experiments/e2b_confirmatory_protocol.json"
E4B_POLICY="$REPO_DIR/experiments/e4b_reward_routed_v2_policy.json"
PREREG_COMMIT="${PREREG_COMMIT:-$(git -C "$REPO_DIR" rev-parse HEAD)}"
LOG="$RUN_ROOT/experiment.log"

mkdir -p "$RUN_ROOT" "$RUN_ROOT/screening" "$RUN_ROOT/selection" \
  "$RUN_ROOT/test/seed-$PRIMARY_SEED" "$RUN_ROOT/robustness" "$RUN_ROOT/analysis"

write_status() {
  local state="$1"
  local detail="$2"
  python3 - "$RUN_ROOT/status.json" "$state" "$detail" <<'PY'
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

on_exit() {
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    write_status "failed" "experiment exited with code $rc"
    printf '[%s] FAILED rc=%s\n' "$(date -Is)" "$rc" | tee -a "$LOG"
  fi
}
trap on_exit EXIT

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"
}

prediction_exists() {
  [[ -f "$1/predictions.jsonl" || -f "$1/predictions.jsonl.gz" ]]
}

run_eval() {
  local phase="$1"
  local name="$2"
  local prompt_strategy="$3"
  local self_consistency_k="$4"
  local temperature="$5"
  local max_tokens="$6"
  local response_selection="$7"
  local selection_max_tokens="$8"
  local seed="$9"
  local limit_per_task="${10}"
  local skip_per_task="${11}"
  local policy_path="${12}"
  local out_dir="$RUN_ROOT/$phase/$name"
  mkdir -p "$out_dir"
  if [[ -f "$out_dir/summary.json" ]] && prediction_exists "$out_dir"; then
    log "SKIP completed $phase/$name"
    return
  fi

  local cmd=(
    python3 "$REPO_DIR/eval_benchmarks.py"
    --datasets-root "$DATASETS_ROOT"
    --base-url "$BASE_URL"
    --model "$MODEL"
    --benchmarks "$BENCHMARKS"
    --prompt-strategy "$prompt_strategy"
    --self-consistency-k "$self_consistency_k"
    --temperature "$temperature"
    --max-tokens "$max_tokens"
    --response-selection "$response_selection"
    --selection-max-tokens "$selection_max_tokens"
    --seed "$seed"
    --parallel "$PARALLEL"
    --timeout "$TIMEOUT"
    --retries "$RETRIES"
    --output-dir "$out_dir"
  )
  if [[ -n "$limit_per_task" ]]; then
    cmd+=(--limit-per-task "$limit_per_task")
  fi
  if [[ -n "$skip_per_task" ]]; then
    cmd+=(--skip-per-task "$skip_per_task")
  fi
  if [[ -n "$policy_path" ]]; then
    cmd+=(--prompt-policy "$policy_path")
  fi
  printf '%q ' "${cmd[@]}" > "$out_dir/command.txt"
  printf '\n' >> "$out_dir/command.txt"
  log "START $phase/$name"
  "${cmd[@]}" > "$out_dir/stdout.log" 2> "$out_dir/stderr.log"
  log "DONE $phase/$name"
}

cd "$REPO_DIR"
write_status "initializing" "validating preregistration and endpoint"

git cat-file -e "$PREREG_COMMIT^{commit}"
git merge-base --is-ancestor "$PREREG_COMMIT" HEAD
for frozen_file in "$MANIFEST" "$PROTOCOL" "$E4B_POLICY"; do
  git diff --quiet "$PREREG_COMMIT" -- "$frozen_file"
done

curl --fail --silent --show-error "$BASE_URL/models" > "$RUN_ROOT/models.json"
python3 - "$RUN_ROOT/models.json" "$MODEL" <<'PY'
import json
from pathlib import Path
import sys

models = json.loads(Path(sys.argv[1]).read_text())
ids = {str(row.get("id")) for row in models.get("data", [])}
if sys.argv[2] not in ids:
    raise SystemExit(f"served model absent: {sys.argv[2]}; available={sorted(ids)}")
PY

cp "$MANIFEST" "$RUN_ROOT/arm_manifest.jsonl"
cp "$PROTOCOL" "$RUN_ROOT/protocol.json"
cp "$E4B_POLICY" "$RUN_ROOT/e4b_transfer_policy.json"
sha256sum "$RUN_ROOT/arm_manifest.jsonl" "$RUN_ROOT/protocol.json" \
  "$RUN_ROOT/e4b_transfer_policy.json" > "$RUN_ROOT/frozen_inputs.sha256"

python3 - "$RUN_ROOT/environment.json" <<PY
import json
from pathlib import Path
from datetime import datetime, timezone

Path("$RUN_ROOT/environment.json").write_text(json.dumps({
    "created_at": datetime.now(timezone.utc).isoformat(),
    "hostname": "$(hostname)",
    "repo_dir": "$REPO_DIR",
    "repo_revision": "$(git rev-parse HEAD)",
    "preregistration_commit": "$PREREG_COMMIT",
    "datasets_root": "$DATASETS_ROOT",
    "base_url": "$BASE_URL",
    "model": "$MODEL",
    "benchmarks": "$BENCHMARKS",
    "parallel": int("$PARALLEL"),
    "timeout": int("$TIMEOUT"),
    "retries": int("$RETRIES"),
    "primary_seed": int("$PRIMARY_SEED"),
    "system_messages_sent": 0,
}, indent=2) + "\n")
PY
nvidia-smi -q > "$RUN_ROOT/nvidia-smi-q.txt"
python3 --version > "$RUN_ROOT/python-version.txt" 2>&1

write_status "screening" "running every preregistered arm on indices below 50"
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
while IFS=$'\t' read -r name prompt_strategy self_consistency_k temperature max_tokens \
  response_selection selection_max_tokens; do
  run_eval "screening" "$name" "$prompt_strategy" "$self_consistency_k" "$temperature" \
    "$max_tokens" "$response_selection" "$selection_max_tokens" "$PRIMARY_SEED" \
    "50" "" ""
done

python3 "$REPO_DIR/scripts/summarize_strategy_runs.py" "$RUN_ROOT/screening" \
  > "$RUN_ROOT/screening/aggregate_summary.json"

write_status "selecting" "freezing validation winner and conservative Bayesian router"
python3 "$REPO_DIR/scripts/select_e2b_finalists.py" \
  --runs-root "$RUN_ROOT/screening" \
  --manifest "$MANIFEST" \
  --e4b-policy "$E4B_POLICY" \
  --output-dir "$RUN_ROOT/selection" \
  > "$RUN_ROOT/selection/stdout.log" \
  2> "$RUN_ROOT/selection/stderr.log"
(cd "$RUN_ROOT/selection" && sha256sum -c selection.sha256)

python3 - "$RUN_ROOT/selection/selection.json" "$MANIFEST" \
  "$RUN_ROOT/test_manifest.jsonl" <<'PY'
import json
from pathlib import Path
import sys

selection = json.loads(Path(sys.argv[1]).read_text())
manifest = {
    row["name"]: row
    for row in (json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines())
}
rows = []
for finalist in selection["finalists"]:
    if finalist["kind"] == "arm":
        row = dict(manifest[finalist["arm"]])
        row["name"] = finalist["name"]
        row["source_arm"] = finalist["arm"]
        row["kind"] = "arm"
        row["policy_path"] = ""
    else:
        row = {
            "name": finalist["name"],
            "source_arm": "direct_answer",
            "kind": "policy",
            "prompt_strategy": "direct_answer",
            "self_consistency_k": 1,
            "temperature": 0.0,
            "max_tokens": finalist["max_tokens"],
            "response_selection": "majority_vote",
            "selection_max_tokens": 64,
            "policy_path": finalist["policy_path"],
        }
    rows.append(row)
Path(sys.argv[3]).write_text("".join(json.dumps(row) + "\n" for row in rows))
PY

write_status "confirmatory_test" "running frozen finalists on untouched indices"
python3 - "$RUN_ROOT/test_manifest.jsonl" <<'PY' |
import json
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text().splitlines():
    row = json.loads(line)
    print("\t".join(str(row.get(key, "")) for key in (
        "name", "prompt_strategy", "self_consistency_k", "temperature", "max_tokens",
        "response_selection", "selection_max_tokens", "policy_path",
    )))
PY
while IFS=$'\t' read -r name prompt_strategy self_consistency_k temperature max_tokens \
  response_selection selection_max_tokens policy_path; do
  run_eval "test/seed-$PRIMARY_SEED" "$name" "$prompt_strategy" "$self_consistency_k" \
    "$temperature" "$max_tokens" "$response_selection" "$selection_max_tokens" \
    "$PRIMARY_SEED" "" "50" "$policy_path"
done

write_status "robustness" "repeating direct answer and primary router at two fixed seeds"
CBRR_POLICY="$RUN_ROOT/selection/cbrr_policy.json"
for seed in 20260710 20260711; do
  run_eval "robustness/seed-$seed" "direct_answer" "direct_answer" "1" "0.0" "64" \
    "majority_vote" "64" "$seed" "" "50" ""
  run_eval "robustness/seed-$seed" "cbrr_policy" "direct_answer" "1" "0.0" "256" \
    "majority_vote" "64" "$seed" "" "50" "$CBRR_POLICY"
done

write_status "analyzing" "running paired tests, bootstrap intervals, costs, and examples"
python3 "$REPO_DIR/scripts/analyze_confirmatory_results.py" \
  --test-root "$RUN_ROOT/test/seed-$PRIMARY_SEED" \
  --selection "$RUN_ROOT/selection/selection.json" \
  --datasets-root "$DATASETS_ROOT" \
  --robustness-root "$RUN_ROOT/robustness" \
  --output-dir "$RUN_ROOT/analysis" \
  --bootstrap-replicates 10000 \
  --bootstrap-seed "$PRIMARY_SEED" \
  > "$RUN_ROOT/analysis/stdout.log" \
  2> "$RUN_ROOT/analysis/stderr.log"

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
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest,
    })
(root / "artifact_manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
PY

write_status "complete" "all screening, confirmatory, robustness, and analysis stages completed"
touch "$RUN_ROOT/COMPLETE"
log "E2B confirmatory experiment complete: $RUN_ROOT"
trap - EXIT
