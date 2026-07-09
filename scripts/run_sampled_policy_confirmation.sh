#!/usr/bin/env bash
set -euo pipefail

SAMPLE_ROOT="${SAMPLE_ROOT:?set SAMPLE_ROOT to a completed calibration sweep}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/benwulab/gemma4-eval/datasets}"
RUNS_BASE="${RUNS_BASE:-/data/benwulab/gemma4-eval/runs}"
RUN_ROOT="${RUN_ROOT:-$RUNS_BASE/reward-routed-v2-$(date +%Y%m%d_%H%M%S)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${MODEL:-SubTokenLLM}"
BENCHMARKS="${BENCHMARKS:-bbh,bbeh,usr}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-25}"
MIN_REWARD_GAIN="${MIN_REWARD_GAIN:-2}"
PARALLEL="${PARALLEL:-2}"
TIMEOUT="${TIMEOUT:-300}"
RETRIES="${RETRIES:-2}"
STRATEGIES="${STRATEGIES:-direct_answer,private_verify,canonical_short,compare_then_commit,constraint_guard,selective_verify,draft_verify}"
DIRECT_HELDOUT_CORRECT="${DIRECT_HELDOUT_CORRECT:-3632}"
DIRECT_HELDOUT_TOTAL="${DIRECT_HELDOUT_TOTAL:-11040}"
REPO_REVISION="${REPO_REVISION:-$(git rev-parse HEAD 2>/dev/null || true)}"

mkdir -p "$RUN_ROOT/calibration" "$RUN_ROOT/heldout"

cat > "$RUN_ROOT/environment.json" <<EOF
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
  "calibration_size": "$CALIBRATION_SIZE",
  "min_reward_gain": "$MIN_REWARD_GAIN",
  "parallel": "$PARALLEL",
  "strategies": "$STRATEGIES",
  "direct_heldout_correct": "$DIRECT_HELDOUT_CORRECT",
  "direct_heldout_total": "$DIRECT_HELDOUT_TOTAL",
  "system_messages_sent": 0
}
EOF

python3 scripts/calibrate_prompt_policy.py \
  --runs-root "$SAMPLE_ROOT" \
  --strategies "$STRATEGIES" \
  --baseline direct_answer \
  --calibration-size "$CALIBRATION_SIZE" \
  --min-reward-gain "$MIN_REWARD_GAIN" \
  --policy-name reward_routed_v2 \
  --calibration-only \
  --output-dir "$RUN_ROOT/calibration" \
  > "$RUN_ROOT/calibration/stdout.log" \
  2> "$RUN_ROOT/calibration/stderr.log"

cat > "$RUN_ROOT/heldout/command.txt" <<EOF
python3 eval_benchmarks.py \\
  --datasets-root "$DATASETS_ROOT" \\
  --base-url "$BASE_URL" \\
  --model "$MODEL" \\
  --benchmarks "$BENCHMARKS" \\
  --prompt-policy "$RUN_ROOT/calibration/policy.json" \\
  --skip-per-task "$CALIBRATION_SIZE" \\
  --self-consistency-k 1 \\
  --temperature 0.0 \\
  --max-tokens 128 \\
  --parallel "$PARALLEL" \\
  --timeout "$TIMEOUT" \\
  --retries "$RETRIES" \\
  --output-dir "$RUN_ROOT/heldout"
EOF

echo "[$(date -Is)] START reward-routed-v2 held-out confirmation" | tee -a "$RUN_ROOT/policy.log"
bash "$RUN_ROOT/heldout/command.txt" \
  > "$RUN_ROOT/heldout/stdout.log" \
  2> "$RUN_ROOT/heldout/stderr.log"
echo "[$(date -Is)] DONE reward-routed-v2 held-out confirmation" | tee -a "$RUN_ROOT/policy.log"

python3 - "$RUN_ROOT" "$DIRECT_HELDOUT_CORRECT" "$DIRECT_HELDOUT_TOTAL" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
direct_correct = int(sys.argv[2])
direct_total = int(sys.argv[3])
summary = json.loads((root / "heldout" / "summary.json").read_text())
policy_correct = sum(item["correct"] for item in summary["benchmarks"])
policy_total = sum(item["total"] for item in summary["benchmarks"])
if policy_total != direct_total:
    raise SystemExit(f"held-out total mismatch: policy={policy_total} direct={direct_total}")
payload = {
    "status": "complete",
    "evaluation_split": f"index >= {summary['skip_per_task']} within each task",
    "baseline": {
        "strategy": "direct_answer",
        "correct": direct_correct,
        "total": direct_total,
        "accuracy": direct_correct / direct_total,
        "source": "archived full direct-answer predictions",
    },
    "reward_routed_v2": {
        "correct": policy_correct,
        "total": policy_total,
        "accuracy": policy_correct / policy_total,
        "correct_gain_over_direct": policy_correct - direct_correct,
        "accuracy_gain_over_direct": (policy_correct - direct_correct) / policy_total,
        "benchmarks": summary["benchmarks"],
    },
    "system_messages_sent": 0,
}
(root / "aggregate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "[$(date -Is)] reward-routed-v2 complete: $RUN_ROOT" | tee -a "$RUN_ROOT/policy.log"
