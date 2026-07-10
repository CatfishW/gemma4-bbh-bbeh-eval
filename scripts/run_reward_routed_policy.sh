#!/usr/bin/env bash
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-/data/benwulab/gemma4-eval/datasets}"
RUNS_BASE="${RUNS_BASE:-/data/benwulab/gemma4-eval/runs}"
BBH_BBEH_ROOT="${BBH_BBEH_ROOT:-$RUNS_BASE/full-strategy-matrix-20260706_025955}"
USR_ROOT="${USR_ROOT:-$RUNS_BASE/usr-strategy-matrix-20260707_022230}"
CHALLENGER_ROOT="${CHALLENGER_ROOT:-$RUNS_BASE/full-challenger-winners-20260709_120053}"
RUN_ROOT="${RUN_ROOT:-$RUNS_BASE/reward-routed-policy-$(date +%Y%m%d_%H%M%S)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${MODEL:-SubTokenLLM}"
BENCHMARKS="${BENCHMARKS:-bbh,bbeh,usr}"
CALIBRATION_SIZE="${CALIBRATION_SIZE:-25}"
PARALLEL="${PARALLEL:-2}"
TIMEOUT="${TIMEOUT:-300}"
RETRIES="${RETRIES:-2}"
STRATEGIES="${STRATEGIES:-direct_answer,strict_json,concise_cot,chain_of_draft,plan_and_solve,step_back,premise_conclusion,symbolic_proof,canonical_short,private_verify}"
REPO_REVISION="${REPO_REVISION:-$(git rev-parse HEAD 2>/dev/null || true)}"

mkdir -p "$RUN_ROOT/offline" "$RUN_ROOT/online_heldout"

cat > "$RUN_ROOT/environment.json" <<EOF
{
  "created_at": "$(date -Is)",
  "hostname": "$(hostname)",
  "repo_dir": "$(pwd)",
  "repo_revision": "$REPO_REVISION",
  "datasets_root": "$DATASETS_ROOT",
  "source_roots": ["$BBH_BBEH_ROOT", "$USR_ROOT", "$CHALLENGER_ROOT"],
  "base_url": "$BASE_URL",
  "model": "$MODEL",
  "benchmarks": "$BENCHMARKS",
  "calibration_size": "$CALIBRATION_SIZE",
  "parallel": "$PARALLEL",
  "strategies": "$STRATEGIES",
  "system_messages_sent": 0
}
EOF

python3 scripts/calibrate_prompt_policy.py \
  --runs-root "$BBH_BBEH_ROOT" \
  --runs-root "$USR_ROOT" \
  --runs-root "$CHALLENGER_ROOT" \
  --strategies "$STRATEGIES" \
  --baseline direct_answer \
  --calibration-size "$CALIBRATION_SIZE" \
  --min-reward-gain 1 \
  --policy-name reward_routed_v1 \
  --datasets-root "$DATASETS_ROOT" \
  --output-dir "$RUN_ROOT/offline" \
  > "$RUN_ROOT/offline/stdout.log" \
  2> "$RUN_ROOT/offline/stderr.log"

cat > "$RUN_ROOT/online_heldout/command.txt" <<EOF
python3 eval_benchmarks.py \\
  --datasets-root "$DATASETS_ROOT" \\
  --base-url "$BASE_URL" \\
  --model "$MODEL" \\
  --benchmarks "$BENCHMARKS" \\
  --prompt-policy "$RUN_ROOT/offline/policy.json" \\
  --skip-per-task "$CALIBRATION_SIZE" \\
  --self-consistency-k 1 \\
  --temperature 0.0 \\
  --max-tokens 256 \\
  --parallel "$PARALLEL" \\
  --timeout "$TIMEOUT" \\
  --retries "$RETRIES" \\
  --output-dir "$RUN_ROOT/online_heldout"
EOF

echo "[$(date -Is)] START online held-out reward-routed policy" | tee -a "$RUN_ROOT/policy.log"
bash "$RUN_ROOT/online_heldout/command.txt" \
  > "$RUN_ROOT/online_heldout/stdout.log" \
  2> "$RUN_ROOT/online_heldout/stderr.log"
echo "[$(date -Is)] DONE online held-out reward-routed policy" | tee -a "$RUN_ROOT/policy.log"

python3 - "$RUN_ROOT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
offline = json.loads((root / "offline" / "heldout_summary.json").read_text())
online = json.loads((root / "online_heldout" / "summary.json").read_text())
online_correct = sum(item["correct"] for item in online["benchmarks"])
online_total = sum(item["total"] for item in online["benchmarks"])
baseline = offline["baseline"]
payload = {
    "run_root": str(root),
    "evaluation_split": offline["evaluation_split"],
    "calibration_examples_excluded": offline["calibration_examples_excluded"],
    "baseline": baseline,
    "offline_policy_replay": offline["policy"],
    "online_policy": {
        "correct": online_correct,
        "total": online_total,
        "accuracy": online_correct / online_total,
        "correct_gain_over_direct": online_correct - baseline["correct"],
        "accuracy_gain_over_direct": (online_correct - baseline["correct"]) / online_total,
        "benchmarks": online["benchmarks"],
    },
    "system_messages_sent": 0,
}
(root / "aggregate_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "[$(date -Is)] reward-routed policy complete: $RUN_ROOT" | tee -a "$RUN_ROOT/policy.log"
