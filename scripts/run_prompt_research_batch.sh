#!/usr/bin/env bash
set -euo pipefail

RUNS_BASE="${RUNS_BASE:-/data/benwulab/gemma4-eval/runs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BATCH_ROOT="${BATCH_ROOT:-$RUNS_BASE/prompt-research-batch-$STAMP}"
POLICY_ROOT="${POLICY_ROOT:-$RUNS_BASE/reward-routed-policy-$STAMP}"
SWEEP_ROOT="${SWEEP_ROOT:-$RUNS_BASE/research-strategy-sweep-$STAMP}"
REPO_REVISION="${REPO_REVISION:-$(git rev-parse HEAD 2>/dev/null || true)}"

mkdir -p "$BATCH_ROOT"
cat > "$BATCH_ROOT/batch_manifest.json" <<EOF
{
  "created_at": "$(date -Is)",
  "hostname": "$(hostname)",
  "repo_dir": "$(pwd)",
  "repo_revision": "$REPO_REVISION",
  "policy_root": "$POLICY_ROOT",
  "sweep_root": "$SWEEP_ROOT",
  "execution_order": ["reward_routed_policy", "research_strategy_sweep"],
  "system_messages_sent": 0
}
EOF

echo "[$(date -Is)] START reward-routed policy: $POLICY_ROOT"
REPO_REVISION="$REPO_REVISION" RUN_ROOT="$POLICY_ROOT" \
  ./scripts/run_reward_routed_policy.sh
echo "[$(date -Is)] DONE reward-routed policy"

echo "[$(date -Is)] START research strategy sweep: $SWEEP_ROOT"
REPO_REVISION="$REPO_REVISION" RUNS_ROOT="$SWEEP_ROOT" \
  ./scripts/run_research_strategy_sweep.sh
echo "[$(date -Is)] DONE research strategy sweep"

python3 - "$BATCH_ROOT" "$POLICY_ROOT" "$SWEEP_ROOT" <<'PY'
import json
from pathlib import Path
import sys

batch_root = Path(sys.argv[1])
policy_root = Path(sys.argv[2])
sweep_root = Path(sys.argv[3])
payload = {
    "batch_root": str(batch_root),
    "policy": json.loads((policy_root / "aggregate_summary.json").read_text()),
    "research_sweep": json.loads((sweep_root / "aggregate_summary.json").read_text()),
}
(batch_root / "batch_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "[$(date -Is)] prompt research batch complete: $BATCH_ROOT"
