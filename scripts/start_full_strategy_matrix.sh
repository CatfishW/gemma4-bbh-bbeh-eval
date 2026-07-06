#!/usr/bin/env bash
set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-/data/benwulab/gemma4-eval/runs/full-strategy-matrix-$(date +%Y%m%d_%H%M%S)}"
export RUNS_ROOT
mkdir -p "$RUNS_ROOT"

nohup bash scripts/run_full_strategy_matrix.sh > "$RUNS_ROOT/launcher.log" 2>&1 < /dev/null &
echo "$!" > "$RUNS_ROOT/launcher.pid"
echo "$RUNS_ROOT"
echo "pid=$(cat "$RUNS_ROOT/launcher.pid")"
