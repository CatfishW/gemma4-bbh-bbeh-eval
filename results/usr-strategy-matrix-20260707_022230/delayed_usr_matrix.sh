#!/usr/bin/env bash
set -euo pipefail
OLD_AGG="/data/benwulab/gemma4-eval/runs/full-strategy-matrix-20260706_025955/aggregate_summary.json"
while [ ! -f "$OLD_AGG" ]; do
  echo "[$(date -Is)] waiting for BBH/BBEH matrix: $OLD_AGG"
  sleep 300
done
cd /data/benwulab/gemma4-eval/repo-usr
export REPO_REVISION=a7fbd20e28fbdebc2a9ba112c39b064268ab4a76
export RUNS_ROOT="/data/benwulab/gemma4-eval/runs/usr-strategy-matrix-20260707_022230"
export BENCHMARKS=usr
export PARALLEL=2
export BASE_URL=http://127.0.0.1:8888/v1
export MODEL=SubTokenLLM
export TIMEOUT=300
export RETRIES=2
bash scripts/run_full_strategy_matrix.sh
