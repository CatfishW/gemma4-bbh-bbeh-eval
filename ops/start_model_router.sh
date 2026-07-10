#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/benwulab/anaconda3/envs/SGLang/bin/python}"
PORT="${PORT:-8890}"
LOG_DIR="${LOG_DIR:-/data/benwulab/gemma4-eval/deployments/model-router}"
DEFAULT_MODEL="${DEFAULT_MODEL:-SubTokenLLM}"
MODEL_BACKENDS="${MODEL_BACKENDS:-{\"SubTokenLLM\":\"http://127.0.0.1:8888/v1\",\"SubTokenLLM-E2B\":\"http://127.0.0.1:8889/v1\"}}"

mkdir -p "$LOG_DIR"
if pgrep -af "uvicorn ops.model_router:app.*--port ${PORT}" > "$LOG_DIR/already-running.txt"; then
  cat "$LOG_DIR/already-running.txt"
  exit 0
fi

cat > "$LOG_DIR/launch-command.txt" <<EOF
MODEL_BACKENDS='$MODEL_BACKENDS' DEFAULT_MODEL='$DEFAULT_MODEL' $PYTHON_BIN -m uvicorn ops.model_router:app --app-dir $REPO_DIR --host 0.0.0.0 --port $PORT
EOF

cd "$REPO_DIR"
nohup env MODEL_BACKENDS="$MODEL_BACKENDS" DEFAULT_MODEL="$DEFAULT_MODEL" \
  "$PYTHON_BIN" -m uvicorn ops.model_router:app \
  --app-dir "$REPO_DIR" \
  --host 0.0.0.0 \
  --port "$PORT" \
  > "$LOG_DIR/router.log" 2>&1 &

echo "$!" > "$LOG_DIR/router.pid"
echo "started pid=$! port=$PORT default_model=$DEFAULT_MODEL"
