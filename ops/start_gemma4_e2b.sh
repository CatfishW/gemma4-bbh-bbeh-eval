#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data/models/gemma-4-E2B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-SubTokenLLM-E2B}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
PORT="${PORT:-8889}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.58}"
LOG_DIR="${LOG_DIR:-/data/benwulab/gemma4-eval/deployments/e2b}"
LAUNCHER="${LAUNCHER:-/home/benwulab/run_sglang_qwen.py}"

mkdir -p "$LOG_DIR"
if pgrep -af "${LAUNCHER}.*--port ${PORT}" > "$LOG_DIR/already-running.txt"; then
  cat "$LOG_DIR/already-running.txt"
  exit 0
fi

test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model.safetensors"

cat > "$LOG_DIR/launch-command.txt" <<EOF
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python $LAUNCHER --model-path $MODEL_PATH --served-model-name $SERVED_MODEL_NAME --context-length 131072 --dtype bfloat16 --mem-fraction-static $MEM_FRACTION_STATIC --max-total-tokens 131072 --chunked-prefill-size 4096 --max-prefill-tokens 8192 --enable-multimodal --reasoning-parser gemma4 --tool-call-parser gemma4 --sampling-backend pytorch --attention-backend triton --disable-cuda-graph --stream-interval 1 --watchdog-timeout 1800 --trust-remote-code --allow-auto-truncate --host 0.0.0.0 --port $PORT
EOF

nohup env CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python "$LAUNCHER" \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --context-length 131072 \
  --dtype bfloat16 \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --max-total-tokens 131072 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 8192 \
  --enable-multimodal \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --sampling-backend pytorch \
  --attention-backend triton \
  --disable-cuda-graph \
  --stream-interval 1 \
  --watchdog-timeout 1800 \
  --trust-remote-code \
  --allow-auto-truncate \
  --host 0.0.0.0 \
  --port "$PORT" \
  > "$LOG_DIR/server.log" 2>&1 &

echo "$!" > "$LOG_DIR/server.pid"
echo "started pid=$! port=$PORT gpu=$CUDA_DEVICE model=$SERVED_MODEL_NAME"
