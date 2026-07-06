#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-tang-server-org}"
REMOTE_PORT="${REMOTE_PORT:-25570}"
LOCAL_PORT="${LOCAL_PORT:-8888}"
RECONNECT_DELAY="${RECONNECT_DELAY:-5}"
SSH_CONFIG="${SSH_CONFIG:-/home/benwulab/.ssh/config}"

while true; do
  echo "[$(date -Is)] tunnel ${REMOTE_HOST}:${REMOTE_PORT} -> 127.0.0.1:${LOCAL_PORT}"
  ssh -F "$SSH_CONFIG" \
    -R "${REMOTE_PORT}:127.0.0.1:${LOCAL_PORT}" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=no \
    -N "$REMOTE_HOST"
  rc=$?
  echo "[$(date -Is)] tunnel exited rc=${rc}; reconnecting in ${RECONNECT_DELAY}s"
  sleep "$RECONNECT_DELAY"
done

