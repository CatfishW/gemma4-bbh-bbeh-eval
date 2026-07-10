#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DEPLOY_DIR="${DEPLOY_DIR:-/data/benwulab/gemma4-eval/router}"
UNIT_DIR="${UNIT_DIR:-$HOME/.config/systemd/user}"

install -d "$DEPLOY_DIR/ops" "$UNIT_DIR"
install -m 0644 "$REPO_DIR/ops/model_router.py" "$DEPLOY_DIR/ops/model_router.py"
install -m 0644 "$REPO_DIR/ops/model_router_core.py" "$DEPLOY_DIR/ops/model_router_core.py"
install -m 0755 "$REPO_DIR/ops/tunnel_tang_25570.sh" \
  "$DEPLOY_DIR/ops/tunnel_tang_25570.sh"
install -m 0644 "$REPO_DIR"/ops/systemd/*.service "$UNIT_DIR/"

systemctl --user daemon-reload
systemctl --user enable \
  gemma4-e4b.service \
  gemma4-e2b.service \
  gemma4-router.service \
  gemma4-public-tunnel.service

printf '%s\n' \
  "Installed and enabled Gemma 4 user services without starting them." \
  "Stop legacy processes before starting these units to avoid port conflicts."
