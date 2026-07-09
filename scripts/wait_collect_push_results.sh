#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-benwulab-remote}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:?set REMOTE_RUN_ROOT to the completed remote run directory}"
REMOTE_PID="${REMOTE_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-300}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAX_GITHUB_FILE_BYTES="${MAX_GITHUB_FILE_BYTES:-95000000}"

RUN_NAME="$(basename "$REMOTE_RUN_ROOT")"
LOCAL_RESULT_DIR="$REPO_DIR/results/$RUN_NAME"
LOG_ROOT="$REPO_DIR/runs/result-uploaders"
LOG_FILE="$LOG_ROOT/$RUN_NAME.log"
DONE_MARKER="$LOG_ROOT/$RUN_NAME.done"

mkdir -p "$LOG_ROOT"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$LOG_FILE"
}

remote_quote() {
  python3 - "$1" <<'PY'
import shlex
import sys
print(shlex.quote(sys.argv[1]))
PY
}

remote_run_root_quoted="$(remote_quote "$REMOTE_RUN_ROOT")"

if [[ -f "$DONE_MARKER" ]]; then
  log "upload already marked complete; exiting"
  exit 0
fi

log "watching $REMOTE_HOST:$REMOTE_RUN_ROOT"
log "local result dir: $LOCAL_RESULT_DIR"

while true; do
  if ssh "$REMOTE_HOST" "test -f $remote_run_root_quoted/aggregate_summary.json"; then
    log "aggregate_summary.json found; remote run is complete"
    break
  fi

  if [[ -n "$REMOTE_PID" ]]; then
    if ! ssh "$REMOTE_HOST" "ps -p $REMOTE_PID >/dev/null 2>&1"; then
      log "remote pid $REMOTE_PID exited before aggregate_summary.json appeared"
      exit 1
    fi
  fi

  log "still running; sleeping ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

mkdir -p "$LOCAL_RESULT_DIR"
log "rsyncing completed run"
rsync -az --delete "$REMOTE_HOST:$REMOTE_RUN_ROOT/" "$LOCAL_RESULT_DIR/"

log "compressing prediction JSONL files"
find "$LOCAL_RESULT_DIR" -type f -name 'predictions.jsonl' -print0 |
  while IFS= read -r -d '' prediction_file; do
    gzip -f "$prediction_file"
  done

log "splitting any GitHub-large files over ${MAX_GITHUB_FILE_BYTES} bytes"
python3 - "$LOCAL_RESULT_DIR" "$MAX_GITHUB_FILE_BYTES" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
max_bytes = int(sys.argv[2])
chunk_bytes = min(90_000_000, max_bytes)

for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path.stat().st_size <= max_bytes:
        continue
    data = path.read_bytes()
    for index, start in enumerate(range(0, len(data), chunk_bytes)):
        chunk = path.with_name(f"{path.name}.part-{index:03d}")
        chunk.write_bytes(data[start : start + chunk_bytes])
    path.unlink()
    print(f"split {path}")
PY

log "writing upload manifest"
python3 - "$LOCAL_RESULT_DIR" "$REMOTE_HOST" "$REMOTE_RUN_ROOT" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
remote_host = sys.argv[2]
remote_run_root = sys.argv[3]
files = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path.name == "upload_manifest.json":
        continue
    rel = path.relative_to(root).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({"path": rel, "bytes": path.stat().st_size, "sha256": digest})

manifest = {
    "remote_host": remote_host,
    "remote_run_root": remote_run_root,
    "files": files,
}
(root / "upload_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
)
PY

cd "$REPO_DIR"
git add "$LOCAL_RESULT_DIR"

if git diff --cached --quiet; then
  log "no result changes to commit"
  touch "$DONE_MARKER"
  exit 0
fi

log "committing results"
git commit -m "Add full strategy matrix results $RUN_NAME"

log "pushing results to GitHub"
git push origin main

touch "$DONE_MARKER"
log "upload complete"
