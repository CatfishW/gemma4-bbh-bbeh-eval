#!/usr/bin/env bash
set -euo pipefail

REMOTE_JUMP="${REMOTE_JUMP:-tang-server}"
REMOTE_HOST="${REMOTE_HOST:-benwulab-remote}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:?set REMOTE_RUN_ROOT to the completed E2B run directory}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAX_GITHUB_BYTES="${MAX_GITHUB_BYTES:-95000000}"
BRANCH="${BRANCH:-agent/rl-informed-prompt-strategies}"

RUN_NAME="$(basename "$REMOTE_RUN_ROOT")"
REMOTE_PARENT="$(dirname "$REMOTE_RUN_ROOT")"
if [[ ! "$RUN_NAME" =~ ^e2b-confirmatory-[0-9]{8}_[0-9]{6}$ ]]; then
  echo "unexpected run name: $RUN_NAME" >&2
  exit 1
fi
LOCAL_RESULT_DIR="$REPO_DIR/results/$RUN_NAME"

ssh "$REMOTE_JUMP" \
  "ssh $REMOTE_HOST 'test -f $REMOTE_RUN_ROOT/COMPLETE && test -f $REMOTE_RUN_ROOT/artifact_manifest.json'"

mkdir -p "$REPO_DIR/results"
rm -rf "$LOCAL_RESULT_DIR"
ssh "$REMOTE_JUMP" \
  "ssh $REMOTE_HOST 'tar -C $REMOTE_PARENT -czf - $RUN_NAME'" |
  tar -xzf - -C "$REPO_DIR/results"

python3 - "$LOCAL_RESULT_DIR" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "artifact_manifest.json").read_text())
errors = []
for row in manifest["files"]:
    path = root / row["path"]
    if not path.is_file():
        errors.append(f"missing {row['path']}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != row["bytes"] or digest != row["sha256"]:
        errors.append(f"mismatch {row['path']}")
if errors:
    raise SystemExit("remote artifact verification failed:\n" + "\n".join(errors))
print(f"verified {len(manifest['files'])} remote artifacts")
PY

python3 - "$LOCAL_RESULT_DIR" "$MAX_GITHUB_BYTES" "$REMOTE_HOST" "$REMOTE_RUN_ROOT" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1])
max_bytes = int(sys.argv[2])
remote_host = sys.argv[3]
remote_root = sys.argv[4]
chunk_bytes = min(90_000_000, max_bytes)
split_files = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path.stat().st_size <= max_bytes:
        continue
    original = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parts": [],
    }
    with path.open("rb") as source:
        index = 0
        while True:
            data = source.read(chunk_bytes)
            if not data:
                break
            part = path.with_name(f"{path.name}.part-{index:03d}")
            part.write_bytes(data)
            original["parts"].append({
                "path": part.relative_to(root).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
            index += 1
    path.unlink()
    split_files.append(original)

files = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path.name == "upload_manifest.json":
        continue
    files.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
(root / "upload_manifest.json").write_text(json.dumps({
    "remote_host": remote_host,
    "remote_run_root": remote_root,
    "remote_artifact_manifest_verified": True,
    "split_files": split_files,
    "files": files,
}, indent=2) + "\n")
PY

cd "$REPO_DIR"
git add "$LOCAL_RESULT_DIR" scripts/collect_e2b_results.sh
git diff --cached --check
if git diff --cached --quiet; then
  echo "no changes to commit"
  exit 0
fi
git commit -m "results: add Gemma 4 E2B confirmatory experiment"
git push origin "$BRANCH"
echo "published $LOCAL_RESULT_DIR to $BRANCH"
