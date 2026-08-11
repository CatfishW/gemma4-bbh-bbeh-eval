"""Small persistence helpers for resumable RL runs."""
from __future__ import annotations

import json
from pathlib import Path


def truncate_jsonl_at_iteration(path: Path, next_iteration: int) -> int:
    """Atomically remove rows at or beyond a resumed checkpoint iteration."""
    if not path.exists():
        return 0

    kept: list[str] = []
    removed = 0
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if "iteration" not in payload:
            raise ValueError(f"missing iteration in {path}:{line_number}")
        if int(payload["iteration"]) < next_iteration:
            kept.append(line)
        else:
            removed += 1

    if removed:
        temporary = path.with_name(f".{path.name}.resume.tmp")
        temporary.write_text("".join(f"{line}\n" for line in kept))
        temporary.replace(path)
    return removed
