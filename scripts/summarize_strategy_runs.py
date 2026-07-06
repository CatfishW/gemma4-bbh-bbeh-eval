#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_strategy_runs.py <runs_root>")
    root = Path(sys.argv[1])
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        row = {
            "strategy_dir": summary_path.parent.name,
            "prompt_strategy": summary.get("prompt_strategy"),
            "self_consistency_k": summary.get("self_consistency_k"),
            "max_tokens": summary.get("max_tokens"),
            "temperature": summary.get("temperature"),
            "started_at": summary.get("started_at"),
            "finished_at": summary.get("finished_at"),
            "benchmarks": summary.get("benchmarks", []),
        }
        rows.append(row)
    print(json.dumps({"runs_root": str(root), "strategies": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

