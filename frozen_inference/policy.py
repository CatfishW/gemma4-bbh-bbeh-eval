"""Frozen, paired repairability routing; no neural training or test-time rewards."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
from statistics import mean

from .executor import Rejected, digest_text


def features(question: str) -> str:
    length = "short" if len(question) < 1000 else "medium" if len(question) < 4000 else "long"
    choices = bool(re.search(r"(?m)^\(?[A-Z]\)\s+", question))
    return f"{length}|choices={int(choices)}"


def _wilson(k: int, n: int, z: float = 2.2414) -> tuple[float, float]:
    """Bonferroni-style separate intervals for wins and losses; not a global guarantee."""
    if not n:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def fit_policy(baseline: list[dict], challenger: list[dict], *, min_examples: int = 20,
               penalty_per_second: float = 0.01) -> dict:
    """Paired rows MUST be calibration outputs with identical input hashes.

    No labels are retained in the exported policy. Multiple buckets/selection still
    need a separate validation set; these intervals are not a safety certificate.
    """
    if min_examples < 1 or penalty_per_second < 0 or not math.isfinite(penalty_per_second):
        raise Rejected("invalid calibration settings")

    def index(rows: list[dict]) -> dict:
        indexed = {}
        for row in rows:
            key = row.get("case_id")
            if not isinstance(key, str) or not key or key in indexed:
                raise Rejected("missing/duplicate calibration case id")
            if row.get("split") != "calibration" or type(row.get("index")) is not int or not 0 <= row["index"] < 25:
                raise Rejected("policy fitting accepts only indices 0..24 labeled calibration")
            if type(row.get("correct")) is not bool or not isinstance(row.get("feature_bucket"), str):
                raise Rejected("missing calibration correctness/features")
            seconds = row.get("elapsed_seconds")
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise Rejected("invalid observed cost")
            if not re.fullmatch(r"[a-f0-9]{64}", str(row.get("question_sha256", ""))):
                raise Rejected("missing input digest")
            indexed[key] = row
        return indexed

    left, right = index(baseline), index(challenger)
    if not left or set(left) != set(right):
        raise Rejected("paired coverage mismatch/empty calibration")
    buckets = defaultdict(list)
    for key, a in left.items():
        b = right[key]
        if any(a.get(field) != b.get(field) for field in ("question_sha256", "feature_bucket", "index")):
            raise Rejected("paired input/features mismatch")
        # The gate runs AFTER exact/template dispatch. Their gains must not be
        # attributed to compilation, or the learned action value is biased.
        if b.get("compile_attempted") is True:
            buckets[a["feature_bucket"]].append((a, b))
    if not buckets:
        raise Rejected("no attempted compilations in calibration")
    rows = {}
    for bucket, pairs in sorted(buckets.items()):
        wins = sum(not a["correct"] and b["correct"] for a, b in pairs)
        losses = sum(a["correct"] and not b["correct"] for a, b in pairs)
        n = len(pairs)
        low = _wilson(wins, n)[0] - _wilson(losses, n)[1]
        extra = max(0.0, mean(b["elapsed_seconds"] - a["elapsed_seconds"] for a, b in pairs))
        rows[bucket] = {"n": n, "wins": wins, "losses": losses,
                        "lower_net_gain": low, "incremental_seconds": extra,
                        "allow_compile": n >= min_examples and low > penalty_per_second * extra}
    return {"schema": 1, "action": "compile", "source_split": "calibration",
            "min_examples": min_examples, "penalty_per_second": penalty_per_second,
            "source_digest": digest_text(json.dumps([baseline, challenger], sort_keys=True)),
            "buckets": rows}


class RepairPolicy:
    def __init__(self, payload: dict):
        if not isinstance(payload, dict) or payload.get("schema") != 1 or payload.get("action") != "compile":
            raise Rejected("policy schema/action")
        if payload.get("source_split") != "calibration" or not isinstance(payload.get("buckets"), dict):
            raise Rejected("policy must be frozen from calibration")
        for bucket, row in payload["buckets"].items():
            if not isinstance(bucket, str) or not isinstance(row, dict) or type(row.get("allow_compile")) is not bool:
                raise Rejected("invalid policy bucket")
        self.payload = payload
        self.fingerprint = digest_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "RepairPolicy":
        path = Path(path)
        if path.stat().st_size > 1_000_000:
            raise Rejected("policy file size limit")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def allows(self, question: str) -> bool:
        return self.payload["buckets"].get(features(question), {}).get("allow_compile", False)
