"""Read-only, guarded arithmetic microprogram memory (not an answer cache).

Templates are literal text plus bounded {slot:int} placeholders, NOT user-provided
regular expressions. Stored examples are development checks, not a proof that the
English template faithfully represents every possible question.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .executor import Rejected, digest_text, expression

_SLOT = re.compile(r"\{([a-z][a-z0-9_]{0,31}):int\}")


@dataclass(frozen=True)
class Skill:
    name: str
    template: str
    formula: str
    pattern: re.Pattern
    slots: tuple[str, ...]
    bounds: dict[str, tuple[int, int]]

    def match(self, question: str) -> str | None:
        if len(question) > 4096:
            return None
        match = self.pattern.fullmatch(question.strip())
        if match is None:
            return None
        values = {key: int(match[key]) for key in self.slots}
        if any(not self.bounds[key][0] <= value <= self.bounds[key][1] for key, value in values.items()):
            return None
        return expression(self.formula, values)


class SkillLibrary:
    def __init__(self, payload: dict):
        if not isinstance(payload, dict) or set(payload) != {"schema", "origin", "skills"}:
            raise Rejected("skill library schema")
        if payload["schema"] != 1 or payload["origin"] not in {"synthetic-development", "calibration"}:
            raise Rejected("memory must be versioned and built on development data")
        rows = payload["skills"]
        if not isinstance(rows, list) or len(rows) > 64:
            raise Rejected("skill count limit")
        self.fingerprint = digest_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        self.skills: list[Skill] = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"name", "template", "formula", "bounds", "positive", "negative"}:
                raise Rejected("skill schema")
            name, template, formula = row["name"], row["template"], row["formula"]
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) or name in seen:
                raise Rejected("invalid/duplicate skill name")
            seen.add(name)
            if not isinstance(template, str) or len(template) > 2048 or not isinstance(formula, str):
                raise Rejected("invalid template/formula")
            parts, slots, start = [], [], 0
            for match in _SLOT.finditer(template):
                parts.append(re.escape(template[start:match.start()]))
                slot = match[1]
                if slot in slots:
                    raise Rejected("repeated slot names are not supported")
                slots.append(slot)
                parts.append(r"(?P<" + slot + r">-?(?:0|[1-9][0-9]{0,11}))")
                start = match.end()
            parts.append(re.escape(template[start:]))
            if not 1 <= len(slots) <= 12 or "{" in _SLOT.sub("", template) or "}" in _SLOT.sub("", template):
                raise Rejected("invalid slot syntax")
            bounds = row["bounds"]
            if not isinstance(bounds, dict) or set(bounds) != set(slots):
                raise Rejected("every slot requires explicit integer bounds")
            for pair in bounds.values():
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(type(v) is not int for v in pair) or pair[0] > pair[1]):
                    raise Rejected("invalid slot bounds")
            skill = Skill(name, template, formula, re.compile("".join(parts)), tuple(slots),
                          {k: tuple(v) for k, v in bounds.items()})
            positives, negatives = row["positive"], row["negative"]
            if not isinstance(positives, list) or not 2 <= len(positives) <= 100:
                raise Rejected("at least two development checks required")
            if not isinstance(negatives, list) or not 1 <= len(negatives) <= 100:
                raise Rejected("negative applicability checks required")
            for case in positives:
                if not isinstance(case, dict) or set(case) != {"input", "answer"}:
                    raise Rejected("development case schema")
                if not isinstance(case["input"], str) or not isinstance(case["answer"], str):
                    raise Rejected("development cases must contain strings")
                if skill.match(case["input"]) != case["answer"]:
                    raise Rejected(f"development check failed: {name}")
            for text in negatives:
                if not isinstance(text, str) or skill.match(text) is not None:
                    raise Rejected(f"negative applicability check failed: {name}")
            self.skills.append(skill)

    @classmethod
    def load(cls, path: str | Path) -> "SkillLibrary":
        path = Path(path)
        if path.stat().st_size > 1_000_000:
            raise Rejected("skill file size limit")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def solve(self, question: str) -> dict | None:
        matches = []
        for skill in self.skills:
            try:
                answer = skill.match(question)
            except Rejected:
                continue
            if answer is not None:
                matches.append((skill.name, answer))
        if not matches or len({answer for _, answer in matches}) != 1:
            return None
        return {"answer": matches[0][1], "skills": [name for name, _ in matches],
                "library_sha256": self.fingerprint,
                "certificate_scope": "development-tested-template-not-semantic-proof"}
