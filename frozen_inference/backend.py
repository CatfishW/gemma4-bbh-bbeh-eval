"""Small OpenAI-compatible client with complete per-attempt audit records.

No network calls occur at import. API keys are read by the CLI and never recorded.
Each request contains exactly one user message. Server-rendered templates may add
control turns; deployment must audit those separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from urllib import error, parse, request

from .executor import digest_text


class BackendError(RuntimeError):
    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


def merge_usage(records: list[dict]) -> dict:
    """Sum known nested integer fields, retaining absent fields as absent (not zero)."""
    def add(target: dict, source: dict) -> None:
        for key, value in source.items():
            if type(value) is int and value >= 0:
                target[key] = target.get(key, 0) + value if type(target.get(key, 0)) is int else value
            elif isinstance(value, dict):
                if not isinstance(target.get(key), dict):
                    target[key] = {}
                add(target[key], value)
    combined = {}
    for record in records:
        usage = record.get("usage")
        if isinstance(usage, dict):
            add(combined, usage)
    return combined


@dataclass(frozen=True)
class ChatClient:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60.0
    retries: int = 0

    def __post_init__(self) -> None:
        url = parse.urlsplit(self.base_url)
        if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base URL must be HTTP(S), without credentials/query/fragment")
        if not self.model or not 0 < self.timeout <= 600 or not 0 <= self.retries <= 2:
            raise ValueError("invalid model, timeout or retries")

    def complete(self, prompt: str, *, max_tokens: int, seed: int,
                 timeout: float | None = None) -> dict:
        if not 1 <= max_tokens <= 8192:
            raise ValueError("invalid completion budget")
        budget = min(self.timeout, self.timeout if timeout is None else timeout)
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("invalid request time budget")
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                   "max_completion_tokens": max_tokens, "temperature": 0.0, "seed": seed}
        record = {"prompt_sha256": digest_text(prompt), "request": payload,
                  "attempts": [], "usage": {}, "usage_complete": False}
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        deadline = time.monotonic() + budget
        for attempt in range(self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            started = time.perf_counter()
            retryable = False
            try:
                req = request.Request(self.base_url.rstrip("/") + "/chat/completions",
                                      data=body, headers=headers, method="POST")
                with request.urlopen(req, timeout=remaining) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("response size limit")
                reply = json.loads(raw)
                choice = reply["choices"][0]
                message = choice["message"]
                if not isinstance(message, dict):
                    raise ValueError("response message must be an object")
                content = message.get("content") or ""
                if not isinstance(content, str):
                    raise ValueError("response content must be text")
                usage = reply.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ValueError("invalid usage object")
                record["attempts"].append({"attempt": attempt, "elapsed_seconds": time.perf_counter() - started,
                                           "error": None})
                record.update(content=content, message=message, finish_reason=choice.get("finish_reason"),
                              usage=usage, response_id=reply.get("id"),
                              system_fingerprint=reply.get("system_fingerprint"),
                              usage_complete=attempt == 0 and any(type(usage.get(k)) is int for k in
                                                                 ("completion_tokens", "output_tokens")))
                return record
            except error.HTTPError as exc:
                message = f"HTTP {exc.code}"
                retryable = exc.code in {429, 500, 502, 503, 504}
            except (error.URLError, TimeoutError, OSError) as exc:
                message = type(exc).__name__
                retryable = True
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                message = f"invalid response: {type(exc).__name__}"
            record["attempts"].append({"attempt": attempt, "elapsed_seconds": time.perf_counter() - started,
                                       "error": message})
            if not retryable or attempt == self.retries:
                break
            delay = min(2 ** attempt, max(0.0, deadline - time.monotonic()))
            time.sleep(delay)
        raise BackendError("model request failed", record)
