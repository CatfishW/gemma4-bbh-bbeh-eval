from __future__ import annotations

import json


DEFAULT_BACKENDS = {
    "SubTokenLLM": "http://127.0.0.1:8888/v1",
    "SubTokenLLM-E2B": "http://127.0.0.1:8889/v1",
}


def parse_backends(value: str | None) -> dict[str, str]:
    if not value:
        return dict(DEFAULT_BACKENDS)
    payload = json.loads(value)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("MODEL_BACKENDS must be a non-empty JSON object")
    result = {}
    for model, url in payload.items():
        if not isinstance(model, str) or not isinstance(url, str):
            raise ValueError("MODEL_BACKENDS keys and values must be strings")
        result[model] = url.rstrip("/")
    return result


def select_model(body: bytes, backends: dict[str, str], default_model: str) -> str:
    if not body:
        return default_model
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default_model
    if not isinstance(payload, dict):
        return default_model
    model = str(payload.get("model") or default_model)
    if model not in backends:
        raise ValueError(f"unknown model {model!r}; available={sorted(backends)}")
    return model
