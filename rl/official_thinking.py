"""Utilities for the Gemma 4 native-thinking BBEH replication.

This module is deliberately separate from :mod:`rl.rollout`.  The original RL
evaluation is a frozen, non-thinking protocol and must remain reproducible.
The helpers here implement the public Gemma 4 chat contract: a single user
message, ``enable_thinking=True``, channel-aware response parsing, and explicit
generation-stop accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable


OFFICIAL_TEMPERATURE = 1.0
OFFICIAL_TOP_P = 0.95
OFFICIAL_TOP_K = 64
DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_SEED = 20260709

EXPECTED_MODEL_REVISION = "70af34e20bd4b7a91f0de6b22675850c43922a03"
EXPECTED_MODEL_SHA256 = "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
EXPECTED_BBEH_REVISION = "80d12ca916b7158f22293fcf3144f4d3d854d4be"

# Published in Appendix C ("Reproducibility") of the BBEH paper.  This is part
# of the benchmark protocol, not a prompt strategy selected on our data.
BBEH_EVALUATION_SUFFIX = (
    'Think step by step, and when you provide the final answer, please use the prefix "The '
    'answer is:" without any modification, and provide the answer directly, with no formatting, '
    'no bolding, and no markup. For instance: "The answer is: 42" or "The answer is: yes". If '
    "the question is multiple choice with a single correct answer, the final answer must only be "
    'the letter corresponding to the correct answer. For example, "The answer is: (a)".'
)


@dataclass(frozen=True)
class ParsedThinkingResponse:
    """A generated response split according to Gemma's response schema."""

    raw_response: str
    prediction: str
    thinking: str
    parse_error: str | None


@dataclass(frozen=True)
class TrimmedGeneration:
    """One generated token sequence with batch padding removed."""

    token_ids: list[int]
    stopped: bool
    stop_token_id: int | None
    truncated: bool


def native_thinking_prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    """Render one raw benchmark prompt with Gemma's native thinking enabled.

    No manual system message is supplied.  Gemma's pinned chat template adds
    the leading system ``<|think|>`` turn itself when ``enable_thinking=True``;
    manually adding that turn would duplicate the control token.
    """

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one chat-template sequence")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def official_bbeh_prompt(dataset_input: str) -> str:
    """Append the exact evaluation suffix published by the BBEH authors."""

    return dataset_input.rstrip() + "\n\n" + BBEH_EVALUATION_SUFFIX


def validate_native_thinking_prompt(tokenizer, token_ids: list[int]) -> None:
    """Fail closed if the tokenizer did not inject exactly one thinking turn."""

    rendered = tokenizer.decode(token_ids, skip_special_tokens=False)
    required = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\n"
    if required not in rendered:
        raise RuntimeError(
            "Gemma chat template did not produce the required native-thinking "
            "system turn"
        )
    think_id = tokenizer.convert_tokens_to_ids("<|think|>")
    if think_id is None or think_id < 0:
        raise RuntimeError("tokenizer has no <|think|> token")
    if token_ids.count(int(think_id)) != 1:
        raise RuntimeError("native-thinking prompt must contain exactly one <|think|> token")


def parse_native_thinking_response(tokenizer, token_ids: list[int]) -> ParsedThinkingResponse:
    """Parse Gemma output and expose only final-channel content for scoring.

    Falling back to ``decode(..., skip_special_tokens=True)`` is intentionally
    forbidden: that operation concatenates the private thinking trace and the
    final answer, which can silently corrupt exact-match evaluation.
    """

    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    try:
        parsed = tokenizer.parse_response(raw)
    except Exception as error:  # Store an auditable failed parse; score empty.
        return ParsedThinkingResponse(
            raw_response=raw,
            prediction="",
            thinking="",
            parse_error=f"{type(error).__name__}: {error}",
        )
    if not isinstance(parsed, dict):
        return ParsedThinkingResponse(
            raw_response=raw,
            prediction="",
            thinking="",
            parse_error=f"unexpected parsed response type: {type(parsed).__name__}",
        )
    content = parsed.get("content")
    thinking = parsed.get("thinking")
    if content is None:
        parse_error = "missing final content"
    elif not isinstance(content, str):
        parse_error = "non-string final content"
    else:
        parse_error = None
    return ParsedThinkingResponse(
        raw_response=raw,
        prediction=content if isinstance(content, str) else "",
        thinking=thinking if isinstance(thinking, str) else "",
        parse_error=parse_error,
    )


def trim_generated_token_ids(
    token_ids: Iterable[int],
    *,
    stop_token_ids: Iterable[int],
    pad_token_id: int | None,
    max_new_tokens: int,
) -> TrimmedGeneration:
    """Trim at the first configured stop token and remove trailing batch pads."""

    ids = [int(token_id) for token_id in token_ids]
    stop_ids = {int(token_id) for token_id in stop_token_ids}
    stop_index = next((index for index, token_id in enumerate(ids) if token_id in stop_ids), None)
    if stop_index is not None:
        stop_id = ids[stop_index]
        ids = ids[: stop_index + 1]
        return TrimmedGeneration(ids, True, stop_id, False)

    if pad_token_id is not None:
        while ids and ids[-1] == pad_token_id:
            ids.pop()
    return TrimmedGeneration(
        token_ids=ids,
        stopped=False,
        stop_token_id=None,
        truncated=len(ids) >= max_new_tokens,
    )


def normalize_stop_token_ids(value: int | Iterable[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(int(token_id) for token_id in value)


def stable_batch_seed(base_seed: int, prompt_ids: Iterable[str]) -> int:
    identity = f"{base_seed}|official-thinking|" + "|".join(prompt_ids)
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def detect_huggingface_revision(model_path: Path) -> str | None:
    """Recover the Hub revision from a snapshot path or download metadata."""

    resolved = model_path.resolve()
    if resolved.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", resolved.name):
        return resolved.name

    revisions: set[str] = set()
    metadata_root = resolved / ".cache" / "huggingface" / "download"
    for metadata in metadata_root.glob("*.metadata"):
        try:
            first_line = metadata.read_text().splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if re.fullmatch(r"[0-9a-f]{40}", first_line):
            revisions.add(first_line)
    if len(revisions) > 1:
        raise RuntimeError(f"model directory mixes Hugging Face revisions: {sorted(revisions)}")
    return next(iter(revisions), None)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
