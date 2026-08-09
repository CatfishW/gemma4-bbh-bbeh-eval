"""Batched rollout generation with the deployed chat convention.

Requests contain exactly one user message and no system message, matching the
frozen evaluation protocol. Sequences are bucketed by prompt length; each bucket
is left-padded, generated in one `generate` call, and unpadded afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from eval_benchmarks import Example, build_prompt


@dataclass
class RolloutRequest:
    prompt_id: str
    example: Example
    prompt_token_ids: list[int]


@dataclass
class RolloutResult:
    prompt_id: str
    example: Example
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_text: str


def chat_prompt_token_ids(tokenizer, example: Example, prompt_strategy: str) -> list[int]:
    rendered = build_prompt(example.input, prompt_strategy)
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": rendered}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        return list(encoded["input_ids"])
    return list(encoded)


def batch_seed(base_seed: int, iteration: int, batch_index: int) -> int:
    identity = f"{base_seed}|rollout|{iteration}|{batch_index}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def plan_batches(
    requests: list[RolloutRequest],
    batch_size: int,
    max_batch_tokens: int = 16384,
) -> list[list[RolloutRequest]]:
    """Length-sorted batches capped by count and by padded-token volume."""
    ordered = sorted(requests, key=lambda r: len(r.prompt_token_ids))
    batches: list[list[RolloutRequest]] = []
    current: list[RolloutRequest] = []
    for request in ordered:
        longest = len(request.prompt_token_ids)
        if current and (
            len(current) >= batch_size or longest * (len(current) + 1) > max_batch_tokens
        ):
            batches.append(current)
            current = []
        current.append(request)
    if current:
        batches.append(current)
    return batches


@torch.no_grad()
def generate_rollouts(
    model,
    tokenizer,
    requests: list[RolloutRequest],
    *,
    iteration: int,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    max_batch_tokens: int = 16384,
    device: str = "cuda:0",
) -> list[RolloutResult]:
    was_training = model.training
    model.eval()
    results: list[RolloutResult] = []
    pad_id = tokenizer.pad_token_id
    for batch_index, batch in enumerate(plan_batches(requests, batch_size, max_batch_tokens)):
        longest = max(len(r.prompt_token_ids) for r in batch)
        input_ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for row, request in enumerate(batch):
            ids = request.prompt_token_ids
            input_ids[row, longest - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, longest - len(ids) :] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        torch.manual_seed(batch_seed(seed, iteration, batch_index))
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            top_p=top_p if temperature > 0.0 else None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            use_cache=True,
        )
        completions = generated[:, longest:]
        for row, request in enumerate(batch):
            token_ids = completions[row].tolist()
            if tokenizer.eos_token_id in token_ids:
                token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id) + 1]
            trimmed = [t for t in token_ids if t != pad_id] if pad_id != tokenizer.eos_token_id else token_ids
            text = tokenizer.decode(trimmed, skip_special_tokens=True)
            results.append(
                RolloutResult(
                    prompt_id=request.prompt_id,
                    example=request.example,
                    prompt_token_ids=request.prompt_token_ids,
                    completion_token_ids=trimmed,
                    completion_text=text,
                )
            )
    if was_training:
        model.train()
    return results
