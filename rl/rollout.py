"""Batched rollout generation with the deployed chat convention.

Requests contain exactly one user message and no system message, matching the
frozen evaluation protocol. Sequences are bucketed by prompt length; each bucket
is left-padded, generated in one `generate` call, and unpadded afterwards. On
CUDA OOM (the training GPU is shared with co-tenant services whose usage
fluctuates), a batch is recursively split in half and retried.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging

import torch

from eval_benchmarks import Example, build_prompt
from rl.memory import is_cuda_out_of_memory

LOGGER = logging.getLogger(__name__)


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
    max_batch_tokens: int = 12288,
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


def _decode_batch(
    model,
    tokenizer,
    batch: list[RolloutRequest],
    *,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: str,
) -> list[RolloutResult]:
    pad_id = tokenizer.pad_token_id
    longest = max(len(r.prompt_token_ids) for r in batch)
    input_ids = torch.full((len(batch), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
    for row, request in enumerate(batch):
        ids = request.prompt_token_ids
        input_ids[row, longest - len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, longest - len(ids) :] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    torch.manual_seed(seed)
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
    results: list[RolloutResult] = []
    for row, request in enumerate(batch):
        token_ids = completions[row].tolist()
        if tokenizer.eos_token_id in token_ids:
            token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id) + 1]
        if pad_id != tokenizer.eos_token_id:
            token_ids = [t for t in token_ids if t != pad_id]
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        results.append(
            RolloutResult(
                prompt_id=request.prompt_id,
                example=request.example,
                prompt_token_ids=request.prompt_token_ids,
                completion_token_ids=token_ids,
                completion_text=text,
            )
        )
    return results


def _decode_with_oom_splitting(
    model,
    tokenizer,
    batch: list[RolloutRequest],
    **kwargs,
) -> list[RolloutResult]:
    try:
        return _decode_batch(model, tokenizer, batch, **kwargs)
    except Exception as error:
        if not is_cuda_out_of_memory(error):
            raise

    # Leave the exception scope before emptying the allocator so its traceback
    # no longer retains tensors from the failed generation call.
    torch.cuda.empty_cache()
    if len(batch) == 1:
        raise RuntimeError("CUDA OOM while generating a single rollout")
    middle = len(batch) // 2
    LOGGER.warning("generation OOM: splitting batch of %d and retrying", len(batch))
    left = _decode_with_oom_splitting(model, tokenizer, batch[:middle], **kwargs)
    right = _decode_with_oom_splitting(model, tokenizer, batch[middle:], **kwargs)
    return left + right


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
    max_batch_tokens: int = 12288,
    device: str = "cuda:0",
) -> list[RolloutResult]:
    was_training = model.training
    model.eval()
    results: list[RolloutResult] = []
    for batch_index, batch in enumerate(plan_batches(requests, batch_size, max_batch_tokens)):
        results.extend(
            _decode_with_oom_splitting(
                model,
                tokenizer,
                batch,
                seed=batch_seed(seed, iteration, batch_index),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        )
    if was_training:
        model.train()
    return results
