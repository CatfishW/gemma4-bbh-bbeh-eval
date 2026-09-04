"""Optional torch utilities: exact label-sequence scoring and activation transport.

No architecture-specific layer names, downloads, training or monkey patches.
Core inference does not import torch. Callers supply an already loaded model and
its pinned tokenizer, and explicitly choose modules/vectors using development data.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import re
from typing import Callable, Sequence

from .executor import Rejected


def choice_views(question: str, views: int = 3) -> list[tuple[str, dict[str, str]]]:
    """Cyclic label permutations of a guarded, final, independent option block.

    The guard rejects common cross-option references; it is NOT a general semantic
    equivalence proof. Use only with independently specified choices and test this
    assumption separately. Map: new label -> original label.
    """
    if not 1 <= views <= 8 or len(question) > 32000:
        raise Rejected("counterfactual budget")
    lines = question.strip().splitlines()
    choices = []
    while lines:
        match = re.fullmatch(r"\s*(?:\(([A-Z])\)|([A-Z])\))\s+(.+?)\s*", lines[-1])
        if not match:
            break
        choices.append((match[1] or match[2], match[3]))
        lines.pop()
    choices.reverse()
    if not 2 <= len(choices) <= 26 or len({k for k, _ in choices}) != len(choices):
        raise Rejected("unrecognized options")
    stem = "\n".join(lines)
    texts = "\n".join(v for _, v in choices)
    if re.search(r"\([A-Z]\)|\b(?:option|choice|answer)\s+[A-Z]\b", stem):
        raise Rejected("stem refers to option labels")
    if re.search(r"\b(?:above|below|both|neither|either)\b|\([A-Z]\)|\b(?:option|choice|answer)\s+[A-Z]\b",
                 texts, re.IGNORECASE):
        raise Rejected("possibly dependent options")
    result = []
    for shift in range(min(views, len(choices))):
        mapping, output = {}, []
        for j, (new_label, _) in enumerate(choices):
            old_label, text = choices[(j + shift) % len(choices)]
            mapping[new_label] = old_label
            output.append(f"({new_label}) {text}")
        result.append((stem + "\n" + "\n".join(output), mapping))
    return result


def counterfactual_score(question: str, scorer: Callable[[str, Sequence[str]], dict[str, float]],
                         views: int = 3) -> dict:
    """Average mapped sequence log probabilities; agreement is not a correctness proof.

    All scoring passes must be included in cost reporting. A finite log probability
    is required for EVERY candidate, not a truncated top-logprob API response.
    """
    totals, winners, records = {}, [], []
    for text, mapping in choice_views(question, views):
        logs = scorer(text, list(mapping))
        if set(logs) != set(mapping) or any(not math.isfinite(v) or v > 1e-6 for v in logs.values()):
            raise Rejected("all candidate sequence log probabilities required")
        mapped = {original: float(logs[new]) for new, original in mapping.items()}
        for label, score in mapped.items():
            totals[label] = totals.get(label, 0.0) + score
        winners.append(max(sorted(mapped), key=mapped.get))
        records.append({"mapping": mapping, "mapped_log_probabilities": mapped})
    averages = {k: v / len(records) for k, v in totals.items()}
    ranking = sorted(averages, key=lambda k: (-averages[k], k))
    return {"prediction": f"({ranking[0]})", "certificate_scope": "none",
            "mean_log_probabilities": averages, "view_winners": winners,
            "unanimous": len(set(winners)) == 1,
            "log_margin": averages[ranking[0]] - averages[ranking[1]], "views": records}


class FrozenSequenceScorer:
    """Teacher-forced, multi-token continuation likelihoods with a frozen model.

    This correctness-first reference batches full sequences, NOT shared KV prefixes.
    It therefore repeats prefill and does not claim a speedup over direct answering.
    The tokenizer explicitly defines label continuations; labels may be multi-token.
    """
    def __init__(self, model, tokenizer, *, max_context: int = 4096,
                 max_labels: int = 26, template_kwargs: dict | None = None):
        import torch
        self.model, self.tokenizer = model, tokenizer
        self.max_context, self.max_labels = max_context, max_labels
        self.template_kwargs = template_kwargs or {}
        if max_context < 1 or max_labels < 2:
            raise ValueError("invalid scorer limits")
        self.model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.device = model.get_input_embeddings().weight.device
        self.torch = torch
        self.telemetry: list[dict] = []

    def __call__(self, question: str, labels: Sequence[str]) -> dict[str, float]:
        import time
        torch = self.torch
        if not 1 <= len(labels) <= self.max_labels or len(set(labels)) != len(labels):
            raise Rejected("label count/uniqueness")
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question + "\nReturn only the option label, without parentheses."}],
            tokenize=True, add_generation_prompt=True, **self.template_kwargs)
        if not isinstance(prefix, list) or not prefix or any(type(t) is not int for t in prefix):
            raise Rejected("tokenizer must return a nonempty list of token ids")
        continuations = [self.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(not ids for ids in continuations):
            raise Rejected("empty label tokenization")
        for label, ids in zip(labels, continuations):
            if self.tokenizer.decode(ids, skip_special_tokens=False) != label:
                raise Rejected("label tokenization is not an exact continuation")
        sequences = [prefix + ids for ids in continuations]
        width = max(map(len, sequences))
        if width > self.max_context:
            raise Rejected("scoring context limit")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise Rejected("pad/eos token required")
        tokens = torch.full((len(labels), width), pad, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(tokens)
        for i, sequence in enumerate(sequences):
            tokens[i, :len(sequence)] = torch.tensor(sequence, device=self.device)
            mask[i, :len(sequence)] = 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
            scores = {}
            for i, (label, ids) in enumerate(zip(labels, continuations)):
                positions = logits[i, len(prefix) - 1:len(prefix) + len(ids) - 1].float()
                targets = torch.tensor(ids, device=self.device)
                scores[label] = float(positions.log_softmax(-1).gather(1, targets[:, None]).sum())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.telemetry.append({"elapsed_seconds": time.perf_counter() - started,
                               "forward_passes": 1, "sequences": len(labels),
                               "prefill_tokens": len(prefix) * len(labels),
                               "continuation_tokens": sum(map(len, continuations)),
                               "padded_token_positions": len(labels) * width})
        return scores


def contrast_vector(positive, negative):
    """Mean activation contrast only: no optimizer, backprop, or weight changes."""
    import torch
    if positive.ndim != 2 or negative.shape != positive.shape or positive.shape[0] < 2:
        raise ValueError("paired [examples, hidden] activations with at least two examples required")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("nonfinite activations")
    with torch.no_grad():
        return (positive.float() - negative.float()).mean(0).detach()


@contextmanager
def activation_transport(model, vectors: dict, *, scale: float = 0.1,
                         gate: Callable[[], bool] = lambda: True):
    """Temporary named-module hooks; no parameter writes and cleanup even on errors.

    Vectors map exact module paths to 1-D tensors. Add only at the final sequence
    position, so use unpadded/single-example generation (NOT a padded scoring batch).
    Modules must output Tensor or a tuple with Tensor first. Calibrate balanced
    contrasts on development data; use random/wrong-task vectors as controls.
    Not safe for concurrent requests sharing this model instance.
    """
    import torch
    if not math.isfinite(scale) or abs(scale) > 10 or not vectors:
        raise ValueError("invalid transport configuration")
    handles = []
    try:
        for name, vector in vectors.items():
            if not isinstance(name, str) or not isinstance(vector, torch.Tensor):
                raise ValueError("explicit module paths and tensor vectors required")
            if vector.ndim != 1 or not torch.isfinite(vector).all():
                raise ValueError("invalid transport vector")
            module = model.get_submodule(name)
            frozen = vector.detach().clone()

            def hook(_module, _inputs, output, v=frozen):
                if not gate():
                    return output
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError("transport requires [1, sequence, hidden] outputs")
                if hidden.shape[-1] != v.numel():
                    raise ValueError("vector hidden dimension mismatch")
                changed = hidden.clone()
                changed[:, -1, :] += scale * v.to(device=hidden.device, dtype=hidden.dtype)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed
            handles.append(module.register_forward_hook(hook))
        with torch.inference_mode():
            yield
    finally:
        for handle in handles:
            handle.remove()
