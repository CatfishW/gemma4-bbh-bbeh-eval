"""Model loading and memory-safe log-probability computation.

The checkpoint is multimodal (`Gemma4ForConditionalGeneration`); RL touches only
the text stack. Vision/audio towers are moved to CPU, LoRA targets only
language-model projections, and completion log-probabilities are computed by
chunking rows through the LM head so the [batch, seq, 262k-vocab] logits tensor
is never materialized. A startup self-check compares the chunked path against
the model's own forward on a tiny input.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

LORA_PROJECTION_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_policy_model(
    model_path: str,
    attn_implementation: str = "sdpa",
    load_in_4bit: bool = False,
    device: str = "cuda:0",
):
    import transformers

    kwargs: dict = {"dtype": torch.bfloat16, "attn_implementation": attn_implementation}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": device}

    last_error: Exception | None = None
    model = None
    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(model_path, **kwargs)
            LOGGER.info("loaded model via %s", loader_name)
            break
        except (ValueError, KeyError) as error:
            last_error = error
    if model is None:
        raise RuntimeError(f"could not load {model_path}: {last_error}")

    if not load_in_4bit:
        model.to(device)
    offload_non_text_towers(model, "cpu")
    model.config.use_cache = False
    return model


def offload_non_text_towers(model, device: str) -> None:
    for name in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        for holder in (model, getattr(model, "model", None)):
            if holder is None:
                continue
            tower = getattr(holder, name, None)
            if tower is not None and hasattr(tower, "to"):
                tower.to(device)
                LOGGER.info("moved %s to %s", name, device)


def lora_target_module_names(model) -> list[str]:
    """Exact names of text-stack projection Linears (excludes vision/audio)."""
    excluded = ("vision", "audio", "image", "multi_modal", "multimodal")
    names: list[str] = []
    for name, module in model.named_modules():
        if not name.endswith(LORA_PROJECTION_SUFFIXES):
            continue
        if any(marker in name.lower() for marker in excluded):
            continue
        if module.__class__.__name__ not in (
            "Linear",
            "Linear4bit",
            "Linear8bitLt",
        ):
            continue
        names.append(name)
    if not names:
        raise RuntimeError("found no LoRA target modules")
    return names


def attach_lora(model, rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_module_names(model),
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def unwrap_base(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def find_backbone_and_head(model):
    """Locate (backbone, lm_head, final_logit_softcapping) beneath any wrapping."""
    base = unwrap_base(model)
    head = getattr(base, "lm_head", None)
    backbone = getattr(base, "model", None)
    if head is None or backbone is None:
        raise RuntimeError("unexpected model layout: missing lm_head or .model backbone")
    softcapping = None
    config = getattr(base, "config", None)
    if config is not None and hasattr(config, "get_text_config"):
        softcapping = getattr(config.get_text_config(), "final_logit_softcapping", None)
    return backbone, head, softcapping


def completion_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    rows_per_chunk: int = 128,
) -> torch.Tensor:
    """Sum of log pi(token) over completion positions, per sequence.

    input_ids/attention_mask/completion_mask: [B, L] right-padded. A position t
    with completion_mask=1 is predicted from hidden state t-1. Returns [B].
    """
    backbone, head, softcapping = find_backbone_and_head(model)
    hidden = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).last_hidden_state  # [B, L, H]

    batch_index, position_index = completion_mask.nonzero(as_tuple=True)
    source_positions = position_index - 1
    if (source_positions < 0).any():
        raise ValueError("completion at position 0 has no preceding hidden state")
    selected_hidden = hidden[batch_index, source_positions]  # [N, H]
    target_tokens = input_ids[batch_index, position_index]  # [N]

    pieces: list[torch.Tensor] = []
    for start in range(0, selected_hidden.shape[0], rows_per_chunk):
        rows = selected_hidden[start : start + rows_per_chunk]
        logits = head(rows)
        if softcapping is not None:
            logits = torch.tanh(logits / softcapping) * softcapping
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        chunk_targets = target_tokens[start : start + rows_per_chunk]
        pieces.append(log_probs.gather(1, chunk_targets.unsqueeze(1)).squeeze(1))
    token_log_probs = torch.cat(pieces, dim=0)

    per_sequence = torch.zeros(
        input_ids.shape[0], dtype=token_log_probs.dtype, device=token_log_probs.device
    )
    per_sequence.index_add_(0, batch_index, token_log_probs)
    return per_sequence


@torch.no_grad()
def self_check_log_probs(model, tokenizer, device: str) -> None:
    """Assert the chunked path matches the model's own logits on a tiny input."""
    sample = tokenizer("The answer is 42.", return_tensors="pt").to(device)
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]
    completion_mask = torch.zeros_like(input_ids)
    completion_mask[:, 2:] = 1

    ours = completion_log_probs(model, input_ids, attention_mask, completion_mask)

    base = unwrap_base(model)
    logits = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    reference = (
        log_probs[:, 1:-1].gather(2, input_ids[:, 2:].unsqueeze(-1)).squeeze(-1).sum(dim=1)
    )
    gap = (ours - reference).abs().max().item()
    if gap > 5e-2:
        raise RuntimeError(f"chunked log-prob self-check failed: max gap {gap}")
    LOGGER.info("log-prob self-check passed (max gap %.2e)", gap)
