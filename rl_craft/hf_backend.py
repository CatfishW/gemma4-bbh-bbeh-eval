"""Pinned-local Hugging Face/PEFT bridge using the existing rl.modeling loader.

No remote checkpoint downloads, external teacher, top-p truncation, or hidden
sampling penalties. All stochastic token log-probabilities use the SAME
full-support temperature as sampling. Head checkpointing limits retained logits.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .core import Config, Segment


class HFBackend:
    def __init__(self, model_path: str, cfg: Config, device: str = "cuda:0", adapter: str | None = None):
        from rl.modeling import attach_lora, load_policy_model, load_tokenizer, find_backbone_and_head
        path = Path(model_path).resolve(strict=True)
        if not path.is_dir() or not (path/"config.json").is_file():
            raise ValueError("an existing local checkpoint directory is required")
        self.config, self.device = cfg, torch.device(device)
        self.tokenizer = load_tokenizer(str(path))
        base = load_policy_model(str(path), attn_implementation="sdpa", load_in_4bit=False, device=device)
        if adapter is None:
            self.model = attach_lora(base, cfg.rank, cfg.alpha, 0.0)
        else:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, str(Path(adapter).resolve(strict=True)), is_trainable=True)
            configs = list(self.model.peft_config.values())
            if len(configs) != 1 or configs[0].r != cfg.rank or configs[0].lora_alpha != cfg.alpha or configs[0].lora_dropout != 0:
                raise ValueError("adapter rank/alpha/dropout mismatch")
        self.model.eval()
        self.backbone, self.head, self.softcap = find_backbone_and_head(self.model)
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.end_ids = [eos] if type(eos) is int else list(eos or [])
        if not self.end_ids or any(type(i) is not int or i < 0 for i in self.end_ids):
            raise ValueError("explicit model termination tokens are required")
        self.pad = self.tokenizer.pad_token_id
        if self.pad is None:
            self.pad = self.end_ids[0]
        self.gate_ids = []
        for label in ("F", "R"):
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.decode(ids, skip_special_tokens=False) != label:
                raise ValueError("F/R must each be one existing, exactly decodable token; no vocabulary mutation")
            self.gate_ids.append(ids[0])
        if len(set(self.gate_ids)) != 2:
            raise ValueError("gate tokens collide")
        self.parse_failures = 0
        self.self_check()

    def prompt(self, stage: str, question: str, notes: str = "") -> tuple[int, ...]:
        instructions = {
            "notes": "Write brief working notes for this question. Preserve the actual premises.\nQUESTION:\n"+question,
            "continue": "Continue these working notes. Resolve remaining uncertainty; avoid repeating notes.\nQUESTION:\n"+question+"\nNOTES:\n"+notes,
            "gate": "Decide whether the notes suffice to answer. Output F to finalize now, or R for more reasoning.\nQUESTION:\n"+question+"\nNOTES:\n"+notes+"\nDECISION:",
            "answer": "Using the question and notes, output only the final answer in the requested format. No reasoning.\nQUESTION:\n"+question+"\nNOTES:\n"+notes,
        }
        if stage not in instructions:
            raise ValueError("unknown protocol stage")
        ids = self.tokenizer.apply_chat_template([{"role": "user", "content": instructions[stage]}],
                                                tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        result = tuple(ids)
        if not result or any(type(t) is not int for t in result) or len(result) >= self.config.max_context:
            raise ValueError("invalid or overlong rendered prompt; no auto-truncation")
        return result

    def _inputs(self, ids):
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        return tokens, torch.ones_like(tokens)

    def _head(self, hidden):
        logits = self.head(hidden)
        if self.softcap is not None:
            logits = torch.tanh(logits/self.softcap)*self.softcap
        return logits

    def last_logits(self, context):
        tokens, mask = self._inputs(context)
        hidden = self.backbone(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
        return self._head(hidden[0, -1]).float()

    def gate_log_probs(self, context, temperature):
        # Renormalized two-action policy is explicit, and identical at deployment.
        return (self.last_logits(context)[self.gate_ids].double()/temperature).log_softmax(-1)

    def generation_config(self, cap, temperature):
        from transformers import GenerationConfig
        return GenerationConfig(do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                                typical_p=1.0, repetition_penalty=1.0,
                                max_new_tokens=cap, pad_token_id=self.pad,
                                eos_token_id=self.end_ids, use_cache=True)

    def sample(self, context, cap, temperature, seed):
        if not context or cap < 1 or len(context)+cap > self.config.max_context:
            raise ValueError("context/generation cap exceeded; no dropping or retrying hard examples")
        tokens, mask = self._inputs(context)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            generated = self.model.generate(input_ids=tokens, attention_mask=mask,
                                            generation_config=self.generation_config(cap, temperature))
        ids = generated[0, len(context):].tolist()
        terminated = False
        for i, token in enumerate(ids):
            if token in self.end_ids:
                ids, terminated = ids[:i+1], True
                break
        return Segment(tuple(context), tuple(ids), terminated)

    def decode(self, tokens, final=False):
        if final and callable(getattr(self.tokenizer, "parse_response", None)):
            raw = self.tokenizer.decode(list(tokens), skip_special_tokens=False)
            try:
                parsed = self.tokenizer.parse_response(raw)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("content"), str):
                    raise ValueError("unexpected channel parser output")
                return parsed["content"]
            except (ValueError, KeyError, TypeError, IndexError):
                self.parse_failures += 1
                return ""  # never concatenate thought and final channels to earn reward
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True)

    def log_prob(self, segment, temperature):
        full = segment.context + segment.tokens
        if len(full) > self.config.max_context:
            raise ValueError("teacher-forcing context cap")
        ids, mask = self._inputs(full)
        hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        start = len(segment.context)-1
        rows = hidden[0, start:start+len(segment.tokens)]
        targets = ids[0, len(segment.context):]
        def logps(h, y):
            return -F.cross_entropy(self._head(h).float()/temperature, y, reduction="sum")
        pieces = []
        for i in range(0, len(targets), 32):
            h, y = rows[i:i+32], targets[i:i+32]
            # Recompute vocab logits during backward rather than retaining an
            # entire [sequence, 262k] log-softmax graph across all chunks.
            pieces.append(checkpoint(logps, h, y, use_reentrant=False) if torch.is_grad_enabled() else logps(h, y))
        return torch.stack(pieces).sum()

    def self_check(self):
        """Fail before training if optimized head or generation changes the policy."""
        ctx = self.prompt("notes", "What is 2 + 2?")
        ids, mask = self._inputs(ctx)
        with torch.no_grad():
            raw = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0, -1].float()
            own = self.last_logits(ctx)
            if not torch.allclose(raw, own, atol=0.08, rtol=0.005):
                raise RuntimeError("model backbone/head mismatch")
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                generation = self.generation_config(1, self.config.temperature)
                generation.return_dict_in_generate = True
                generation.output_scores = True
                output = self.model.generate(input_ids=ids, attention_mask=mask,
                                             generation_config=generation)
            sampling = output.scores[0][0].float().log_softmax(-1)
            expected = (own/self.config.temperature).log_softmax(-1)
            if not torch.allclose(sampling, expected, atol=0.08, rtol=0.005):
                raise RuntimeError("sampling and scored distributions differ; refusing off-policy RL")
            segment = Segment(ctx, tuple(output.sequences[0, len(ctx):].tolist()), True)
            got = self.log_prob(segment, self.config.temperature)
            want = expected[segment.tokens[0]]
            if abs(float(got-want)) > 0.08:
                raise RuntimeError("teacher-forced token offset or temperature mismatch")

    def save_adapter(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(path, safe_serialization=True)
