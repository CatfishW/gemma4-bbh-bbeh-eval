"""Offline CPU LoRA backend for execution/gradient tests, NOT a Gemma simulator."""
from __future__ import annotations

import torch
from .core import Config, Example, Segment


class ToyLoRA(torch.nn.Module):
    def __init__(self, seed: int = 7, rank: int = 2):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.base = torch.nn.Parameter(torch.randn(16, 2, generator=gen)*0.05, requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.randn(16, rank, generator=gen)*0.2)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, 2))

    def forward(self, index: int):
        return self.base[index] + self.lora_A[index] @ self.lora_B


class ToyBackend:
    def __init__(self, config: Config = Config()):
        self.config = config
        self.model = ToyLoRA(config.seed)

    def prompt(self, stage, question, notes=""):
        cue = int(question[-1])
        s = {"notes": 0, "gate": 1, "continue": 2, "answer": 3}[stage]
        # State depends on the actually sampled prefix, so root credit is tested.
        parity = sum(int(x) for x in notes.split() if x in {"0", "1"}) % 2
        return (s*4 + cue*2 + parity,)

    def sample(self, context, cap, temperature, seed):
        if cap < 1:
            raise ValueError("invalid cap")
        with torch.no_grad():
            p = (self.model(context[0])/temperature).softmax(-1)
            token = int(torch.multinomial(p, 1, generator=torch.Generator().manual_seed(seed)))
        return Segment(tuple(context), (token,), True)

    def decode(self, tokens, final=False):
        return " ".join(map(str, tokens))

    def log_prob(self, segment, temperature):
        return (self.model(segment.context[0])/temperature).log_softmax(-1)[segment.tokens[0]]

    def gate_log_probs(self, context, temperature):
        return (self.model(context[0]).double()/temperature).log_softmax(-1)

    def save_adapter(self, path):
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        torch.save({"lora_A": self.model.lora_A.detach(), "lora_B": self.model.lora_B.detach()}, path/"adapter.pt")

    def load_adapter(self, path):
        from pathlib import Path
        state = torch.load(Path(path)/"adapter.pt", map_location="cpu", weights_only=True)
        with torch.no_grad():
            self.model.lora_A.copy_(state["lora_A"])
            self.model.lora_B.copy_(state["lora_B"])


def examples():
    return [Example(f"synthetic/{t}/{i}", f"task{t}", f"Return bit {i%2}", str(i%2), i)
            for t in range(2) for i in range(4)]
