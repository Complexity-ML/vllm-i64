"""
vllm-i64 :: Mistral Config

Wraps HuggingFace MistralConfig + i64 fields.
Mistral = Llama + sliding window + GQA (8 KV heads).
dense = modulo 1 in i64 routing theory.

INL - 2025
"""

import json
from transformers import MistralConfig as _HFMistralConfig


class MistralConfig(_HFMistralConfig):
    """Mistral config with i64 routing. num_experts=1 — dense = token_id % 1 = 0."""

    model_type = "mistral"

    def __init__(self, **kwargs):
        kwargs.setdefault("use_qk_norm", False)
        self.num_experts = kwargs.pop("num_experts", 1)
        self.use_qk_norm = kwargs.pop("use_qk_norm")
        super().__init__(**kwargs)
        if not getattr(self, 'head_dim', None):
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str) -> "MistralConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = cls(**data)
        config.num_experts = 1
        return config
