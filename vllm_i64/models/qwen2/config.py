"""
vllm-i64 :: Qwen2 Config

Wraps HuggingFace Qwen2Config + i64 fields.
dense = modulo 1 in i64 routing theory.

INL - 2025
"""

import json
from transformers import Qwen2Config as _HFQwen2Config


class Qwen2Config(_HFQwen2Config):
    """Qwen2 config with i64 routing. num_experts=1 — dense = token_id % 1 = 0."""

    model_type = "qwen2"

    def __init__(self, **kwargs):
        kwargs.setdefault("use_qk_norm", False)
        self.num_experts = kwargs.pop("num_experts", 1)
        self.use_qk_norm = kwargs.pop("use_qk_norm")
        self.attention_bias = kwargs.pop("attention_bias", True)
        super().__init__(**kwargs)
        if not getattr(self, 'head_dim', None):
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str) -> "Qwen2Config":
        with open(path, "r") as f:
            data = json.load(f)
        config = cls(**data)
        config.num_experts = 1
        return config
