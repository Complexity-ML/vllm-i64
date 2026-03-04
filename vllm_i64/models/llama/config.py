"""
vllm-i64 :: Llama Config

Wraps HuggingFace LlamaConfig + i64 fields.
dense = modulo 1 in i64 routing theory.

INL - 2025
"""

import json
from transformers import LlamaConfig as _HFLlamaConfig


class LlamaConfig(_HFLlamaConfig):
    """Llama config with i64 routing. num_experts=1 — dense = token_id % 1 = 0."""

    model_type = "llama"

    def __init__(self, **kwargs):
        kwargs.setdefault("use_qk_norm", False)
        self.num_experts = kwargs.pop("num_experts", 1)
        self.use_qk_norm = kwargs.pop("use_qk_norm")
        super().__init__(**kwargs)

    @classmethod
    def from_json(cls, path: str) -> "LlamaConfig":
        with open(path, "r") as f:
            data = json.load(f)
        config = cls(**data)
        config.num_experts = 1
        return config
