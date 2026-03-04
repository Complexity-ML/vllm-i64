"""
vllm-i64 :: Mixtral Config

Wraps HuggingFace MixtralConfig + i64 fields.
Mixtral = Mistral + sparse MoE (8 experts, top-2 routing).

INL - 2025
"""

import json
from transformers import MixtralConfig as _HFMixtralConfig


class MixtralConfig(_HFMixtralConfig):
    """Mixtral config. num_experts maps to num_local_experts for engine compat."""

    model_type = "mixtral"

    def __init__(self, **kwargs):
        kwargs.setdefault("use_qk_norm", False)
        self.use_qk_norm = kwargs.pop("use_qk_norm")
        super().__init__(**kwargs)
        # Engine-facing: num_experts = actual expert count
        self.num_experts = self.num_local_experts
        if not getattr(self, 'head_dim', None):
            self.head_dim = self.hidden_size // self.num_attention_heads
        if not hasattr(self, 'attention_bias'):
            self.attention_bias = False

    @classmethod
    def from_json(cls, path: str) -> "MixtralConfig":
        with open(path, "r") as f:
            data = json.load(f)
        config = cls(**data)
        return config
