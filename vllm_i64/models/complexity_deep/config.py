"""
vllm-i64 :: Complexity Deep Config

Config specific to Complexity Deep / Pacific-Prime models.
Mirrors checkpoints/*/config.json.

INL - 2025
"""

import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class ComplexityDeepConfig:
    """
    Complexity Deep model config.
    Mirrors checkpoints/pacific-prime-chat/config.json.
    """
    # Architecture
    model_type: str = "complexity-deep"
    architecture: str = "DeepForCausalLM"
    version: str = "0.13.0"

    # Dimensions
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8          # GQA

    # Positions
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0

    # Norms & activation
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "silu"

    # Embeddings
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    # Token IDs
    pad_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 0

    # Token-Routed MLP (i64)
    use_token_routed_mlp: bool = True
    num_experts: int = 4

    # Attention features
    use_qk_norm: bool = True
    use_sdpa: bool = True
    sliding_window: Optional[int] = None

    # INL Dynamics
    dynamics_alpha: float = 0.9
    dynamics_beta: float = 0.1
    dynamics_gate: float = 0.5
    dynamics_dt: float = 0.1
    dynamics_controller_hidden: int = 64

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def expert_intermediate_size(self) -> int:
        return self.intermediate_size // self.num_experts

    @staticmethod
    def from_json(path: str) -> "ComplexityDeepConfig":
        """Load from a checkpoint config.json."""
        with open(path, "r") as f:
            data = json.load(f)
        config = ComplexityDeepConfig()
        for key, val in data.items():
            if key in ("parameters", "innovations"):
                continue
            if hasattr(config, key):
                setattr(config, key, val)
        return config
