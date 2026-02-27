"""
vllm-i64 :: Model Registry

Maps model names to their classes.
All registered models must use i64 token routing.

INL - 2025
"""

from typing import Dict, Type
from vllm_i64.models.token_routed_model import TokenRoutedModel, TokenRoutedConfig


# =========================================================================
# Pre-defined model configurations (all sizes are integers)
# =========================================================================

MODEL_CONFIGS: Dict[str, TokenRoutedConfig] = {
    "inl-125m": TokenRoutedConfig(
        vocab_size=32000,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        num_experts=4,
        max_seq_len=2048,
        head_dim=64,
        expert_inter=192,
    ),
    "inl-350m": TokenRoutedConfig(
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        num_experts=4,
        max_seq_len=2048,
        head_dim=64,
        expert_inter=256,
    ),
    "inl-760m": TokenRoutedConfig(
        vocab_size=32000,
        hidden_dim=1536,
        num_layers=24,
        num_heads=24,
        num_experts=8,
        max_seq_len=4096,
        head_dim=64,
        expert_inter=192,
    ),
    "inl-1.3b": TokenRoutedConfig(
        vocab_size=32000,
        hidden_dim=2048,
        num_layers=24,
        num_heads=32,
        num_experts=8,
        max_seq_len=4096,
        head_dim=64,
        expert_inter=256,
    ),
    "inl-7b": TokenRoutedConfig(
        vocab_size=32000,
        hidden_dim=4096,
        num_layers=32,
        num_heads=32,
        num_experts=16,
        max_seq_len=8192,
        head_dim=128,
        expert_inter=256,
    ),
}


def get_model(name: str) -> TokenRoutedModel:
    """Load a pre-defined model by name."""
    if name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    config = MODEL_CONFIGS[name]
    return TokenRoutedModel(config)


def list_models() -> list:
    """List available model names."""
    return list(MODEL_CONFIGS.keys())
