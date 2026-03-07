"""
vllm-i64 :: Engine Configuration

Structured configuration for the inference engine.
Replaces scattered getattr(config, ...) with a single typed dataclass.

All fields have sensible defaults — construct with EngineConfig()
for a working configuration, or override specific fields.

INL - 2025
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    """Unified configuration for I64Engine / CPUEngine."""

    # Model
    num_experts: int = 4
    hidden_dim: int = 768
    vocab_size: int = 100_000

    # Scheduling
    max_batch_size: int = 32
    max_seq_len: int = 2048
    max_prefill_tokens: int = 512

    # KV cache
    max_kv_blocks: int = 0          # 0 = auto: max(256, max_batch_size * 8)
    enable_prefix_caching: bool = True
    kv_cache_dtype: Optional[str] = None  # None, "fp8", "fp8_e5m2"

    # Device
    device: str = "cuda"

    # Timeouts
    default_timeout_s: float = 300.0

    # Features
    enable_swap: bool = False
    enable_merge: bool = False

    def resolve_kv_blocks(self) -> int:
        """Resolve auto KV block count."""
        if self.max_kv_blocks <= 0:
            return max(256, self.max_batch_size * 8)
        return self.max_kv_blocks
