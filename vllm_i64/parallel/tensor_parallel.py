"""
vllm-i64 :: Tensor Parallelism

Sharding strategy for token-routed models:
  - Routing: replicated (i64, cheap — run on all ranks)
  - Expert weights: sharded on intermediate dimension
  - Attention Q/K/V: sharded on head dimension
  - All-reduce after expert compute and attention output

INL - 2025
"""

import torch
import torch.distributed as dist
from typing import Optional
from dataclasses import dataclass


@dataclass
class TPConfig:
    """Tensor parallel configuration."""
    tp_size: int = 1
    tp_rank: int = 0
    tp_group: Optional[dist.ProcessGroup] = None


_TP_CONFIG: Optional[TPConfig] = None


def init_tp(tp_size: int, rank: int, group: Optional[dist.ProcessGroup] = None):
    """Initialize tensor parallelism."""
    global _TP_CONFIG
    _TP_CONFIG = TPConfig(tp_size=tp_size, tp_rank=rank, tp_group=group)


def get_tp_rank() -> int:
    if _TP_CONFIG is None:
        return 0
    return _TP_CONFIG.tp_rank


def get_tp_world_size() -> int:
    if _TP_CONFIG is None:
        return 1
    return _TP_CONFIG.tp_size


def shard_expert_weights(
    gate_up: torch.Tensor,      # (num_experts, hidden, 2 * inter)
    down: torch.Tensor,         # (num_experts, inter, hidden)
    tp_rank: int,
    tp_size: int,
) -> tuple:
    """
    Shard expert weights for tensor parallelism.

    gate_up: split on last dim (2 * inter → 2 * inter/tp)
    down: split on dim 1 (inter → inter/tp)

    Returns: (gate_up_shard, down_shard)
    """
    num_experts = gate_up.shape[0]
    full_inter = gate_up.shape[2] // 2
    per_tp = full_inter // tp_size
    offset = tp_rank * per_tp

    # gate_up: split gate and up separately, then concat
    gate_full = gate_up[:, :, :full_inter]
    up_full = gate_up[:, :, full_inter:]
    gate_shard = gate_full[:, :, offset:offset + per_tp].contiguous()
    up_shard = up_full[:, :, offset:offset + per_tp].contiguous()
    gate_up_shard = torch.cat([gate_shard, up_shard], dim=2)

    # down: shard on input dim
    down_shard = down[:, offset:offset + per_tp, :].contiguous()

    return gate_up_shard, down_shard


def shard_qkv(
    weight: torch.Tensor,       # (out_features, in_features)
    num_heads: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Shard Q/K/V projection by head."""
    heads_per_rank = num_heads // tp_size
    head_dim = weight.shape[0] // num_heads
    start = tp_rank * heads_per_rank * head_dim
    end = start + heads_per_rank * head_dim
    return weight[start:end, :].contiguous()


def all_reduce_output(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce across TP ranks (after expert compute or attention output)."""
    if _TP_CONFIG is None or _TP_CONFIG.tp_size <= 1:
        return tensor
    dist.all_reduce(tensor, group=_TP_CONFIG.tp_group)
    return tensor
