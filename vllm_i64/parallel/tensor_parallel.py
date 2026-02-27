"""
vllm-i64 :: Tensor Parallelism

Sharding strategy for token-routed models:
  - Routing: replicated on all ranks (i64, cheap)
  - Expert weights: sharded on intermediate dim
  - Attention Q/K/V: sharded on head dimension
  - All-reduce after down_proj and o_proj

Usage:
    init_distributed(tp_size=4)
    model = ComplexityDeepModel(config)
    shard_model(model)

INL - 2025
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional
from dataclasses import dataclass


# =========================================================================
# Global TP state
# =========================================================================

@dataclass
class TPState:
    tp_size: int = 1
    tp_rank: int = 0
    tp_group: Optional[dist.ProcessGroup] = None
    device: str = "cuda:0"


_TP: TPState = TPState()


def init_distributed(tp_size: int = 1, backend: str = "nccl"):
    """
    Initialize distributed + TP.
    Uses RANK / LOCAL_RANK env vars (set by torchrun).
    """
    global _TP

    if tp_size <= 1:
        _TP = TPState(tp_size=1, tp_rank=0, device="cuda:0" if torch.cuda.is_available() else "cpu")
        return

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    tp_group = dist.new_group(list(range(tp_size)))

    _TP = TPState(tp_size=tp_size, tp_rank=rank, tp_group=tp_group, device=device)
    print(f"[TP] rank={rank}/{tp_size} device={device}")


def get_tp() -> TPState:
    return _TP


def get_tp_rank() -> int:
    return _TP.tp_rank


def get_tp_world_size() -> int:
    return _TP.tp_size


# =========================================================================
# Parallel Linear layers
# =========================================================================

class ColumnParallelLinear(nn.Module):
    """
    Output dim sharded across TP ranks.
    Full: (in, out) → Rank k: (in, out // tp)
    For: Q, K, V, gate_up
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        tp = get_tp()
        self.tp_size = tp.tp_size
        self.out_per_rank = out_features // tp.tp_size
        self.linear = nn.Linear(in_features, self.out_per_rank, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def load_full_weight(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None):
        """Take our TP slice from unsharded weight."""
        tp = get_tp()
        start = tp.tp_rank * self.out_per_rank
        end = start + self.out_per_rank
        self.linear.weight.data.copy_(full_weight[start:end])
        if full_bias is not None and self.linear.bias is not None:
            self.linear.bias.data.copy_(full_bias[start:end])


class RowParallelLinear(nn.Module):
    """
    Input dim sharded across TP ranks. Output all-reduced.
    Full: (in, out) → Rank k: (in // tp, out) + all_reduce
    For: o_proj, down_proj
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        tp = get_tp()
        self.tp_size = tp.tp_size
        self.in_per_rank = in_features // tp.tp_size
        self.linear = nn.Linear(self.in_per_rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.tp_size > 1:
            dist.all_reduce(out, group=get_tp().tp_group)
        return out

    def load_full_weight(self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None):
        """Take our TP slice from unsharded weight."""
        tp = get_tp()
        start = tp.tp_rank * self.in_per_rank
        end = start + self.in_per_rank
        self.linear.weight.data.copy_(full_weight[:, start:end])
        if full_bias is not None and self.linear.bias is not None:
            if tp.tp_rank == 0:
                self.linear.bias.data.copy_(full_bias)
            else:
                self.linear.bias.data.zero_()


# =========================================================================
# Expert weight sharding
# =========================================================================

def shard_expert_weights(
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> tuple:
    """
    Shard expert weights for current TP rank.
    gate_up: split gate and up on intermediate dim
    down: split on input dim
    """
    tp = get_tp()
    if tp.tp_size <= 1:
        return gate_up, down

    full_inter = gate_up.shape[2] // 2
    per_tp = full_inter // tp.tp_size
    offset = tp.tp_rank * per_tp

    gate_shard = gate_up[:, :, offset:offset + per_tp].contiguous()
    up_shard = gate_up[:, :, full_inter + offset:full_inter + offset + per_tp].contiguous()
    gate_up_shard = torch.cat([gate_shard, up_shard], dim=2)

    down_shard = down[:, offset:offset + per_tp, :].contiguous()

    return gate_up_shard, down_shard


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce across TP group."""
    if _TP.tp_size <= 1:
        return tensor
    dist.all_reduce(tensor, group=_TP.tp_group)
    return tensor
