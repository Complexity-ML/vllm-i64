"""
vllm-i64 :: Pipeline Parallelism

Distributes model layers across multiple ranks (stages).
Each stage processes a subset of decoder layers.

TP shards within a stage, PP distributes stages across nodes.

Usage:
    init_pp(pp_size=2)
    pp = get_pp()
    start, end = get_pp_indices(num_layers, pp.pp_rank, pp.pp_size)

INL - 2025
"""

import torch
import torch.distributed as dist
from typing import Optional
from dataclasses import dataclass


@dataclass
class PPState:
    pp_size: int = 1
    pp_rank: int = 0
    pp_group: Optional[dist.ProcessGroup] = None


_PP: PPState = PPState()


def init_pp(pp_size: int = 1):
    """Initialize pipeline parallel state."""
    global _PP

    if pp_size <= 1:
        _PP = PPState(pp_size=1, pp_rank=0)
        return

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before init_pp")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # PP rank = rank // tp_size (TP ranks are contiguous within a PP stage)
    from vllm_i64.parallel.tensor_parallel import get_tp
    tp = get_tp()
    pp_rank = rank // tp.tp_size

    pp_group = dist.new_group(list(range(0, world_size, tp.tp_size)))

    _PP = PPState(pp_size=pp_size, pp_rank=pp_rank, pp_group=pp_group)
    print(f"[PP] rank={pp_rank}/{pp_size}")


def get_pp() -> PPState:
    return _PP


def get_pp_rank() -> int:
    return _PP.pp_rank


def get_pp_world_size() -> int:
    return _PP.pp_size


def is_first_pp_rank() -> bool:
    return _PP.pp_rank == 0


def is_last_pp_rank() -> bool:
    return _PP.pp_rank == _PP.pp_size - 1


def get_pp_indices(num_layers: int, pp_rank: int, pp_size: int) -> tuple:
    """
    Compute layer range for a PP rank.

    Distributes layers as evenly as possible across PP ranks.

    Returns:
        (start_layer, end_layer) — exclusive end
    """
    layers_per_rank = num_layers // pp_size
    remainder = num_layers % pp_size

    if pp_rank < remainder:
        start = pp_rank * (layers_per_rank + 1)
        end = start + layers_per_rank + 1
    else:
        start = pp_rank * layers_per_rank + remainder
        end = start + layers_per_rank

    return start, end


def pp_send(tensor: torch.Tensor, dst: int):
    """Send tensor to next PP stage."""
    if _PP.pp_size <= 1:
        return
    dist.send(tensor, dst=dst, group=_PP.pp_group)


def pp_recv(tensor: torch.Tensor, src: int):
    """Receive tensor from previous PP stage."""
    if _PP.pp_size <= 1:
        return
    dist.recv(tensor, src=src, group=_PP.pp_group)
