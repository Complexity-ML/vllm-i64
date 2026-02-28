"""
vllm-i64 :: Pipeline Parallelism Utilities

Model-level utilities for PP:
  - PPMissingLayer: placeholder for layers not on this rank
  - IntermediateTensors: dict wrapper for inter-stage communication
  - make_layers: creates ModuleList with PP-aware layer placement

INL - 2025
"""

import torch
import torch.nn as nn
from typing import Callable, Optional


class PPMissingLayer(nn.Module):
    """Placeholder for decoder layers not assigned to this PP rank."""

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "PPMissingLayer.forward() should never be called. "
            "Check PP layer indexing."
        )


class IntermediateTensors(dict):
    """
    Dict wrapper for tensors passed between PP stages.

    Keys for Complexity Deep:
      - hidden_states: (num_tokens, hidden_size)
      - velocity_states: (num_tokens, hidden_size)
      - mu_prev: (num_tokens, hidden_size) or None
      - mu_residual: (num_tokens, hidden_size) or None
    """
    pass


def make_layers(
    num_layers: int,
    layer_fn: Callable[[int], nn.Module],
) -> tuple:
    """
    Create PP-aware layer list.

    Args:
        num_layers: total number of decoder layers
        layer_fn: callable(layer_idx) → nn.Module

    Returns:
        (start_layer, end_layer, nn.ModuleList)
    """
    from vllm_i64.parallel.pipeline_parallel import get_pp, get_pp_indices

    pp = get_pp()
    start, end = get_pp_indices(num_layers, pp.pp_rank, pp.pp_size)

    layers = nn.ModuleList(
        [PPMissingLayer() for _ in range(start)]
        + [layer_fn(idx) for idx in range(start, end)]
        + [PPMissingLayer() for _ in range(end, num_layers)]
    )

    return start, end, layers
