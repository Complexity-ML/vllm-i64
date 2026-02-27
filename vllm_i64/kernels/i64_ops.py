"""
vllm-i64 :: Python fallback for i64 CUDA kernels.

Pure PyTorch implementation for when CUDA kernels aren't compiled.
Mirrors the C/CUDA kernels in csrc/ exactly.

All routing/scheduling is integer. FP16 only in expert compute.

INL - 2025
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def i64_route_tokens(
    token_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Route tokens to experts using integer bit masking.

    Equivalent to csrc/i64_router.cu::i64_route_tokens_kernel

    Args:
        token_ids: [num_tokens] int64
        num_experts: power of 2

    Returns:
        expert_ids: [num_tokens] int32
    """
    expert_mask = num_experts - 1
    return (token_ids & expert_mask).to(torch.int32)


def i64_scatter(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Scatter tokens by expert into contiguous buffers.

    Equivalent to csrc/i64_router.cu::i64_scatter_kernel

    Args:
        hidden_states: [num_tokens, hidden_dim] fp16
        expert_ids: [num_tokens] int32
        num_experts: int

    Returns:
        scattered: [num_tokens, hidden_dim] fp16 (reordered by expert)
        scatter_indices: [num_tokens] int32 (where each token went)
        expert_offsets: [num_experts] int32 (start of each expert's segment)
        expert_counts: [num_experts] int32 (tokens per expert)
    """
    # Count per expert (integer)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=hidden_states.device)
    for e in range(num_experts):
        expert_counts[e] = (expert_ids == e).sum().to(torch.int32)

    # Prefix sum (integer)
    expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=hidden_states.device)
    for e in range(1, num_experts):
        expert_offsets[e] = expert_offsets[e - 1] + expert_counts[e - 1]

    # Scatter (integer indexing, fp16 data copy)
    sorted_indices = torch.argsort(expert_ids)  # Integer sort
    scattered = hidden_states[sorted_indices]

    # Build reverse mapping
    scatter_indices = torch.zeros_like(sorted_indices)
    scatter_indices[sorted_indices] = torch.arange(
        len(sorted_indices), dtype=sorted_indices.dtype, device=hidden_states.device
    )

    return scattered, scatter_indices.to(torch.int32), expert_offsets, expert_counts


def i64_gather(
    expert_output: torch.Tensor,
    scatter_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Gather expert outputs back to original token order.

    Equivalent to csrc/i64_router.cu::i64_gather_kernel

    Args:
        expert_output: [num_tokens, hidden_dim] fp16 (scattered order)
        scatter_indices: [num_tokens] int32

    Returns:
        output: [num_tokens, hidden_dim] fp16 (original order)
    """
    return expert_output[scatter_indices]


def i64_expert_forward(
    scattered: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_counts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Execute expert MLP on scattered tokens.

    Integer control flow, fp16 compute.

    Equivalent to csrc/i64_expert_dispatch.cu::i64_expert_dispatch

    Args:
        scattered: [num_tokens, hidden_dim] fp16
        gate_up_weights: [num_experts, hidden_dim, 2*expert_inter] fp16
        down_weights: [num_experts, expert_inter, hidden_dim] fp16
        expert_offsets: [num_experts] int32
        expert_counts: [num_experts] int32
        num_experts: int

    Returns:
        output: [num_tokens, hidden_dim] fp16
    """
    output = torch.zeros_like(scattered)
    expert_inter = down_weights.shape[1]

    for e in range(num_experts):  # Integer loop
        count = int(expert_counts[e])  # Integer
        offset = int(expert_offsets[e])  # Integer

        if count == 0:  # Integer comparison
            continue

        # Integer slice indices
        tokens_e = scattered[offset:offset + count]

        # --- FLOAT ZONE ---
        w_gu = gate_up_weights[e]   # [hidden_dim, 2*expert_inter]
        w_d = down_weights[e]       # [expert_inter, hidden_dim]

        # Gate+Up fused matmul
        gate_up = tokens_e @ w_gu   # [count, 2*expert_inter]
        gate = gate_up[:, :expert_inter]
        up = gate_up[:, expert_inter:]

        # SiLU + Hadamard
        intermediate = F.silu(gate) * up

        # Down projection
        result = intermediate @ w_d   # [count, hidden_dim]
        # --- END FLOAT ZONE ---

        # Integer indexing for output placement
        output[offset:offset + count] = result

    return output


def i64_full_pipeline(
    token_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Full i64 pipeline: route → scatter → expert → gather.

    Integer operations: route, scatter indices, gather indices, loop control
    Float operations: expert MLP matmuls and SiLU activation ONLY

    Args:
        token_ids: [num_tokens] int64
        hidden_states: [num_tokens, hidden_dim] fp16
        gate_up_weights: [num_experts, hidden_dim, 2*expert_inter] fp16
        down_weights: [num_experts, expert_inter, hidden_dim] fp16
        num_experts: int

    Returns:
        output: [num_tokens, hidden_dim] fp16
    """
    # 1. ROUTE (i64)
    expert_ids = i64_route_tokens(token_ids, num_experts)

    # 2. SCATTER (i64 indexing, fp16 data)
    scattered, scatter_indices, expert_offsets, expert_counts = i64_scatter(
        hidden_states, expert_ids, num_experts
    )

    # 3. EXPERT MLP (fp16 compute, i64 control)
    expert_output = i64_expert_forward(
        scattered, gate_up_weights, down_weights,
        expert_offsets, expert_counts, num_experts,
    )

    # 4. GATHER (i64 indexing)
    output = i64_gather(expert_output, scatter_indices)

    return output
