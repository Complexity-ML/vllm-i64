"""
vllm-i64 :: Fused Expert Dispatch

Optimized expert forward pass for i64 token-routed models.
Replaces Python per-expert loop with batched operations.

Two modes (auto-selected by batch size):
  - BMM mode  (N <= threshold): torch.bmm, fully parallel, zero loops
  - Chunked mode (N > threshold): sort-by-expert, memory-efficient

Since routing is deterministic (token_id % num_experts), no learned
gating is needed — dispatch is pure integer indexing.

INL - 2025
"""

import torch
import torch.nn.functional as F
from typing import Optional

from vllm_i64.kernels.triton_fused_expert import (
    is_triton_available,
    triton_expert_forward,
    triton_expert_forward_int8,
)


def fused_token_routed_forward(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    intermediate_per_tp: int,
    bmm_threshold: int = 48,
) -> torch.Tensor:
    """
    Fused expert forward with SwiGLU activation.

    Args:
        x: (num_tokens, hidden_size) — input hidden states
        gate_up_proj: (num_experts, hidden_size, 2 * intermediate_per_tp)
        down_proj: (num_experts, intermediate_per_tp, hidden_size)
        expert_ids: (num_tokens,) long — expert assignment per token
        num_experts: number of local experts
        intermediate_per_tp: intermediate dimension per TP shard
        bmm_threshold: max tokens for BMM mode (above → chunked)

    Returns:
        output: (num_tokens, hidden_size)
    """
    if x.shape[0] == 0:
        return x

    if x.shape[0] <= bmm_threshold:
        return _bmm_forward(x, gate_up_proj, down_proj, expert_ids,
                            intermediate_per_tp)
    return _chunked_forward(x, gate_up_proj, down_proj, expert_ids,
                            num_experts, intermediate_per_tp)


def _bmm_forward(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
    intermediate_per_tp: int,
) -> torch.Tensor:
    """
    Batched matmul forward — fully parallel across tokens.

    Selects each token's expert weights via indexing, then uses
    torch.bmm for zero-loop computation. Optimal for decode phase
    where batch sizes are small (1-48 tokens).
    """
    # Select each token's expert weights
    sel_gu = gate_up_proj[expert_ids]
    sel_down = down_proj[expert_ids]

    # Gate+Up: (N, 1, H) @ (N, H, 2I) -> (N, 2I)
    gu = torch.bmm(x.unsqueeze(1), sel_gu).squeeze(1)

    # SwiGLU activation
    gate = gu[..., :intermediate_per_tp]
    up = gu[..., intermediate_per_tp:]
    inter = F.silu(gate) * up

    # Down: (N, 1, I) @ (N, I, H) -> (N, H)
    return torch.bmm(inter.unsqueeze(1), sel_down).squeeze(1)


def _chunked_forward(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    intermediate_per_tp: int,
) -> torch.Tensor:
    """
    Chunked forward — groups tokens by expert, processes in bulk.

    More memory-efficient than BMM for large batches (prefill phase).
    Sorts once, splits by expert boundaries.
    """
    N = x.shape[0]

    # Sort by expert for coalesced memory access
    sorted_idx = expert_ids.argsort(stable=True)
    sorted_x = x[sorted_idx]
    sorted_eid = expert_ids[sorted_idx]

    # Expert boundaries
    counts = torch.bincount(sorted_eid, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=x.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    output = torch.empty(N, x.shape[1], device=x.device, dtype=x.dtype)

    use_triton = is_triton_available() and x.is_cuda

    for eid in range(num_experts):
        s = offsets[eid].item()
        e = offsets[eid + 1].item()
        if s == e:
            continue

        chunk = sorted_x[s:e]

        if use_triton:
            # Triton fused SwiGLU: gate+up → silu(gate)*up → down in one kernel
            out = triton_expert_forward(chunk, gate_up_proj[eid], down_proj[eid], e - s)
            if out is not None:
                output[s:e] = out
                continue

        # PyTorch fallback
        gu = chunk @ gate_up_proj[eid]
        gate = gu[..., :intermediate_per_tp]
        up = gu[..., intermediate_per_tp:]
        inter = F.silu(gate) * up
        output[s:e] = inter @ down_proj[eid]

    # Unsort to original token order
    result = torch.empty_like(output)
    result[sorted_idx] = output
    return result


def fused_token_routed_forward_int8(
    x: torch.Tensor,
    gate_up_int8: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_int8: torch.Tensor,
    down_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    intermediate_per_tp: int,
) -> torch.Tensor:
    """
    Fused expert forward with INT8 dequantization on-the-fly.

    Dequantizes per-expert weights before matmul.
    Uses chunked mode (BMM would expand INT8 weights per-token).

    Args:
        gate_up_int8: (num_experts, hidden, 2*inter) int8
        gate_up_scale: (num_experts, hidden) float — per-channel scale
        down_int8: (num_experts, inter, hidden) int8
        down_scale: (num_experts, inter) float — per-channel scale
    """
    N = x.shape[0]
    if N == 0:
        return x

    sorted_idx = expert_ids.argsort(stable=True)
    sorted_x = x[sorted_idx]
    sorted_eid = expert_ids[sorted_idx]

    counts = torch.bincount(sorted_eid, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=x.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    output = torch.empty(N, x.shape[1], device=x.device, dtype=x.dtype)

    use_triton = is_triton_available() and x.is_cuda

    for eid in range(num_experts):
        s = offsets[eid].item()
        e = offsets[eid + 1].item()
        if s == e:
            continue

        chunk = sorted_x[s:e]

        if use_triton:
            out = triton_expert_forward_int8(
                chunk,
                gate_up_int8[eid], gate_up_scale[eid],
                down_int8[eid], down_scale[eid],
                e - s,
            )
            if out is not None:
                output[s:e] = out
                continue

        # PyTorch fallback: dequantize + matmul
        w_gu = gate_up_int8[eid].float() * gate_up_scale[eid].unsqueeze(-1)
        w_gu = w_gu.to(x.dtype)

        gu = chunk @ w_gu
        gate = gu[..., :intermediate_per_tp]
        up = gu[..., intermediate_per_tp:]
        inter = F.silu(gate) * up

        w_d = down_int8[eid].float() * down_scale[eid].unsqueeze(-1)
        w_d = w_d.to(x.dtype)

        output[s:e] = inter @ w_d

    result = torch.empty_like(output)
    result[sorted_idx] = output
    return result


def fused_token_routed_forward_int4(
    x: torch.Tensor,
    gate_up_int4: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_zero: torch.Tensor,
    down_int4: torch.Tensor,
    down_scale: torch.Tensor,
    down_zero: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
    intermediate_per_tp: int,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Fused expert forward with INT4 dequantization on-the-fly.

    Unpacks packed uint8 (2 values per byte), dequantizes per-group,
    then runs SwiGLU. Chunked mode only.

    Args:
        gate_up_int4: (num_experts, out, in//2) uint8 — packed
        gate_up_scale: (num_experts, out, num_groups) float
        gate_up_zero: (num_experts, out, num_groups) float
        down_int4: (num_experts, out, in//2) uint8 — packed
        down_scale: (num_experts, out, num_groups) float
        down_zero: (num_experts, out, num_groups) float
    """
    from vllm_i64.core.quantization import dequantize_int4

    N = x.shape[0]
    if N == 0:
        return x

    sorted_idx = expert_ids.argsort(stable=True)
    sorted_x = x[sorted_idx]
    sorted_eid = expert_ids[sorted_idx]

    counts = torch.bincount(sorted_eid, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=x.device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    output = torch.empty(N, x.shape[1], device=x.device, dtype=x.dtype)

    for eid in range(num_experts):
        s = offsets[eid].item()
        e = offsets[eid + 1].item()
        if s == e:
            continue

        chunk = sorted_x[s:e]

        # Dequantize gate_up: int4 packed → float
        w_gu = dequantize_int4(
            gate_up_int4[eid], gate_up_scale[eid], gate_up_zero[eid], group_size,
        )
        w_gu = w_gu.reshape(gate_up_int4[eid].shape[0], -1).to(x.dtype)

        gu = chunk @ w_gu
        gate = gu[..., :intermediate_per_tp]
        up = gu[..., intermediate_per_tp:]
        inter = F.silu(gate) * up

        # Dequantize down: int4 packed → float
        w_d = dequantize_int4(
            down_int4[eid], down_scale[eid], down_zero[eid], group_size,
        )
        w_d = w_d.reshape(down_int4[eid].shape[0], -1).to(x.dtype)

        output[s:e] = inter @ w_d

    result = torch.empty_like(output)
    result[sorted_idx] = output
    return result
