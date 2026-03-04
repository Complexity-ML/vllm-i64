"""
vllm-i64 :: I64_fused_rmsnorm_quant

Triton kernel: fused RMSNorm + INT8 activation quantization.

Standard PyTorch path (3 kernels, 3 global memory round-trips):
    1. norm = rsqrt(mean(x^2) + eps)    → write normalized x to DRAM
    2. x_norm = x * norm * weight        → write weighted x to DRAM
    3. x_int8 = round(x_norm / scale)    → write INT8 to DRAM

Fused path (1 kernel, 1 read + 1 write):
    Load x → compute variance → rsqrt → weight multiply → quantize → store INT8 + scale

Saves ~2x memory bandwidth. Critical for decode phase where
the pipeline is memory-bound (small batch, large hidden dim).

INL - 2025
"""

import torch
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    _TRITON = True
except ImportError:
    _TRITON = False


if _TRITON:
    @triton.jit
    def _I64_rmsnorm_quant_kernel(
        # Inputs
        x_ptr,           # (N, H) float
        weight_ptr,      # (H,) float — RMSNorm weight
        # Outputs
        out_int8_ptr,    # (N, H) int8
        out_scale_ptr,   # (N,) float — per-token scale
        # Params
        N, H,
        eps: tl.constexpr,
        stride_x_n, stride_x_h,
        stride_o_n, stride_o_h,
        BLOCK_H: tl.constexpr,
    ):
        """One program per token row. Loads full row, normalizes, quantizes."""
        row = tl.program_id(0)
        if row >= N:
            return

        # Load x row
        h_offsets = tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        x_ptrs = x_ptr + row * stride_x_n + h_offsets * stride_x_h
        x = tl.load(x_ptrs, mask=h_mask, other=0.0).to(tl.float32)

        # RMSNorm: rsqrt(mean(x^2) + eps)
        var = tl.sum(x * x, axis=0) / H
        rrms = 1.0 / tl.sqrt(var + eps)

        # Normalize + weight
        w = tl.load(weight_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
        x_norm = x * rrms * w

        # INT8 quantize: per-token symmetric
        abs_max = tl.max(tl.abs(x_norm), axis=0)
        abs_max = tl.maximum(abs_max, 1e-8)
        scale = abs_max / 127.0
        x_q = tl.extra.cuda.libdevice.rint(x_norm / scale)
        x_q = tl.minimum(tl.maximum(x_q, -128.0), 127.0)

        # Store INT8 output
        o_ptrs = out_int8_ptr + row * stride_o_n + h_offsets * stride_o_h
        tl.store(o_ptrs, x_q.to(tl.int8), mask=h_mask)

        # Store per-token scale
        tl.store(out_scale_ptr + row, scale)


    @triton.jit
    def _I64_rmsnorm_kernel(
        # Inputs
        x_ptr,
        weight_ptr,
        # Output (float, not quantized)
        out_ptr,
        # Params
        N, H,
        eps: tl.constexpr,
        stride_x_n, stride_x_h,
        stride_o_n, stride_o_h,
        BLOCK_H: tl.constexpr,
    ):
        """Fused RMSNorm without quantization — for non-INT8 paths."""
        row = tl.program_id(0)
        if row >= N:
            return

        h_offsets = tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        x_ptrs = x_ptr + row * stride_x_n + h_offsets * stride_x_h
        x = tl.load(x_ptrs, mask=h_mask, other=0.0).to(tl.float32)

        var = tl.sum(x * x, axis=0) / H
        rrms = 1.0 / tl.sqrt(var + eps)

        w = tl.load(weight_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
        x_norm = x * rrms * w

        o_ptrs = out_ptr + row * stride_o_n + h_offsets * stride_o_h
        tl.store(o_ptrs, x_norm.to(out_ptr.dtype.element_ty), mask=h_mask)


def triton_fused_rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Fused RMSNorm + INT8 quantization in a single Triton kernel.

    Args:
        x: (N, H) float activations
        weight: (H,) float RMSNorm weight
        eps: epsilon for numerical stability

    Returns:
        (x_int8, scale) — (N, H) int8 + (N,) float per-token scale
        None if Triton not available
    """
    if not (_TRITON and x.is_cuda):
        return None

    N, H = x.shape
    BLOCK_H = triton.next_power_of_2(H)

    out_int8 = torch.empty(N, H, dtype=torch.int8, device=x.device)
    out_scale = torch.empty(N, dtype=torch.float32, device=x.device)

    grid = (N,)
    _I64_rmsnorm_quant_kernel[grid](
        x, weight,
        out_int8, out_scale,
        N, H, eps,
        x.stride(0), x.stride(1),
        out_int8.stride(0), out_int8.stride(1),
        BLOCK_H=BLOCK_H,
    )

    return out_int8, out_scale


def triton_fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """
    Fused RMSNorm (no quantization) via Triton.

    Args:
        x: (N, H) float activations
        weight: (H,) float RMSNorm weight
        eps: epsilon

    Returns:
        x_norm: (N, H) same dtype as x, or None if Triton not available
    """
    if not (_TRITON and x.is_cuda):
        return None

    N, H = x.shape
    BLOCK_H = triton.next_power_of_2(H)

    out = torch.empty_like(x)

    grid = (N,)
    _I64_rmsnorm_kernel[grid](
        x, weight,
        out,
        N, H, eps,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_H=BLOCK_H,
    )

    return out
