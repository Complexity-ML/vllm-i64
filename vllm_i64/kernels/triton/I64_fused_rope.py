"""
vllm-i64 :: I64_fused_rope

Triton kernel: fused rotary position embedding application.

Standard PyTorch path (5 ops):
    1. Split x into x1, x2
    2. cos lookup + unsqueeze
    3. sin lookup + unsqueeze
    4. x1*cos - x2*sin, x2*cos + x1*sin  (4 element-wise ops)
    5. Concatenate result

Fused path (1 kernel):
    Load x, cos, sin → rotate all heads at once → store

Eliminates intermediate tensors and multiple kernel launches.
Works for both float cos/sin and Q14 integer cos/sin.

INL - 2025
"""

import torch
from typing import Optional

try:
    import triton
    import triton.language as tl
    _TRITON = True
except ImportError:
    _TRITON = False


if _TRITON:
    @triton.jit
    def _I64_rope_kernel(
        # Input/output
        x_ptr,           # (N, num_heads, head_dim) float
        cos_ptr,         # (N, head_dim) float
        sin_ptr,         # (N, head_dim) float
        out_ptr,         # (N, num_heads, head_dim) float
        # Params
        N,
        num_heads,
        head_dim,
        half_dim,        # head_dim // 2
        stride_x_n, stride_x_h, stride_x_d,
        stride_cs_n, stride_cs_d,
        stride_o_n, stride_o_h, stride_o_d,
        BLOCK_D: tl.constexpr,
    ):
        """
        One program per (token, head). Applies rotary embedding to one head.
        """
        pid = tl.program_id(0)
        token_id = pid // num_heads
        head_id = pid % num_heads

        if token_id >= N:
            return

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < half_dim

        # Load x1 = x[..., :half_dim], x2 = x[..., half_dim:]
        x1_ptrs = x_ptr + token_id * stride_x_n + head_id * stride_x_h + d_offsets * stride_x_d
        x2_ptrs = x_ptr + token_id * stride_x_n + head_id * stride_x_h + (d_offsets + half_dim) * stride_x_d

        x1 = tl.load(x1_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        # Load cos, sin (first half of head_dim)
        cos_ptrs = cos_ptr + token_id * stride_cs_n + d_offsets * stride_cs_d
        sin_ptrs = sin_ptr + token_id * stride_cs_n + d_offsets * stride_cs_d

        cos = tl.load(cos_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        # Rotate
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin

        # Store
        o1_ptrs = out_ptr + token_id * stride_o_n + head_id * stride_o_h + d_offsets * stride_o_d
        o2_ptrs = out_ptr + token_id * stride_o_n + head_id * stride_o_h + (d_offsets + half_dim) * stride_o_d

        tl.store(o1_ptrs, r1.to(out_ptr.dtype.element_ty), mask=d_mask)
        tl.store(o2_ptrs, r2.to(out_ptr.dtype.element_ty), mask=d_mask)


    @triton.jit
    def _I64_rope_integer_kernel(
        # Input/output
        x_ptr,              # (N, num_heads, head_dim) float
        cos_q14_ptr,        # (N, head_dim) int16 — Q14 cos
        sin_q14_ptr,        # (N, head_dim) int16 — Q14 sin
        out_ptr,            # (N, num_heads, head_dim) float
        # Params
        N,
        num_heads,
        head_dim,
        half_dim,
        Q_ROPE_IN: tl.constexpr,      # 128 (Q7)
        Q_ROPE_SCALE: tl.constexpr,    # 128 * 16384 = dequant divisor
        stride_x_n, stride_x_h, stride_x_d,
        stride_cs_n, stride_cs_d,
        stride_o_n, stride_o_h, stride_o_d,
        BLOCK_D: tl.constexpr,
    ):
        """
        Integer RoPE via Triton: Q7 input × Q14 cos/sin → Q21, dequant.
        Matches CPU apply_rotary_integer() exactly.
        """
        pid = tl.program_id(0)
        token_id = pid // num_heads
        head_id = pid % num_heads

        if token_id >= N:
            return

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < half_dim

        x1_ptrs = x_ptr + token_id * stride_x_n + head_id * stride_x_h + d_offsets * stride_x_d
        x2_ptrs = x_ptr + token_id * stride_x_n + head_id * stride_x_h + (d_offsets + half_dim) * stride_x_d

        x1 = tl.load(x1_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        # Quantize input to Q7
        x1_q = tl.extra.cuda.libdevice.rint(x1 * Q_ROPE_IN).to(tl.int32)
        x2_q = tl.extra.cuda.libdevice.rint(x2 * Q_ROPE_IN).to(tl.int32)

        # Load Q14 cos/sin
        cos_ptrs = cos_q14_ptr + token_id * stride_cs_n + d_offsets * stride_cs_d
        sin_ptrs = sin_q14_ptr + token_id * stride_cs_n + d_offsets * stride_cs_d

        cos_q = tl.load(cos_ptrs, mask=d_mask, other=0).to(tl.int32)
        sin_q = tl.load(sin_ptrs, mask=d_mask, other=0).to(tl.int32)

        # Integer rotation: Q7 × Q14 → Q21
        r1_q = x1_q * cos_q - x2_q * sin_q
        r2_q = x2_q * cos_q + x1_q * sin_q

        # Dequant: ÷ (128 × 16384)
        r1 = r1_q.to(tl.float32) / Q_ROPE_SCALE
        r2 = r2_q.to(tl.float32) / Q_ROPE_SCALE

        o1_ptrs = out_ptr + token_id * stride_o_n + head_id * stride_o_h + d_offsets * stride_o_d
        o2_ptrs = out_ptr + token_id * stride_o_n + head_id * stride_o_h + (d_offsets + half_dim) * stride_o_d

        tl.store(o1_ptrs, r1.to(out_ptr.dtype.element_ty), mask=d_mask)
        tl.store(o2_ptrs, r2.to(out_ptr.dtype.element_ty), mask=d_mask)


def triton_fused_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    integer_mode: bool = False,
) -> Optional[torch.Tensor]:
    """
    Fused rotary embedding via Triton.

    Args:
        x: (N, num_heads, head_dim) float
        cos: (N, head_dim) float or int16 (Q14 if integer_mode)
        sin: (N, head_dim) float or int16
        integer_mode: if True, cos/sin are Q14 INT16

    Returns:
        rotated: (N, num_heads, head_dim) float
        None if Triton not available
    """
    if not (_TRITON and x.is_cuda):
        return None

    N, num_heads, head_dim = x.shape
    half_dim = head_dim // 2
    BLOCK_D = triton.next_power_of_2(half_dim)

    out = torch.empty_like(x)
    grid = (N * num_heads,)

    if integer_mode:
        _I64_rope_integer_kernel[grid](
            x, cos, sin, out,
            N, num_heads, head_dim, half_dim,
            Q_ROPE_IN=128,
            Q_ROPE_SCALE=128 * 16384,
            stride_x_n=x.stride(0), stride_x_h=x.stride(1), stride_x_d=x.stride(2),
            stride_cs_n=cos.stride(0), stride_cs_d=cos.stride(1),
            stride_o_n=out.stride(0), stride_o_h=out.stride(1), stride_o_d=out.stride(2),
            BLOCK_D=BLOCK_D,
        )
    else:
        _I64_rope_kernel[grid](
            x, cos, sin, out,
            N, num_heads, head_dim, half_dim,
            stride_x_n=x.stride(0), stride_x_h=x.stride(1), stride_x_d=x.stride(2),
            stride_cs_n=cos.stride(0), stride_cs_d=cos.stride(1),
            stride_o_n=out.stride(0), stride_o_h=out.stride(1), stride_o_d=out.stride(2),
            BLOCK_D=BLOCK_D,
        )

    return out
