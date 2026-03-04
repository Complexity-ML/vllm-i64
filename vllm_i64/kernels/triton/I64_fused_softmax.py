"""
vllm-i64 :: I64_fused_softmax

Triton kernel: integer softmax using Q7 LUT — matches CPU integer softmax exactly.

Standard softmax: exp(x - max) / sum(exp) — float transcendentals, expensive.

Integer softmax (I64):
    1. Quantize logits to Q7 (×128 → INT32)
    2. Subtract row-max (all ≤ 0, stability)
    3. Clamp to [-1024, 0] — below that exp() ≈ 0
    4. LUT exp(): 1025-entry table, exp(idx/128) × 2^16
    5. Normalize: w_i = exp_i / sum(exp)

Eliminates float exp() entirely — replaced by table lookup.

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

# Integer softmax constants (must match moe.py)
_Q_IN = 128
_Q_OUT = 1 << 16
_LUT_MIN = -1024
_LUT_SIZE = -_LUT_MIN + 1  # 1025


def _build_exp_lut_gpu(device: torch.device) -> torch.Tensor:
    """Build exp() LUT on GPU for Triton kernel."""
    indices = torch.arange(_LUT_MIN, 1, dtype=torch.float32, device=device)
    return (torch.exp(indices / _Q_IN) * _Q_OUT).to(torch.int32)


_GPU_LUT_CACHE = {}


def _get_gpu_lut(device: torch.device) -> torch.Tensor:
    """Cached LUT per device."""
    key = str(device)
    if key not in _GPU_LUT_CACHE:
        _GPU_LUT_CACHE[key] = _build_exp_lut_gpu(device)
    return _GPU_LUT_CACHE[key]


if _TRITON:
    @triton.jit
    def _I64_softmax_integer_kernel(
        # Inputs
        logits_ptr,      # (N, D) float
        lut_ptr,         # (LUT_SIZE,) int32
        # Output
        out_ptr,         # (N, D) float
        # Params
        N, D,
        stride_l_n, stride_l_d,
        stride_o_n, stride_o_d,
        Q_IN: tl.constexpr,
        LUT_MIN: tl.constexpr,
        LUT_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        One program per row. Full row loaded, integer softmax computed,
        float result stored.
        """
        row = tl.program_id(0)
        if row >= N:
            return

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        # Load logits row
        l_ptrs = logits_ptr + row * stride_l_n + d_offsets * stride_l_d
        logits = tl.load(l_ptrs, mask=d_mask, other=-1e9).to(tl.float32)

        # Step 1: Quantize to Q7
        logits_q = tl.extra.cuda.libdevice.rint(logits * Q_IN).to(tl.int32)

        # Step 2: Subtract max for stability
        row_max = tl.max(logits_q, axis=0)
        shifted = logits_q - row_max

        # Step 3: Clamp to LUT range
        shifted = tl.maximum(shifted, LUT_MIN)
        shifted = tl.minimum(shifted, 0)

        # Step 4: LUT lookup — index = shifted - LUT_MIN
        lut_idx = shifted - LUT_MIN
        exp_vals = tl.load(lut_ptr + lut_idx, mask=d_mask, other=0)

        # Step 5: Normalize (integer division → float for precision)
        exp_sum = tl.sum(exp_vals, axis=0)
        exp_sum = tl.maximum(exp_sum, 1)  # avoid div by zero
        weights = exp_vals.to(tl.float32) / exp_sum.to(tl.float32)

        # Store
        o_ptrs = out_ptr + row * stride_o_n + d_offsets * stride_o_d
        tl.store(o_ptrs, weights.to(out_ptr.dtype.element_ty), mask=d_mask)


def triton_fused_softmax_integer(
    logits: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Integer softmax via Triton — Q7 LUT-based, matches CPU path exactly.

    Args:
        logits: (N, D) float — attention scores or gate logits

    Returns:
        weights: (N, D) float — softmax probabilities
        None if Triton not available
    """
    if not (_TRITON and logits.is_cuda):
        return None

    N, D = logits.shape
    BLOCK_D = triton.next_power_of_2(D)

    lut = _get_gpu_lut(logits.device)
    out = torch.empty_like(logits)

    grid = (N,)
    _I64_softmax_integer_kernel[grid](
        logits, lut,
        out,
        N, D,
        logits.stride(0), logits.stride(1),
        out.stride(0), out.stride(1),
        Q_IN=_Q_IN,
        LUT_MIN=_LUT_MIN,
        LUT_SIZE=_LUT_SIZE,
        BLOCK_D=BLOCK_D,
    )

    return out
