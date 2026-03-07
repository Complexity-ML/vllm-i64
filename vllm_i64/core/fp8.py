"""
vllm-i64 :: FP8 Quantization & Compute

Native FP8 E4M3 support for Hopper (H100) and Ada (RTX 4090) GPUs.
Uses torch._scaled_mm for hardware-accelerated FP8 tensor core ops.

FP8 E4M3 format:
  - 4 exponent bits, 3 mantissa bits → range [-448, 448]
  - ~2x throughput vs FP16 on H100 tensor cores
  - Per-tensor or per-channel scaling for accuracy

Three entry points:
  - fp8_linear():        single projection (Q, K, V, O, gate, up, down)
  - fp8_fused_gate_up(): fused gate+up with single activation quantize
  - quantize_fp8():      offline weight quantization

Auto-fallback: INT8 path on SM80 (A100), FP16 on older GPUs.

INL - 2025
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

_logger = logging.getLogger("vllm_i64.fp8")

# =========================================================================
# FP8 support detection
# =========================================================================

_FP8_DTYPE = None
_SCALED_MM_AVAILABLE = False

try:
    _FP8_DTYPE = torch.float8_e4m3fn
    if hasattr(torch, '_scaled_mm'):
        # _scaled_mm per-row/per-col scales require PyTorch >= 2.4
        _pt_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if _pt_version >= (2, 4):
            _SCALED_MM_AVAILABLE = True
        else:
            logging.getLogger("vllm_i64.fp8").info(
                "torch._scaled_mm found but PyTorch %s < 2.4 — per-row scales not supported, disabling",
                torch.__version__,
            )
except (AttributeError, ValueError):
    pass


def fp8_available() -> bool:
    """Check if FP8 compute is available on current hardware."""
    if not _SCALED_MM_AVAILABLE or _FP8_DTYPE is None:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability()
        return cap[0] >= 9 or (cap[0] == 8 and cap[1] >= 9)  # SM89+ (Ada/Hopper)
    except Exception:
        return False


def fp8_dtype():
    """Get FP8 E4M3 dtype, or None if unsupported."""
    return _FP8_DTYPE


# =========================================================================
# FP8 quantization
# =========================================================================

_FP8_MAX = 448.0  # E4M3 max representable value


def quantize_fp8(
    weight: torch.Tensor,
    per_channel: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight to FP8 E4M3 with per-channel or per-tensor scaling.

    weight: (out_features, in_features) float
    Returns: (fp8_weight, scale) where scale is per-channel (out,) or scalar (1,)

    FP8 E4M3 range: [-448, 448]. Scale = max(|w|) / 448.
    """
    if _FP8_DTYPE is None:
        raise RuntimeError("FP8 dtype not available (requires PyTorch 2.1+)")

    w = weight.float()

    if per_channel:
        abs_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    else:
        abs_max = w.abs().amax().clamp(min=1e-12).unsqueeze(0).unsqueeze(0)

    scale = abs_max / _FP8_MAX
    w_scaled = w / scale

    # Clamp and cast to FP8
    w_fp8 = w_scaled.clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)

    if per_channel:
        scale = scale.squeeze(-1)  # (out_features,)
    else:
        scale = scale.squeeze()  # scalar

    return w_fp8, scale


def quantize_activations_fp8(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic per-token FP8 quantization of activations.

    x: (tokens, features) float → (tokens, features) fp8 + (tokens, 1) scale
    """
    if _FP8_DTYPE is None:
        raise RuntimeError("FP8 dtype not available")

    x_f32 = x.float()
    abs_max = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = abs_max / _FP8_MAX
    x_scaled = x_f32 / scale
    x_fp8 = x_scaled.clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)

    return x_fp8, scale  # scale: (tokens, 1)


# =========================================================================
# FP8 linear — torch._scaled_mm
# =========================================================================

def fp8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FP8 linear: y = x @ W^T using native FP8 tensor cores.

    Uses torch._scaled_mm (Hopper/Ada):
        1. Quantize activations → FP8 + per-token scale
        2. _scaled_mm(x_fp8, W_fp8^T, x_scale, w_scale) → FP16/BF16 output
        3. Add bias if present

    Falls back to dequant + F.linear on unsupported hardware.

    Args:
        x: (*, in_features) float activations
        weight_fp8: (out_features, in_features) float8_e4m3fn
        weight_scale: (out_features,) float per-channel scale
        bias: optional (out_features,) float
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M = x_2d.shape[0]

    if not (_SCALED_MM_AVAILABLE and x.is_cuda and fp8_available()):
        # Fallback: dequantize FP8 → float, then F.linear
        w_float = weight_fp8.float() * weight_scale.unsqueeze(-1)
        out = F.linear(x_2d.float(), w_float, bias)
        return out.reshape(*orig_shape[:-1], -1)

    # Dynamic FP8 activation quantization
    x_fp8, x_scale = quantize_activations_fp8(x_2d)

    # _scaled_mm: (M, K) @ (N, K)^T → (M, N) with scale factors
    # Per-channel weight scale needs to be (1, N) for _scaled_mm
    w_scale_2d = weight_scale.unsqueeze(0).float()  # (1, out_features)
    x_scale_2d = x_scale.float()                     # (M, 1)

    # torch._scaled_mm expects contiguous inputs
    wt = weight_fp8.t().contiguous()  # (in, out) FP8
    x_fp8_c = x_fp8.contiguous()

    # Scale product: each output[i,j] *= x_scale[i] * w_scale[j]
    # _scaled_mm can take scale_a (per-row) and scale_b (per-col)
    out = torch._scaled_mm(
        x_fp8_c,
        wt,
        scale_a=x_scale_2d,
        scale_b=w_scale_2d,
        out_dtype=x.dtype if x.is_floating_point() else torch.float16,
    )

    if bias is not None:
        out = out + bias

    return out.reshape(*orig_shape[:-1], -1)


def fp8_fused_gate_up(
    x: torch.Tensor,
    fused_fp8: torch.Tensor,
    fused_scale: torch.Tensor,
    inter_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused gate+up projection in FP8: single activation quantize + single matmul.

    gate_fp8 and up_fp8 pre-concatenated along dim 0 into fused_fp8.
    One _scaled_mm does both projections. Output split at inter_size.

    Args:
        x: (*, hidden) float activations
        fused_fp8: (2*inter, hidden) float8_e4m3fn — cat([gate, up], dim=0)
        fused_scale: (2*inter,) float — cat([gate_scale, up_scale])
        inter_size: intermediate_size (split point)

    Returns:
        (gate, up) — each (*, inter_size) float
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    if not (_SCALED_MM_AVAILABLE and x.is_cuda and fp8_available()):
        # Fallback: dequant + F.linear
        w_float = fused_fp8.float() * fused_scale.unsqueeze(-1)
        result = F.linear(x_2d.float(), w_float)
        gate, up = result.split(inter_size, dim=-1)
        return (
            gate.reshape(*orig_shape[:-1], inter_size),
            up.reshape(*orig_shape[:-1], inter_size),
        )

    x_fp8, x_scale = quantize_activations_fp8(x_2d)

    w_scale_2d = fused_scale.unsqueeze(0).float()
    x_scale_2d = x_scale.float()

    wt = fused_fp8.t().contiguous()
    x_fp8_c = x_fp8.contiguous()

    result = torch._scaled_mm(
        x_fp8_c,
        wt,
        scale_a=x_scale_2d,
        scale_b=w_scale_2d,
        out_dtype=x.dtype if x.is_floating_point() else torch.float16,
    )

    gate, up = result.split(inter_size, dim=-1)
    return (
        gate.reshape(*orig_shape[:-1], inter_size),
        up.reshape(*orig_shape[:-1], inter_size),
    )


# =========================================================================
# FP8 expert quantization (batch all experts)
# =========================================================================

def quantize_experts_fp8(
    gate_up: torch.Tensor,       # (num_experts, hidden, 2*inter)
    down: torch.Tensor,          # (num_experts, inter, hidden)
) -> dict:
    """
    Quantize all expert weights to FP8 E4M3.

    Returns dict with FP8 tensors and per-channel scales.
    """
    num_experts = gate_up.shape[0]
    gu_q, gu_s = [], []
    dn_q, dn_s = [], []

    for e in range(num_experts):
        q, s = quantize_fp8(gate_up[e])
        gu_q.append(q)
        gu_s.append(s)
        q, s = quantize_fp8(down[e])
        dn_q.append(q)
        dn_s.append(s)

    return {
        "method": "fp8",
        "gate_up_fp8": torch.stack(gu_q),
        "gate_up_scale": torch.stack(gu_s),
        "down_fp8": torch.stack(dn_q),
        "down_scale": torch.stack(dn_s),
    }


# =========================================================================
# Dense model FP8 quantization
# =========================================================================

def quantize_dense_fp8(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a single dense weight matrix to FP8 (for DenseMLP / attention).
    Returns (weight_fp8, scale).
    """
    return quantize_fp8(weight, per_channel=True)
