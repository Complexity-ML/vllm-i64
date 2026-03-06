"""
vllm-i64 :: Integer Activation Functions

LUT-based integer approximations for activation functions.
Zero compute — just table lookups. INT8 in → INT8 out.

All LUTs use Q7 scale (x128) covering [-8.0, 8.0] range.
Canonical location for all integer activations:
  - silu_integer, silu_multiply_integer
  - sigmoid_integer
  - softplus_integer

INL - 2025
"""

import torch
import torch.nn.functional as F

_Q7 = 128  # Fixed-point scale: 7 fractional bits


# =========================================================================
# SiLU LUT — silu(x) = x * sigmoid(x)
# =========================================================================

_SILU_LUT_MIN = -1024    # -8.0 at Q7
_SILU_LUT_MAX = 1024     # +8.0 at Q7

def _build_silu_lut() -> torch.Tensor:
    """Build SiLU LUT: integer index in [-1024, 1024] -> silu(index/128) * 128."""
    indices = torch.arange(_SILU_LUT_MIN, _SILU_LUT_MAX + 1, dtype=torch.float32)
    x = indices / _Q7
    return (F.silu(x) * _Q7).round().to(torch.int32)

_SILU_LUT = _build_silu_lut()


def silu_integer(x_q7: torch.Tensor) -> torch.Tensor:
    """
    Fixed-point SiLU via LUT. Input/output in Q7 scale.

    Values outside [-8, 8]: silu(x) ~ x for x > 8, ~ 0 for x < -8.
    """
    lut = _SILU_LUT.to(x_q7.device)
    clamped = x_q7.clamp(_SILU_LUT_MIN, _SILU_LUT_MAX)
    indices = (clamped - _SILU_LUT_MIN).long()
    result = lut[indices]
    result = torch.where(x_q7 > _SILU_LUT_MAX, x_q7, result)
    result = torch.where(x_q7 < _SILU_LUT_MIN, torch.zeros_like(result), result)
    return result


def silu_multiply_integer(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Integer SiLU + multiply: silu(gate) * up, all in fixed-point INT32.

    1. Quantize gate/up to Q7 (x128, round to INT32)
    2. SiLU via 2049-entry LUT on gate
    3. Multiply in INT32 (Q7 x Q7 -> Q14)
    4. Dequantize back to float (/ 128^2)
    """
    gate_q7 = (gate.float() * _Q7).round().to(torch.int32)
    silu_q7 = silu_integer(gate_q7)
    up_q7 = (up.float() * _Q7).round().to(torch.int32)
    inter_q14 = silu_q7 * up_q7
    return inter_q14.float() / (_Q7 * _Q7)


# =========================================================================
# Sigmoid LUT — sigmoid(x) = 1/(1+exp(-x))
# =========================================================================
# Output range [0, 1] -> stored as Q7 [0, 128]

_SIG_LUT_MIN = -1024   # -8.0 at Q7
_SIG_LUT_MAX = 1024    # +8.0 at Q7
_SIG_LUT_SIZE = _SIG_LUT_MAX - _SIG_LUT_MIN + 1  # 2049

def _build_sigmoid_lut() -> torch.Tensor:
    """sigmoid(index/128) * 128, for index in [-1024, 1024]."""
    indices = torch.arange(_SIG_LUT_MIN, _SIG_LUT_MAX + 1, dtype=torch.float32)
    x = indices / _Q7
    return (torch.sigmoid(x) * _Q7).round().to(torch.int32)

_SIGMOID_LUT = _build_sigmoid_lut()


def sigmoid_integer(x_q7: torch.Tensor) -> torch.Tensor:
    """
    Fixed-point sigmoid via LUT. Input Q7, output Q7.

    Values outside [-8, 8]: sigmoid(x) ~ 1 for x > 8, ~ 0 for x < -8.
    """
    lut = _SIGMOID_LUT.to(x_q7.device)
    clamped = x_q7.clamp(_SIG_LUT_MIN, _SIG_LUT_MAX)
    indices = (clamped - _SIG_LUT_MIN).long()
    result = lut[indices]
    result = torch.where(x_q7 > _SIG_LUT_MAX, torch.full_like(result, _Q7), result)
    result = torch.where(x_q7 < _SIG_LUT_MIN, torch.zeros_like(result), result)
    return result


# =========================================================================
# Softplus LUT — softplus(x) = ln(1+exp(x))
# =========================================================================
# Output range [0, ~8] -> stored as Q7

_SP_LUT_MIN = -1024
_SP_LUT_MAX = 1024
_SP_LUT_SIZE = _SP_LUT_MAX - _SP_LUT_MIN + 1

def _build_softplus_lut() -> torch.Tensor:
    """softplus(index/128) * 128, for index in [-1024, 1024]."""
    indices = torch.arange(_SP_LUT_MIN, _SP_LUT_MAX + 1, dtype=torch.float32)
    x = indices / _Q7
    return (F.softplus(x) * _Q7).round().to(torch.int32)

_SOFTPLUS_LUT = _build_softplus_lut()


def softplus_integer(x_q7: torch.Tensor) -> torch.Tensor:
    """
    Fixed-point softplus via LUT. Input Q7, output Q7.

    Values outside range: softplus(x) ~ x for x >> 0, ~ 0 for x << 0.
    """
    lut = _SOFTPLUS_LUT.to(x_q7.device)
    clamped = x_q7.clamp(_SP_LUT_MIN, _SP_LUT_MAX)
    indices = (clamped - _SP_LUT_MIN).long()
    result = lut[indices]
    # x >> 8: softplus(x) ~ x
    result = torch.where(x_q7 > _SP_LUT_MAX, x_q7, result)
    result = torch.where(x_q7 < _SP_LUT_MIN, torch.zeros_like(result), result)
    return result
