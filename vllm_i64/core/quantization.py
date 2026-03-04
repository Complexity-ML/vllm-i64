"""
vllm-i64 :: Quantization

INT8/INT4 quantization for expert weights.
Routing stays i64 (no quantization needed — it's integer already).
Only expert MLP weights get quantized.

Strategies:
  - INT8: per-channel symmetric quantization (native _int_mm when available)
  - INT4: per-group asymmetric quantization (group_size=128)

Native INT8 matmul:
  torch._int_mm (PyTorch 2.2+, SM80+ GPU) does true INT8×INT8→INT32.
  Activations dynamically quantized per-token, weights statically per-channel.
  Fallback to dequant+F.linear on CPU or older GPUs.

INL - 2025
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

_logger = logging.getLogger("vllm_i64.quantization")

# Native INT8 matmul support detection
_INT_MM_AVAILABLE = hasattr(torch, '_int_mm')


@dataclass
class QuantConfig:
    """Quantization configuration."""
    method: str = "none"         # "none", "int8", "int4"
    group_size: int = 128        # for INT4


def quantize_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel INT8 symmetric quantization.

    weight: (out, in) float → (out, in) int8 + (out,) float scale

    Dequantize: weight_fp = weight_int8 * scale
    """
    abs_max = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / 127.0
    quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(-1)


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """INT8 → float."""
    return quantized.float() * scale.unsqueeze(-1)


# =========================================================================
# Native INT8 matmul — torch._int_mm
# =========================================================================

def quantize_activations_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamic per-token INT8 symmetric quantization of activations.

    x: (tokens, features) any float → (tokens, features) int8 + (tokens,) float32 scale

    Always computes in float32 — bf16 has only 8-bit mantissa,
    not enough precision for scale computation before INT8 quantization.
    """
    x_f32 = x.float()
    abs_max = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / 127.0
    x_int8 = (x_f32 / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale.squeeze(-1)


def int8_linear_native(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    INT8 linear: y = x @ W^T, with native INT8 matmul when available.

    Uses torch._int_mm (PyTorch 2.2+, SM80+ GPU) for true integer compute:
        1. Dynamic-quantize activations x → x_int8, x_scale  (per-token)
        2. _int_mm(x_int8, W_int8^T) → result_int32          (native INT8)
        3. Rescale: result_float = result_int32 * x_scale * w_scale

    Falls back to dequant + F.linear on CPU or unsupported GPUs.

    Args:
        x: (*, in_features) float activations
        weight_int8: (out_features, in_features) int8 weights
        weight_scale: (out_features,) float per-channel scale
        bias: optional (out_features,) float

    Returns:
        y: (*, out_features) float
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    if not (_INT_MM_AVAILABLE and x.is_cuda):
        # Fallback: dequant weight → float32 matmul (handles bf16/fp16 input)
        w_float = dequantize_int8(weight_int8, weight_scale)
        out = F.linear(x_2d.float(), w_float, bias)
        return out.reshape(*orig_shape[:-1], weight_int8.shape[0])

    # Dynamic quantize activations
    x_int8, x_scale = quantize_activations_int8(x_2d)

    # Native INT8 matmul: (M, K) @ (K, N) → (M, N) int32
    # weight_int8 is (out, in), need (in, out) for second operand
    wt = weight_int8.t().contiguous()
    result_i32 = torch._int_mm(x_int8, wt)

    # Rescale: y[i,j] = result_i32[i,j] * x_scale[i] * w_scale[j]
    out = result_i32.float() * (x_scale.unsqueeze(1) * weight_scale.unsqueeze(0))

    if bias is not None:
        out = out + bias

    return out.reshape(*orig_shape[:-1], weight_int8.shape[0])


def int8_fused_gate_up_native(
    x: torch.Tensor,
    fused_int8: torch.Tensor,
    fused_scale: torch.Tensor,
    inter_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused gate+up: single activation quantization + single INT8 matmul.

    gate_int8 and up_int8 pre-concatenated along dim 0 into fused_int8.
    One _int_mm does both projections. Output split at inter_size.

    Saves vs two separate int8_linear_native calls:
      - 1 fewer activation quantization (expensive: abs, div, round, clamp)
      - 1 fewer _int_mm kernel launch
      - Larger matrix → better tensor core utilization

    Args:
        x: (*, hidden) float activations
        fused_int8: (2*inter, hidden) int8 — cat([gate, up], dim=0)
        fused_scale: (2*inter,) float — cat([gate_scale, up_scale])
        inter_size: intermediate_size (split point)

    Returns:
        (gate, up) — each (*, inter_size) float
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    if not (_INT_MM_AVAILABLE and x.is_cuda):
        w_float = dequantize_int8(fused_int8, fused_scale)
        result = F.linear(x_2d.float(), w_float)
        gate, up = result.split(inter_size, dim=-1)
        return (
            gate.reshape(*orig_shape[:-1], inter_size),
            up.reshape(*orig_shape[:-1], inter_size),
        )

    x_int8, x_scale = quantize_activations_int8(x_2d)
    wt = fused_int8.t().contiguous()
    result_i32 = torch._int_mm(x_int8, wt)
    result = result_i32.float() * (x_scale.unsqueeze(1) * fused_scale.unsqueeze(0))

    gate, up = result.split(inter_size, dim=-1)
    return (
        gate.reshape(*orig_shape[:-1], inter_size),
        up.reshape(*orig_shape[:-1], inter_size),
    )


def int8_linear_available() -> bool:
    """Check if native INT8 matmul is available on current hardware."""
    if not _INT_MM_AVAILABLE:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        # Probe with tiny matmul
        a = torch.ones(1, 8, dtype=torch.int8, device='cuda')
        b = torch.ones(8, 1, dtype=torch.int8, device='cuda')
        torch._int_mm(a, b)
        return True
    except (RuntimeError, Exception):
        return False


def quantize_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-group INT4 asymmetric quantization.

    weight: (out, in) float → (out, in) uint8 (packed 2 per byte) + scales + zeros

    Returns: (packed, scales, zeros)
    """
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"INT4 quantize: in_features ({in_features}) not divisible by group_size ({group_size})"
        )

    num_groups = in_features // group_size
    weight_grouped = weight.reshape(out_features, num_groups, group_size)

    # Per-group min/max
    w_min = weight_grouped.amin(dim=-1)
    w_max = weight_grouped.amax(dim=-1)

    # Scale and zero point
    scale = (w_max - w_min) / 15.0
    scale = scale.clamp(min=1e-8)
    zero = (-w_min / scale).round().clamp(0, 15)

    # Quantize
    quantized = ((weight_grouped - w_min.unsqueeze(-1)) / scale.unsqueeze(-1))
    quantized = quantized.round().clamp(0, 15).to(torch.uint8)

    # Pack 2 values per byte
    quantized_flat = quantized.reshape(out_features, -1)
    packed = torch.zeros(
        out_features, in_features // 2, dtype=torch.uint8, device=weight.device
    )
    packed = (quantized_flat[:, 0::2] << 4) | quantized_flat[:, 1::2]

    return packed, scale, zero


def dequantize_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """INT4 packed → float. Vectorized unpack + dequant."""
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 2

    if in_features % group_size != 0:
        raise ValueError(
            f"INT4 dequant: in_features ({in_features}) not divisible by group_size ({group_size})"
        )

    # Vectorized unpack: extract high/low nibbles
    high = (packed >> 4) & 0xF
    low = packed & 0xF
    unpacked = torch.stack([high, low], dim=-1).reshape(out_features, in_features)

    # Dequantize per group
    num_groups = in_features // group_size
    unpacked_grouped = unpacked.reshape(out_features, num_groups, group_size).float()
    return (unpacked_grouped - zero.unsqueeze(-1)) * scale.unsqueeze(-1)


def int4_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int = 128,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused INT4 dequant + GEMM: y = x @ dequant(packed).T + bias

    Instead of materializing the full dequantized weight, this processes
    by groups to reduce peak memory. For each group, unpack + dequant + partial GEMM.

    Args:
        x: (batch, in_features) or (in_features,)
        packed: (out_features, in_features // 2) uint8
        scale: (out_features, num_groups) float
        zero: (out_features, num_groups) float
        group_size: quantization group size
        bias: optional (out_features,) float

    Returns:
        y: (batch, out_features) float
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 2
    num_groups = in_features // group_size

    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)

    # Always compute in float32 (bf16/fp16 mixed with dequant float would crash)
    x = x.float()
    batch = x.shape[0]
    y = torch.zeros(batch, out_features, dtype=torch.float32, device=x.device)

    # Process by groups: unpack + dequant + accumulate partial GEMM
    for g in range(num_groups):
        col_start = g * group_size
        col_end = col_start + group_size
        pack_start = col_start // 2
        pack_end = col_end // 2

        # Unpack this group
        group_packed = packed[:, pack_start:pack_end]
        high = ((group_packed >> 4) & 0xF).float()
        low = (group_packed & 0xF).float()
        group_unpacked = torch.stack([high, low], dim=-1).reshape(out_features, group_size)

        # Dequant: (unpacked - zero) * scale
        group_weight = (group_unpacked - zero[:, g:g+1]) * scale[:, g:g+1]

        # Partial GEMM: y += x[:, group_cols] @ group_weight.T
        y += x[:, col_start:col_end] @ group_weight.T

    if bias is not None:
        y += bias

    return y.squeeze(0) if squeeze else y


def quantize_experts(
    gate_up: torch.Tensor,       # (num_experts, hidden, 2*inter)
    down: torch.Tensor,          # (num_experts, inter, hidden)
    config: QuantConfig,
) -> dict:
    """
    Quantize all expert weights.

    Returns dict with quantized tensors and metadata.
    """
    if config.method == "none":
        return {"gate_up": gate_up, "down": down, "method": "none"}

    results = {"method": config.method}
    num_experts = gate_up.shape[0]

    if config.method == "int8":
        gu_q, gu_s = [], []
        dn_q, dn_s = [], []
        for e in range(num_experts):
            q, s = quantize_int8(gate_up[e])
            gu_q.append(q)
            gu_s.append(s)
            q, s = quantize_int8(down[e])
            dn_q.append(q)
            dn_s.append(s)

        results["gate_up_int8"] = torch.stack(gu_q)
        results["gate_up_scale"] = torch.stack(gu_s)
        results["down_int8"] = torch.stack(dn_q)
        results["down_scale"] = torch.stack(dn_s)

    elif config.method == "int4":
        gu_p, gu_s, gu_z = [], [], []
        dn_p, dn_s, dn_z = [], [], []
        for e in range(num_experts):
            p, s, z = quantize_int4(gate_up[e], config.group_size)
            gu_p.append(p)
            gu_s.append(s)
            gu_z.append(z)
            p, s, z = quantize_int4(down[e], config.group_size)
            dn_p.append(p)
            dn_s.append(s)
            dn_z.append(z)

        results["gate_up_int4"] = torch.stack(gu_p)
        results["gate_up_scale"] = torch.stack(gu_s)
        results["gate_up_zero"] = torch.stack(gu_z)
        results["down_int4"] = torch.stack(dn_p)
        results["down_scale"] = torch.stack(dn_s)
        results["down_zero"] = torch.stack(dn_z)

    return results
