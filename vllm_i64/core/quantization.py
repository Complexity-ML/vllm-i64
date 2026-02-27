"""
vllm-i64 :: Quantization

INT8/INT4 quantization for expert weights.
Routing stays i64 (no quantization needed — it's integer already).
Only expert MLP weights get quantized.

Strategies:
  - INT8: per-channel symmetric quantization
  - INT4: per-group asymmetric quantization (group_size=128)

INL - 2025
"""

import torch
from typing import Tuple
from dataclasses import dataclass


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
    assert in_features % group_size == 0

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
    """INT4 packed → float."""
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 2

    # Unpack
    high = (packed >> 4) & 0xF
    low = packed & 0xF
    unpacked = torch.stack([high, low], dim=-1).reshape(out_features, in_features)

    # Dequantize per group
    num_groups = in_features // group_size
    unpacked_grouped = unpacked.reshape(out_features, num_groups, group_size).float()
    return (unpacked_grouped - zero.unsqueeze(-1)) * scale.unsqueeze(-1)


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
