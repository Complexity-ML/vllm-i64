"""
vllm-i64 :: Quantization

INT8/INT4 quantization for expert weights.
Routing stays i64 (no quantization needed — it's integer already).
Only expert MLP weights get quantized.

Strategies:
  - INT8: per-channel symmetric quantization (native _int_mm when available)
  - INT4: per-group asymmetric quantization (group_size=128)

Native INT8 matmul:
  torch._int_mm (PyTorch 2.2+ CPU/GPU) does true INT8×INT8→INT32.
  Activations dynamically quantized per-token, weights statically per-channel.
  Works on CPU (VNNI/AMX) and GPU (SM80+ tensor cores).
  Fallback to dequant+F.linear on unsupported platforms.

INL - 2025
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

_logger = logging.getLogger("vllm_i64.quantization")

# Native INT8 matmul support detection (works on CPU + GPU since PyTorch 2.4)
_INT_MM_AVAILABLE = hasattr(torch, '_int_mm')

# Probe CPU _int_mm once at import time
_INT_MM_CPU_OK = False
if _INT_MM_AVAILABLE:
    try:
        _a = torch.ones(1, 8, dtype=torch.int8)
        _b = torch.ones(8, 1, dtype=torch.int8)
        torch._int_mm(_a, _b)
        _INT_MM_CPU_OK = True
    except (RuntimeError, Exception):
        pass

# Probe GPU _int_mm once (lazy, on first use)
_INT_MM_GPU_OK: Optional[bool] = None

# Log INT8 backend selection once
_INT8_BACKEND_LOGGED = False

def _log_int8_backend(name: str):
    global _INT8_BACKEND_LOGGED
    if not _INT8_BACKEND_LOGGED:
        _logger.info("INT8 matmul backend: %s", name)
        _INT8_BACKEND_LOGGED = True

def _probe_int_mm_gpu() -> bool:
    """Test if torch._int_mm works on the current CUDA device."""
    global _INT_MM_GPU_OK
    if _INT_MM_GPU_OK is not None:
        return _INT_MM_GPU_OK
    try:
        # Use padded size (17) to match real usage path
        a = torch.ones(17, 64, dtype=torch.int8, device="cuda")
        b = torch.ones(64, 64, dtype=torch.int8, device="cuda")
        torch._int_mm(a, b)
        _INT_MM_GPU_OK = True
        _logger.info("GPU INT8 _int_mm: supported")
    except RuntimeError:
        _INT_MM_GPU_OK = False
        _logger.info("GPU INT8 _int_mm: not supported, using dequant fallback")
    return _INT_MM_GPU_OK


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
    x_preq: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
    out_features = weight_int8.shape[0]

    # Check if native _int_mm is usable on this device
    use_int_mm = _INT_MM_AVAILABLE and (
        (x.is_cuda and _probe_int_mm_gpu()) or (not x.is_cuda and _INT_MM_CPU_OK)
    )

    if use_int_mm:
        # Native INT8 matmul path — CPU (VNNI/AMX) or GPU (tensor cores)
        _log_int8_backend("torch._int_mm (native)")
        if x_preq is not None:
            x_int8, x_scale = x_preq[0], x_preq[1]
        else:
            x_int8, x_scale = quantize_activations_int8(x_2d)
        wt = weight_int8.t().contiguous()
        # torch._int_mm requires size(0) > 16 on CUDA — pad if needed
        m = x_int8.shape[0]
        if x_int8.is_cuda and m <= 16:
            pad = 17 - m
            x_int8 = torch.nn.functional.pad(x_int8, (0, 0, 0, pad))
            x_scale = torch.nn.functional.pad(x_scale, (0, pad))
            result_i32 = torch._int_mm(x_int8, wt)
            result_i32 = result_i32[:m]
            x_scale = x_scale[:m]
        else:
            result_i32 = torch._int_mm(x_int8, wt)
        out = result_i32.float() * (x_scale.unsqueeze(1) * weight_scale.unsqueeze(0))
        if bias is not None:
            out = out + bias
        return out.reshape(*orig_shape[:-1], out_features)

    # GPU-only accelerated fallbacks
    if x.is_cuda:
        # Priority 1: Triton native INT8x8→INT32 GEMM (no cuBLAS dependency)
        try:
            from vllm_i64.kernels.triton.I64_int8_gemm import triton_int8_gemm
            if x_preq is not None:
                x_int8, x_scale = x_preq[0], x_preq[1]
            else:
                x_int8, x_scale = quantize_activations_int8(x_2d)
            wt = weight_int8.t().contiguous()
            out = triton_int8_gemm(x_int8, x_scale, wt, weight_scale, bias)
            if out is not None:
                _log_int8_backend("Triton INT8x8→INT32 (native)")
                return out.reshape(*orig_shape[:-1], out_features)
        except ImportError:
            pass
        except RuntimeError as e:
            _logger.debug("Triton INT8 GEMM failed: %s", e)
        # Priority 2: CUDA I64_gemm_dequant_int8
        try:
            from vllm_i64.kernels.cuda import get_i64_cuda_ops
            cuda_ops = get_i64_cuda_ops()
            if cuda_ops is not None:
                out = cuda_ops.gemm_dequant_int8(x_2d.float(), weight_int8, weight_scale)
                if bias is not None:
                    out = out + bias
                _log_int8_backend("CUDA gemm_dequant_int8")
                return out.reshape(*orig_shape[:-1], out_features)
        except ImportError:
            pass
        except RuntimeError as e:
            _logger.debug("CUDA gemm_dequant_int8 failed: %s", e)
        # Priority 3: Triton fused dequant+GEMM
        try:
            from vllm_i64.kernels.triton.I64_fused_dequant_gemm import triton_dequant_gemm_int8
            out = triton_dequant_gemm_int8(x_2d, weight_int8, weight_scale, bias)
            if out is not None:
                _log_int8_backend("Triton fused dequant+GEMM")
                return out.reshape(*orig_shape[:-1], out_features)
        except ImportError:
            pass

    # Final fallback: dequant weight → float32 matmul
    _log_int8_backend("dequant + F.linear (float32 fallback)")
    w_float = dequantize_int8(weight_int8, weight_scale)
    out = F.linear(x_2d.float(), w_float, bias)
    return out.reshape(*orig_shape[:-1], out_features)


def int8_fused_gate_up_native(
    x: torch.Tensor,
    fused_int8: torch.Tensor,
    fused_scale: torch.Tensor,
    inter_size: int,
    x_preq: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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
    use_int_mm = _INT_MM_AVAILABLE and (
        (x.is_cuda and _probe_int_mm_gpu()) or (not x.is_cuda and _INT_MM_CPU_OK)
    )

    if use_int_mm:
        if x_preq is not None:
            x_int8, x_scale = x_preq[0], x_preq[1]
        else:
            x_int8, x_scale = quantize_activations_int8(x_2d)
        wt = fused_int8.t().contiguous()
        # torch._int_mm requires size(0) > 16 on CUDA — pad if needed
        m = x_int8.shape[0]
        if x_int8.is_cuda and m <= 16:
            pad = 17 - m
            x_int8 = torch.nn.functional.pad(x_int8, (0, 0, 0, pad))
            x_scale = torch.nn.functional.pad(x_scale, (0, pad))
            result_i32 = torch._int_mm(x_int8, wt)
            result_i32 = result_i32[:m]
            x_scale = x_scale[:m]
        else:
            result_i32 = torch._int_mm(x_int8, wt)
        result = result_i32.float() * (x_scale.unsqueeze(1) * fused_scale.unsqueeze(0))
    else:
        w_float = dequantize_int8(fused_int8, fused_scale)
        result = F.linear(x_2d.float(), w_float)

    gate, up = result.split(inter_size, dim=-1)
    return (
        gate.reshape(*orig_shape[:-1], inter_size),
        up.reshape(*orig_shape[:-1], inter_size),
    )


def int8_linear_available(device: str = "cpu") -> bool:
    """Check if native INT8 matmul is available on the given device."""
    if not _INT_MM_AVAILABLE:
        return False
    if device == "cpu":
        return _INT_MM_CPU_OK
    # GPU probe
    if not torch.cuda.is_available():
        return False
    try:
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
    Vectorized INT4 dequant + GEMM: y = x @ dequant(packed).T + bias

    Fully vectorized — no Python loops. Unpacks all groups at once,
    dequantizes in one broadcast op, then single F.linear GEMM.

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

    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)

    # Vectorized unpack: extract all nibbles at once
    high = ((packed >> 4) & 0xF).float()
    low = (packed & 0xF).float()
    unpacked = torch.stack([high, low], dim=-1).reshape(out_features, in_features)

    # Vectorized dequant: reshape to (out, groups, group_size), broadcast scale/zero
    num_groups = in_features // group_size
    w_grouped = unpacked.reshape(out_features, num_groups, group_size)
    w_float = ((w_grouped - zero.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape(out_features, in_features)

    # Single GEMM — one big matmul instead of N small ones
    out = F.linear(x.float(), w_float, bias)

    return out.squeeze(0) if squeeze else out


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
