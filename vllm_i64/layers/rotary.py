"""
vllm-i64 :: Rotary Positional Embedding

Float path: standard sin/cos computation + float rotation.
Integer path: pre-computed Q14 cos/sin LUT tables + INT32 arithmetic.

Integer RoPE:
  - Pre-compute cos/sin for all positions as Q14 INT16 ([-1,1] → [-16384,16384])
  - apply_rotary_integer: Q7 input × Q14 cos/sin → Q21, dequant to float
  - Eliminates float trig at inference time (tables computed once)

INL - 2025
"""

import torch
import torch.nn as nn
from typing import Tuple

# Fixed-point scales for integer RoPE
_Q_ROPE = 16384     # Q14 for cos/sin ([-1, 1] fits INT16 with good resolution)
_Q_ROPE_IN = 128    # Q7 for input quantization


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """positions: (batch,) integer → cos, sin: (batch, dim)"""
        t = positions.float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def forward_integer(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integer RoPE: returns Q14 INT16 cos/sin from pre-computed tables.
        Tables built lazily on first call or explicitly via build_integer_tables().
        """
        max_pos = positions.max().item() + 1 if positions.numel() > 0 else 1

        if not hasattr(self, 'cos_table') or max_pos > self.cos_table.shape[0]:
            table_size = max(max_pos, 2048)
            self.build_integer_tables(table_size, positions.device)

        idx = positions.long()
        return self.cos_table[idx], self.sin_table[idx]

    def build_integer_tables(self, max_seq_len: int, device: torch.device) -> None:
        """Pre-compute cos/sin LUT as Q14 INT16."""
        t = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_table = (emb.cos() * _Q_ROPE).round().clamp(-32768, 32767).to(torch.int16)
        self.sin_table = (emb.sin() * _Q_ROPE).round().clamp(-32768, 32767).to(torch.int16)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embedding to x (float path).
    x:   (batch, heads, head_dim)
    cos: (batch, head_dim)  — from RotaryEmbedding.forward()
    sin: (batch, head_dim)
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # cos/sin are (batch, head_dim) = (batch, 2*d), take first half
    cos = cos[..., :d].unsqueeze(1).to(x.dtype)  # (batch, 1, d)
    sin = sin[..., :d].unsqueeze(1).to(x.dtype)  # (batch, 1, d)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def apply_rotary_integer(
    x: torch.Tensor,
    cos_q14: torch.Tensor,
    sin_q14: torch.Tensor,
) -> torch.Tensor:
    """
    Integer rotary embedding: Q7 input × Q14 cos/sin → Q21, dequant to float.

    x:       (batch, heads, head_dim) float
    cos_q14: (batch, head_dim) INT16 — Q14 scaled cos
    sin_q14: (batch, head_dim) INT16 — Q14 scaled sin

    Returns: (batch, heads, head_dim) float, same dtype as input
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]

    # Quantize input to Q7 — float32 for precision
    x1_q = (x1.float() * _Q_ROPE_IN).round().to(torch.int32)
    x2_q = (x2.float() * _Q_ROPE_IN).round().to(torch.int32)

    # cos/sin: (batch, head_dim) → take first half, add head dim
    cos = cos_q14[..., :d].unsqueeze(1).to(torch.int32)  # (batch, 1, d)
    sin = sin_q14[..., :d].unsqueeze(1).to(torch.int32)

    # Rotation in INT32: Q7 × Q14 → Q21
    r1 = x1_q * cos - x2_q * sin
    r2 = x2_q * cos + x1_q * sin

    # Dequant: ÷(128 × 16384)
    scale = _Q_ROPE_IN * _Q_ROPE
    return torch.cat([r1.float() / scale, r2.float() / scale], dim=-1).type_as(x)
