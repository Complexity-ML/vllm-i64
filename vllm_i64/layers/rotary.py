"""
vllm-i64 :: Rotary Positional Embedding

Integer position IDs → float sin/cos rotation.
Model-agnostic.

INL - 2025
"""

import torch
import torch.nn as nn
from typing import Tuple


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


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embedding to x.
    x:   (batch, heads, head_dim)
    cos: (batch, head_dim)  — from RotaryEmbedding.forward()
    sin: (batch, head_dim)
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # cos/sin are (batch, head_dim) = (batch, 2*d), take first half
    cos = cos[..., :d].unsqueeze(1)  # (batch, 1, d)
    sin = sin[..., :d].unsqueeze(1)  # (batch, 1, d)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
