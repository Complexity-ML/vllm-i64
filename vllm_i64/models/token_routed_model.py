"""
vllm-i64 :: Token-Routed Model

Full model implementation for i64 inference.
Routing, attention, KV cache — all integer.
Only expert MLP (SwiGLU) is float.

INL - 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

from vllm_i64.kernels.i64_ops import i64_route_tokens, i64_scatter, i64_gather


@dataclass
class TokenRoutedConfig:
    """Model configuration — all sizes are integers."""
    vocab_size: int = 32000
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_experts: int = 4
    max_seq_len: int = 2048
    head_dim: int = 64           # hidden_dim // num_heads
    expert_inter: int = 192      # hidden_dim // num_experts (per-expert FFN)


# =========================================================================
# RMS Norm (float, but applied per-token)
# =========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# =========================================================================
# Rotary Positional Embedding (integer positions → float sin/cos)
# =========================================================================

class RotaryEmbedding(nn.Module):
    """RoPE with integer position IDs."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        positions: (batch,) integer tensor — i32 or i64
        Returns cos, sin: (batch, dim)
        """
        # Integer positions → float for trig only
        t = positions.float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos.unsqueeze(1)  # (batch, 1, dim)
    sin = sin.unsqueeze(1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# =========================================================================
# Attention (standard multi-head, integer position indexing)
# =========================================================================

class I64Attention(nn.Module):
    """
    Multi-head attention.

    Integer parts:
      - Position indices (i32)
      - KV cache slot indexing (i32)
      - Head indexing

    Float parts:
      - Q/K/V projections and attention scores (unavoidable)
    """

    def __init__(self, config: TokenRoutedConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len)

    def forward(
        self,
        hidden: torch.Tensor,          # (batch, hidden_dim) float
        positions: torch.Tensor,        # (batch,) i32 — integer!
        kv_cache: Optional[Tuple] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = hidden.shape[0]

        q = self.q_proj(hidden).view(bsz, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(bsz, self.num_heads, self.head_dim)
        v = self.v_proj(hidden).view(bsz, self.num_heads, self.head_dim)

        # RoPE: integer positions → float rotation
        cos, sin = self.rope(positions)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # KV cache update (integer slot indexing)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Integer index for cache slot
            cache_slots = positions.to(torch.int32)
            k_cache[:, cache_slots] = k.transpose(0, 1)
            v_cache[:, cache_slots] = v.transpose(0, 1)
            k = k_cache.transpose(0, 1).reshape(-1, self.num_heads, self.head_dim)[:bsz]
            v = v_cache.transpose(0, 1).reshape(-1, self.num_heads, self.head_dim)[:bsz]

        # Attention scores (float)
        scale = 1.0 / math.sqrt(self.head_dim)
        q = q.transpose(0, 1)  # (heads, batch, dim)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        out = out.transpose(0, 1).reshape(bsz, self.hidden_dim)
        return self.o_proj(out)


# =========================================================================
# Token-Routed Expert MLP (the ONLY dense float computation)
# =========================================================================

class TokenRoutedExpertMLP(nn.Module):
    """
    SwiGLU MLP dispatched per-expert.

    Integer parts:
      - expert_id selection (i64 bit mask)
      - scatter/gather (i32 indices)

    Float parts:
      - gate/up projection, SiLU, down projection (FP16/FP32)
    """

    def __init__(self, config: TokenRoutedConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_dim
        self.expert_inter = config.expert_inter

        # Expert weights: (num_experts, hidden_dim, 2 * expert_inter) for gate+up
        self.gate_up = nn.Parameter(
            torch.randn(config.num_experts, config.hidden_dim, 2 * config.expert_inter) * 0.02
        )
        # Down projection: (num_experts, expert_inter, hidden_dim)
        self.down = nn.Parameter(
            torch.randn(config.num_experts, config.expert_inter, config.hidden_dim) * 0.02
        )

    def forward(
        self,
        hidden: torch.Tensor,      # (batch, hidden_dim) float
        expert_ids: torch.Tensor,   # (batch,) i32 — integer routing result
    ) -> torch.Tensor:
        # ---- Integer zone: scatter ----
        scattered, indices, offsets, counts = i64_scatter(hidden, expert_ids, self.num_experts)

        # ---- Float zone: expert compute (SwiGLU) ----
        output_parts = []
        cursor = 0
        for e in range(self.num_experts):
            n = counts[e].item()
            if n == 0:
                continue

            x_e = scattered[cursor:cursor + n]                   # (n, hidden)
            gate_up = x_e @ self.gate_up[e]                      # (n, 2*inter)
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up                        # SwiGLU
            out_e = activated @ self.down[e]                     # (n, hidden)

            output_parts.append(out_e)
            cursor += n

        expert_output = torch.cat(output_parts, dim=0) if output_parts else hidden.new_zeros(0, self.hidden_dim)

        # ---- Integer zone: gather (restore original token order) ----
        output = i64_gather(expert_output, indices)

        return output


# =========================================================================
# Decoder Layer
# =========================================================================

class TokenRoutedDecoderLayer(nn.Module):
    """
    Single decoder layer:
      1. RMSNorm → Attention (float, integer positions)
      2. RMSNorm → Token-Routed Expert MLP (integer route, float compute)
    """

    def __init__(self, config: TokenRoutedConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.attn = I64Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.ffn = TokenRoutedExpertMLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,    # i32
        expert_ids: torch.Tensor,   # i32 — pre-computed by scheduler
        kv_cache=None,
        mask=None,
    ) -> torch.Tensor:
        # Attention with residual
        h = self.attn_norm(hidden)
        h = self.attn(h, positions, kv_cache, mask)
        hidden = hidden + h

        # Expert MLP with residual
        h = self.ffn_norm(hidden)
        h = self.ffn(h, expert_ids)
        hidden = hidden + h

        return hidden


# =========================================================================
# Full Token-Routed Model
# =========================================================================

class TokenRoutedModel(nn.Module):
    """
    Full token-routed transformer.

    Integer pipeline:
      token_ids (i64) → embedding lookup (i32 index)
      → per-layer: attention (i32 positions) + expert MLP (i32 expert_ids)
      → lm_head → logits → argmax (i64 output token)

    The ONLY float: attention projections, expert SwiGLU, embedding vectors.
    All control flow, indexing, routing = integer.
    """

    def __init__(self, config: TokenRoutedConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            TokenRoutedDecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,     # (batch,) i64
        positions: torch.Tensor,     # (batch,) i32
        expert_ids: torch.Tensor,    # (batch,) i32 — pre-routed by scheduler
        kv_caches=None,
        mask=None,
    ) -> torch.Tensor:
        """
        Returns logits: (batch, vocab_size) — float.
        Caller does argmax → i64 token ID.
        """
        # Embedding lookup: integer index → float vector
        hidden = self.embed(token_ids.long())

        # Decoder layers
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches else None
            hidden = layer(hidden, positions, expert_ids, kv, mask)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        return logits

    def num_parameters(self) -> int:
        """Total parameter count (integer)."""
        return sum(p.numel() for p in self.parameters())

    def num_expert_parameters(self) -> int:
        """Parameters in expert MLPs only."""
        total = 0
        for layer in self.layers:
            total += layer.ffn.gate_up.numel()
            total += layer.ffn.down.numel()
        return total

    @staticmethod
    def from_config(
        vocab_size: int = 32000,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_experts: int = 4,
        max_seq_len: int = 2048,
    ) -> "TokenRoutedModel":
        """Convenience constructor."""
        config = TokenRoutedConfig(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_experts=num_experts,
            max_seq_len=max_seq_len,
            head_dim=hidden_dim // num_heads,
            expert_inter=hidden_dim // num_experts,
        )
        return TokenRoutedModel(config)
