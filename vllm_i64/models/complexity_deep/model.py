"""
vllm-i64 :: Complexity Deep Model

Specific implementation for the Complexity Deep architecture.
Uses the GENERIC TokenRoutedMLP layer + adds:
  - INL Dynamics (PID control, velocity tracking)
  - Mu-Guided Attention (mu biases Q, K, V)
  - Mu-Guided Routing (mu overrides i64 base routing)
  - Mu Residual Highway (accumulated context across layers)
  - GQA, QK Norm

Tensor Parallelism:
  - Q/K/V projections: ColumnParallel (heads sharded)
  - O projection: RowParallel + all_reduce
  - Expert MLP: sharded on intermediate dim (via TokenRoutedMLP)
  - INL Dynamics: replicated (small controller)
  - Embeddings: replicated

Other models can use TokenRoutedMLP directly without any of this.

INL - 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from vllm_i64.layers.token_routed_mlp import TokenRoutedMLP
from vllm_i64.layers.rmsnorm import RMSNorm
from vllm_i64.layers.rotary import RotaryEmbedding, apply_rotary
from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
from vllm_i64.parallel.tensor_parallel import (
    get_tp, ColumnParallelLinear, RowParallelLinear,
)


# =========================================================================
# INL Dynamics (Complexity Deep specific)
# =========================================================================

class INLDynamics(nn.Module):
    """
    INL Dynamics — PID-like control with velocity tracking.

        mu(h) = mu_base + mu_proj(h)
        error = h - mu(h)
        v_next = alpha * v - beta * error
        h_next = h + dt * gate * v_next

    alpha, beta, gate learned via controller MLP, clamped via sigmoid.
    """

    def __init__(self, hidden_size: int, controller_hidden: int = 64, dt: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        self.mu = nn.Parameter(torch.zeros(hidden_size))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

        self.controller_in = nn.Linear(hidden_size * 2, controller_hidden)
        self.controller_out = nn.Linear(controller_hidden, hidden_size * 3)

        with torch.no_grad():
            bias = self.controller_out.bias
            bias[:hidden_size].fill_(2.2)
            bias[hidden_size:hidden_size*2].fill_(-2.2)
            bias[hidden_size*2:].fill_(0.0)
            self.controller_out.weight.normal_(0, 0.01)

    def forward(
        self, h: torch.Tensor, v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if v is None:
            v = torch.zeros_like(h)

        hv = torch.cat([h, v], dim=-1)
        ctrl = F.silu(self.controller_in(hv))
        ctrl_out = self.controller_out(ctrl)

        alpha_raw, beta_raw, gate_raw = torch.split(ctrl_out, self.hidden_size, dim=-1)
        alpha = torch.sigmoid(alpha_raw)
        beta = torch.clamp(F.softplus(beta_raw), max=2.0)
        gate = torch.sigmoid(gate_raw)

        mu_contextual = self.mu + self.mu_proj(h)
        error = h - mu_contextual
        v_next = alpha * v - beta * error
        v_next = torch.clamp(v_next, min=-10.0, max=10.0)
        h_next = h + self.dt * gate * v_next

        return h_next, v_next, mu_contextual


# =========================================================================
# Mu-Guided Token-Routed MLP (extends generic TokenRoutedMLP)
# =========================================================================

class MuGuidedTokenRoutedMLP(TokenRoutedMLP):
    """
    Complexity Deep extension of TokenRoutedMLP.

    Adds mu-guided routing on top of the generic i64 routing:
        base_id = token_id % num_experts          (generic, i64)
        mu_logits = mu_router(mu)                  (Complexity Deep specific)
        combined = one_hot(base_id) * 10 + mu_logits
        expert_id = argmax(combined)

    Without mu, falls back to pure i64 routing (generic behavior).
    """

    _BASE_ROUTING_SCALE = 10.0

    def __init__(self, config: ComplexityDeepConfig):
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            vocab_size=config.vocab_size,
        )
        # Mu-guided routing bias (Complexity Deep specific)
        self.mu_router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        nn.init.zeros_(self.mu_router.weight)

    def route(self, token_ids, num_tokens, device, mu=None):
        """i64 routing + optional mu bias."""
        base_ids = super().route(token_ids, num_tokens, device)

        if mu is not None:
            mu_logits = self.mu_router(mu)
            base_one_hot = F.one_hot(base_ids, self.num_experts).float()
            combined = base_one_hot * self._BASE_ROUTING_SCALE + mu_logits
            return combined.argmax(dim=-1)

        return base_ids

    def forward(self, x, token_ids=None, mu=None, **kwargs):
        expert_ids = self.route(token_ids, x.shape[0], x.device, mu=mu)
        return self.expert_forward(x, expert_ids)


# =========================================================================
# Mu-Guided Attention (GQA + QK norm + RoPE)
# =========================================================================

class MuGuidedAttention(nn.Module):
    """
    Complexity Deep attention with Tensor Parallelism.

    TP strategy:
      - Q/K/V: ColumnParallel — heads sharded across ranks
      - O: RowParallel — input dim sharded, all_reduce on output
      - mu_to_q/k/v: ColumnParallel (matches Q/K/V sharding)

    Complexity Deep specific: mu biases Q, K, V from previous layer
    Generic parts: GQA, QK norm, RoPE
    """

    def __init__(self, config: ComplexityDeepConfig):
        super().__init__()
        tp = get_tp()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.tp_size = tp.tp_size

        # TP-sharded head counts
        self.num_heads_per_tp = self.num_heads // tp.tp_size
        self.num_kv_heads_per_tp = self.num_kv_heads // tp.tp_size
        self.num_kv_groups = self.num_heads_per_tp // self.num_kv_heads_per_tp

        # Q/K/V — ColumnParallel (output dim = heads * head_dim, sharded)
        self.q_proj = ColumnParallelLinear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = ColumnParallelLinear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = ColumnParallelLinear(config.hidden_size, self.num_kv_heads * self.head_dim)

        # O — RowParallel (input dim sharded, all_reduce on output)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, config.hidden_size)

        # Mu-guided projections — ColumnParallel (matches Q/K/V sharding)
        self.mu_to_q = ColumnParallelLinear(config.hidden_size, self.num_heads * self.head_dim)
        self.mu_to_k = ColumnParallelLinear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.mu_to_v = ColumnParallelLinear(config.hidden_size, self.num_kv_heads * self.head_dim)
        for proj in [self.mu_to_q, self.mu_to_k, self.mu_to_v]:
            nn.init.normal_(proj.linear.weight, std=0.01)

        # QK Norm (per head_dim, replicated)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # RoPE (replicated)
        self.rope = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        mu_prev: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache=None,
        layer_idx: int = 0,
        seq_ids: Optional[List[int]] = None,
        tokens_per_seq: Optional[List[int]] = None,
    ) -> torch.Tensor:
        bsz = hidden.shape[0]

        # ColumnParallel: output is (bsz, heads_per_tp * head_dim)
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)

        # Mu-guidance (Complexity Deep specific)
        if mu_prev is not None:
            q = q + self.mu_to_q(mu_prev)
            k = k + self.mu_to_k(mu_prev)
            v = v + self.mu_to_v(mu_prev)

        # Reshape to per-TP head counts
        q = q.view(bsz, self.num_heads_per_tp, self.head_dim)
        k = k.view(bsz, self.num_kv_heads_per_tp, self.head_dim)
        v = v.view(bsz, self.num_kv_heads_per_tp, self.head_dim)

        # QK Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE (integer positions → float rotation)
        cos, sin = self.rope(positions)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # === KV Cache path: per-request attention with caching ===
        if kv_cache is not None and seq_ids is not None and tokens_per_seq is not None:
            return self._cached_attention(
                q, k, v, kv_cache, layer_idx, seq_ids, tokens_per_seq, positions,
            )

        # === Standard path: batched attention with causal mask ===
        # GQA: repeat KV heads within this TP shard
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Attention (on this rank's heads only)
        scale = 1.0 / math.sqrt(self.head_dim)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        attn = torch.bmm(q, k.transpose(1, 2)) * scale

        # Causal mask
        seq_len = q.shape[1]
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=hidden.device),
                diagonal=1,
            )
            attn = attn + causal_mask.unsqueeze(0)

        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)

        # Reshape back: (heads_per_tp, bsz, head_dim) → (bsz, heads_per_tp * head_dim)
        out = out.transpose(0, 1).reshape(bsz, self.num_heads_per_tp * self.head_dim)

        # RowParallel: matmul + all_reduce across TP ranks
        return self.o_proj(out)

    def _cached_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache,
        layer_idx: int,
        seq_ids: List[int],
        tokens_per_seq: List[int],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Per-request attention with KV caching."""
        scale = 1.0 / math.sqrt(self.head_dim)
        outputs = []
        offset = 0

        for i, seq_id in enumerate(seq_ids):
            n = tokens_per_seq[i]
            q_i = q[offset:offset + n]
            k_i = k[offset:offset + n]
            v_i = v[offset:offset + n]

            # Write new K/V to cache at their positions
            for t in range(n):
                pos = positions[offset + t].item()
                kv_cache.write_kv(layer_idx, seq_id, pos, k_i[t], v_i[t])

            # Read all cached K/V (includes what we just wrote)
            k_full, v_full = kv_cache.read_kv(layer_idx, seq_id)

            # GQA expand on cached K/V
            if self.num_kv_groups > 1:
                k_full = k_full.repeat_interleave(self.num_kv_groups, dim=1)
                v_full = v_full.repeat_interleave(self.num_kv_groups, dim=1)

            # Attention: q_i against all cached K/V
            q_t = q_i.transpose(0, 1)
            k_t = k_full.transpose(0, 1)
            v_t = v_full.transpose(0, 1)

            attn = torch.bmm(q_t, k_t.transpose(1, 2)) * scale

            # Causal mask (needed for prefill when n > 1)
            if n > 1:
                total = k_full.shape[0]
                q_pos = positions[offset:offset + n].unsqueeze(1).float()
                k_pos = torch.arange(total, device=q.device, dtype=torch.float32).unsqueeze(0)
                causal = torch.where(k_pos <= q_pos, 0.0, float('-inf'))
                attn = attn + causal.unsqueeze(0)

            attn = F.softmax(attn, dim=-1)
            out_i = torch.bmm(attn, v_t)
            out_i = out_i.transpose(0, 1).reshape(n, self.num_heads_per_tp * self.head_dim)
            outputs.append(out_i)
            offset += n

        out = torch.cat(outputs, dim=0)
        return self.o_proj(out)


# =========================================================================
# Complexity Deep Decoder Layer
# =========================================================================

class ComplexityDecoderLayer(nn.Module):
    """
    Complexity Deep decoder layer:
      1. RMSNorm → Mu-Guided Attention
      2. INL Dynamics (PID → h_next, v_next, mu)
      3. Residual
      4. RMSNorm → Mu-Guided Token-Routed MLP
      5. Residual
    """

    def __init__(self, config: ComplexityDeepConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MuGuidedAttention(config)
        self.dynamics = INLDynamics(
            hidden_size=config.hidden_size,
            controller_hidden=config.dynamics_controller_hidden,
            dt=config.dynamics_dt,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MuGuidedTokenRoutedMLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        velocity: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        mu_prev: Optional[torch.Tensor] = None,
        kv_cache=None,
        layer_idx: int = 0,
        seq_ids: Optional[List[int]] = None,
        tokens_per_seq: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(
            hidden, positions, mu_prev=mu_prev,
            kv_cache=kv_cache, layer_idx=layer_idx,
            seq_ids=seq_ids, tokens_per_seq=tokens_per_seq,
        )

        hidden, velocity, mu_current = self.dynamics(hidden, velocity)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden, token_ids=token_ids, mu=mu_current)
        hidden = residual + hidden

        return hidden, velocity, mu_current


# =========================================================================
# Complexity Deep Model
# =========================================================================

class ComplexityDeepModel(nn.Module):
    """
    Complexity Deep transformer.

    Complexity Deep specific:
      - INL Dynamics per layer
      - Mu Residual Highway: mu_prev = mu_current + 0.1 * mu_residual
      - Mu-guided attention and routing
      - Tied embeddings

    Generic (via layers/):
      - TokenRoutedMLP base (i64 routing, SwiGLU)
      - RMSNorm, RoPE
    """

    def __init__(self, config: ComplexityDeepConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            ComplexityDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache=None,
        seq_ids: Optional[List[int]] = None,
        tokens_per_seq: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Returns logits: (batch, vocab_size)."""
        hidden = self.embed_tokens(token_ids.long())
        velocity = torch.zeros_like(hidden)

        mu_prev = None
        mu_residual = None

        for layer_idx, layer in enumerate(self.layers):
            hidden, velocity, mu_current = layer(
                hidden, positions, velocity,
                token_ids=token_ids,
                mu_prev=mu_prev,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                seq_ids=seq_ids,
                tokens_per_seq=tokens_per_seq,
            )

            # Mu Residual Highway
            if mu_residual is None:
                mu_residual = mu_current.clone()
            else:
                mu_residual = mu_residual + mu_current
            mu_prev = mu_current + 0.1 * mu_residual

        hidden = self.norm(hidden)

        if self.tie_word_embeddings:
            logits = F.linear(hidden, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden)

        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
