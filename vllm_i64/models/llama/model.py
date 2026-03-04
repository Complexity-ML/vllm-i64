"""
vllm-i64 :: LlamaForCausalLM

Llama-family transformer — dense = modulo 1 in i64 routing theory.

    expert_id = token_id % 1 = 0  →  no routing, straight GEMM

Architecture:
    embed_tokens → [LlamaDecoderLayer × N] → norm → lm_head

Each LlamaDecoderLayer:
    h = h + attn(norm1(h))
    h = h + mlp(norm2(h))

Standard GQA + SwiGLU + RoPE + RMSNorm.

TP: Q/K/V ColumnParallel, O RowParallel, gate/up ColumnParallel, down RowParallel.
PP: decoder layers distributed across stages.

INL - 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from vllm_i64.layers.dense_mlp import DenseMLP
from vllm_i64.layers.rmsnorm import RMSNorm
from vllm_i64.layers.rotary import RotaryEmbedding, apply_rotary, apply_rotary_integer
from vllm_i64.parallel.tensor_parallel import (
    get_tp, ColumnParallelLinear, RowParallelLinear,
)


# =========================================================================
# Llama GQA Attention
# =========================================================================

class LlamaAttention(nn.Module):
    """
    GQA attention with RoPE and optional QK norm.

    TP: Q/K/V are ColumnParallel (output heads sharded),
        O is RowParallel (input sharded + all_reduce).
    """

    def __init__(self, config):
        super().__init__()
        tp = get_tp()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.tp_size = tp.tp_size

        self.num_heads_per_tp = self.num_heads // tp.tp_size
        self.num_kv_heads_per_tp = self.num_kv_heads // tp.tp_size
        self.num_kv_groups = self.num_heads_per_tp // max(self.num_kv_heads_per_tp, 1)

        has_bias = getattr(config, 'attention_bias', False)

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=has_bias,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=has_bias,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=has_bias,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=has_bias,
        )

        self.use_qk_norm = getattr(config, 'use_qk_norm', False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rope = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def _project_qkv(self, hidden: torch.Tensor) -> tuple:
        """Project hidden → Q, K, V. Uses fused INT8 matmul when quantized."""
        bsz = hidden.shape[0]
        if hasattr(self, 'qkv_int8'):
            from vllm_i64.core.quantization import int8_linear_native
            bias = getattr(self, 'qkv_bias', None)
            qkv = int8_linear_native(hidden, self.qkv_int8, self.qkv_scale, bias)
            q = qkv[..., :self.q_size]
            k = qkv[..., self.q_size:self.q_size + self.kv_size]
            v = qkv[..., self.q_size + self.kv_size:]
        else:
            q = self.q_proj(hidden)
            k = self.k_proj(hidden)
            v = self.v_proj(hidden)
        q = q.view(bsz, self.num_heads_per_tp, self.head_dim)
        k = k.view(bsz, self.num_kv_heads_per_tp, self.head_dim)
        v = v.view(bsz, self.num_kv_heads_per_tp, self.head_dim)
        return q, k, v

    def _apply_o_proj(self, out: torch.Tensor) -> torch.Tensor:
        """Apply output projection. Uses INT8 matmul + all_reduce when quantized."""
        if hasattr(self, 'o_int8'):
            from vllm_i64.core.quantization import int8_linear_native
            from vllm_i64.parallel.tensor_parallel import all_reduce
            bias = getattr(self, 'o_bias', None)
            result = int8_linear_native(out, self.o_int8, self.o_scale, bias)
            return all_reduce(result)
        # Attention may compute in float32; cast to match o_proj weight dtype
        if out.dtype != self.o_proj.linear.weight.dtype:
            out = out.to(self.o_proj.linear.weight.dtype)
        return self.o_proj(out)

    @property
    def _integer_pipeline(self) -> bool:
        """True when INT8 quantized — use integer RoPE + integer attention."""
        return hasattr(self, 'qkv_int8')

    def _apply_rope(self, q, k, positions):
        """Apply rotary embeddings — CUDA → Triton → integer/float fallback."""
        if self._integer_pipeline:
            cos_q14, sin_q14 = self.rope.forward_integer(positions)
            if q.is_cuda:
                # Priority 1: CUDA I64_rope_integer
                try:
                    from vllm_i64.kernels.cuda import get_i64_cuda_ops
                    cuda_ops = get_i64_cuda_ops()
                    if cuda_ops is not None:
                        q_out = cuda_ops.rope_integer_forward(q, cos_q14, sin_q14)
                        k_out = cuda_ops.rope_integer_forward(k, cos_q14, sin_q14)
                        return q_out, k_out
                except (ImportError, Exception):
                    pass
                # Priority 2: Triton fused integer RoPE
                try:
                    from vllm_i64.kernels.triton.I64_fused_rope import triton_fused_rope
                    q_out = triton_fused_rope(q, cos_q14, sin_q14, integer_mode=True)
                    if q_out is not None:
                        k_out = triton_fused_rope(k, cos_q14, sin_q14, integer_mode=True)
                        return q_out, k_out
                except ImportError:
                    pass
            # Priority 3: PyTorch fallback
            q = apply_rotary_integer(q, cos_q14, sin_q14)
            k = apply_rotary_integer(k, cos_q14, sin_q14)
        else:
            cos, sin = self.rope(positions)
            if q.is_cuda:
                # Priority 1: CUDA I64_rope
                try:
                    from vllm_i64.kernels.cuda import get_i64_cuda_ops
                    cuda_ops = get_i64_cuda_ops()
                    if cuda_ops is not None:
                        q_out = cuda_ops.rope_forward(q, cos, sin)
                        k_out = cuda_ops.rope_forward(k, cos, sin)
                        return q_out, k_out
                except (ImportError, Exception):
                    pass
                # Priority 2: Triton fused float RoPE
                try:
                    from vllm_i64.kernels.triton.I64_fused_rope import triton_fused_rope
                    q_out = triton_fused_rope(q, cos, sin, integer_mode=False)
                    if q_out is not None:
                        k_out = triton_fused_rope(k, cos, sin, integer_mode=False)
                        return q_out, k_out
                except ImportError:
                    pass
            # Priority 3: PyTorch fallback
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        return q, k

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache=None,
        layer_idx: int = 0,
        seq_ids: Optional[List[int]] = None,
        tokens_per_seq: Optional[List[int]] = None,
    ) -> torch.Tensor:
        bsz = hidden.shape[0]

        q, k, v = self._project_qkv(hidden)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self._apply_rope(q, k, positions)

        # === KV Cache path ===
        if kv_cache is not None and seq_ids is not None and tokens_per_seq is not None:
            return self._cached_attention(
                q, k, v, kv_cache, layer_idx, seq_ids, tokens_per_seq, positions,
            )

        # === Standard path: prefill without cache ===
        from vllm_i64.layers.attention import (
            is_flash_attn_available, flash_prefill_attention,
            naive_varlen_attention, naive_integer_varlen_attention,
        )

        scale = 1.0 / math.sqrt(self.head_dim)
        tps = tokens_per_seq if tokens_per_seq is not None else [bsz]

        if is_flash_attn_available() and q.is_cuda:
            out = flash_prefill_attention(q, k, v, tps, softmax_scale=scale)
        elif self._integer_pipeline:
            out = naive_integer_varlen_attention(q, k, v, tps, self.num_kv_groups, softmax_scale=scale)
        else:
            out = naive_varlen_attention(q, k, v, tps, self.num_kv_groups, softmax_scale=scale)

        out = out.reshape(bsz, self.num_heads_per_tp * self.head_dim)
        return self._apply_o_proj(out)

    def decode_step(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        layer_idx: int,
        seq_ids_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Decode-only attention. CUDA-graph compatible (all tensor ops)."""
        from vllm_i64.layers.attention import (
            is_flash_attn_available, flash_decode_attention,
            naive_paged_decode_attention, naive_integer_paged_decode_attention,
        )

        bsz = hidden.shape[0]

        q, k, v = self._project_qkv(hidden)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self._apply_rope(q, k, positions)

        kv_cache.write_kv_decode(layer_idx, seq_ids_tensor, positions, k, v)

        scale = 1.0 / math.sqrt(self.head_dim)
        k_cache, v_cache = kv_cache.get_cache_tensors(layer_idx)
        block_table = kv_cache.get_block_table_for_seqs_tensor(seq_ids_tensor).clamp(min=0)
        cache_seqlens = kv_cache.get_cache_seqlens_tensor(seq_ids_tensor)

        use_flash = is_flash_attn_available() and q.is_cuda

        if use_flash:
            q_4d = q.unsqueeze(1)
            out = flash_decode_attention(
                q_4d, k_cache, v_cache,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                softmax_scale=scale,
            )
            out = out.squeeze(1)
        elif self._integer_pipeline:
            out = naive_integer_paged_decode_attention(
                q, k_cache, v_cache, block_table, cache_seqlens,
                self.num_kv_groups, scale,
            )
        else:
            out = naive_paged_decode_attention(
                q, k_cache, v_cache, block_table, cache_seqlens,
                self.num_kv_groups, scale,
            )

        out = out.reshape(bsz, self.num_heads_per_tp * self.head_dim)
        return self._apply_o_proj(out)

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
        from vllm_i64.layers.attention import (
            is_flash_attn_available, flash_decode_attention,
            flash_prefill_with_cache, naive_cached_attention,
            naive_integer_cached_attention,
        )

        scale = 1.0 / math.sqrt(self.head_dim)
        use_flash = is_flash_attn_available() and q.is_cuda

        offset = 0
        for i, seq_id in enumerate(seq_ids):
            n = tokens_per_seq[i]
            pos_i = positions[offset:offset + n]
            k_i = k[offset:offset + n]
            v_i = v[offset:offset + n]
            kv_cache.write_kv_batch(layer_idx, seq_id, pos_i, k_i, v_i)
            offset += n

        is_pure_decode = all(n == 1 for n in tokens_per_seq)

        if use_flash and is_pure_decode:
            batch_size = len(seq_ids)
            q_4d = q.unsqueeze(1)

            k_cache, v_cache = kv_cache.get_cache_tensors(layer_idx)
            block_table = kv_cache.get_block_table_for_seqs(seq_ids).clamp(min=0)
            cache_seqlens = kv_cache.get_cache_seqlens(seq_ids)

            out = flash_decode_attention(
                q_4d, k_cache, v_cache,
                cache_seqlens=cache_seqlens,
                block_table=block_table,
                softmax_scale=scale,
            )
            out = out.squeeze(1)
            out = out.reshape(batch_size, self.num_heads_per_tp * self.head_dim)
            return self._apply_o_proj(out)

        if use_flash:
            cu_q = torch.zeros(len(seq_ids) + 1, dtype=torch.int32, device=q.device)
            cu_k = torch.zeros(len(seq_ids) + 1, dtype=torch.int32, device=q.device)
            k_parts, v_parts = [], []

            import numpy as np
            seq_ids_tensor = torch.from_numpy(np.array(seq_ids, dtype=np.int64)).to(
                device=kv_cache.seq_lens.device, non_blocking=True
            )
            cached_lens = kv_cache.seq_lens[seq_ids_tensor].tolist()

            for i, seq_id in enumerate(seq_ids):
                cu_q[i + 1] = cu_q[i] + tokens_per_seq[i]
                cu_k[i + 1] = cu_k[i] + cached_lens[i]
                kf, vf = kv_cache.read_kv(layer_idx, seq_id)
                k_parts.append(kf)
                v_parts.append(vf)

            k_all = torch.cat(k_parts, dim=0)
            v_all = torch.cat(v_parts, dim=0)

            out = flash_prefill_with_cache(
                q, k_all, v_all,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=max(tokens_per_seq),
                max_seqlen_k=max(cached_lens) if cached_lens else 0,
                softmax_scale=scale,
            )
            total_tokens = q.shape[0]
            out = out.reshape(total_tokens, self.num_heads_per_tp * self.head_dim)
            return self._apply_o_proj(out)

        # === Naive fallback ===
        _cached_fn = naive_integer_cached_attention if self._integer_pipeline else naive_cached_attention
        outputs = []
        offset = 0
        for i, seq_id in enumerate(seq_ids):
            n = tokens_per_seq[i]
            q_i = q[offset:offset + n]
            pos_i = positions[offset:offset + n]

            k_full, v_full = kv_cache.read_kv(layer_idx, seq_id)
            out_i = _cached_fn(
                q_i, k_full, v_full, self.num_kv_groups, pos_i, softmax_scale=scale,
            )
            out_i = out_i.reshape(n, self.num_heads_per_tp * self.head_dim)
            outputs.append(out_i)
            offset += n

        out = torch.cat(outputs, dim=0)
        if not hasattr(self, 'o_int8'):
            if out.dtype != self.o_proj.linear.weight.dtype:
                out = out.to(self.o_proj.linear.weight.dtype)
        return self._apply_o_proj(out)


# =========================================================================
# Llama Decoder Layer
# =========================================================================

class LlamaDecoderLayer(nn.Module):
    """
    Pre-norm transformer layer:
        h = h + attn(norm1(h))
        h = h + mlp(norm2(h))
    """

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = DenseMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache=None,
        layer_idx: int = 0,
        seq_ids: Optional[List[int]] = None,
        tokens_per_seq: Optional[List[int]] = None,
    ) -> torch.Tensor:
        residual = hidden
        hidden = self.self_attn(
            self.input_layernorm(hidden),
            positions,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            seq_ids=seq_ids,
            tokens_per_seq=tokens_per_seq,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.mlp(self.post_attention_layernorm(hidden))
        return residual + hidden

    def decode_step(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        layer_idx: int,
        seq_ids_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Decode-only layer forward. CUDA-graph compatible."""
        residual = hidden
        hidden = self.self_attn.decode_step(
            self.input_layernorm(hidden),
            positions,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            seq_ids_tensor=seq_ids_tensor,
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.mlp(self.post_attention_layernorm(hidden))
        return residual + hidden


# =========================================================================
# LlamaForCausalLM
# =========================================================================

class LlamaForCausalLM(nn.Module):
    """
    Llama-family causal LM — dense = modulo 1 in i64 routing theory.

    Same interface as ComplexityDeepModel:
      - forward(token_ids, positions, kv_cache, seq_ids, tokens_per_seq)
      - decode_step(token_ids, positions, kv_cache, seq_ids_tensor)
    """

    def __init__(self, config):
        super().__init__()
        from vllm_i64.parallel.pipeline_parallel import is_first_pp_rank, is_last_pp_rank
        from vllm_i64.parallel.pp_utils import make_layers

        self.config = config

        if is_first_pp_rank():
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx: LlamaDecoderLayer(config),
        )

        if is_last_pp_rank():
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = None

        self.tie_word_embeddings = getattr(config, 'tie_word_embeddings', False)
        if not self.tie_word_embeddings and is_last_pp_rank():
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache=None,
        seq_ids: Optional[List[int]] = None,
        tokens_per_seq: Optional[List[int]] = None,
        intermediate_tensors=None,
    ) -> torch.Tensor:
        from vllm_i64.parallel.pipeline_parallel import is_first_pp_rank, is_last_pp_rank
        from vllm_i64.parallel.pp_utils import IntermediateTensors

        if is_first_pp_rank():
            hidden = self.embed_tokens(token_ids.long())
        else:
            assert intermediate_tensors is not None
            hidden = intermediate_tensors["hidden_states"]

        for layer_idx in range(self.start_layer, self.end_layer):
            hidden = self.layers[layer_idx](
                hidden, positions,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                seq_ids=seq_ids,
                tokens_per_seq=tokens_per_seq,
            )

        if not is_last_pp_rank():
            return IntermediateTensors({"hidden_states": hidden})

        hidden = self.norm(hidden)

        if self.tie_word_embeddings and self.embed_tokens is not None:
            logits = F.linear(hidden, self.embed_tokens.weight)
        elif hasattr(self, 'lm_head'):
            logits = self.lm_head(hidden)
        else:
            raise RuntimeError(
                "No lm_head or tied embeddings found — cannot compute logits."
            )

        return logits

    def decode_step(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        seq_ids_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """CUDA-graph compatible decode path."""
        hidden = self.embed_tokens(token_ids.long())

        for layer_idx in range(self.start_layer, self.end_layer):
            hidden = self.layers[layer_idx].decode_step(
                hidden, positions,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                seq_ids_tensor=seq_ids_tensor,
            )

        hidden = self.norm(hidden)

        if self.tie_word_embeddings and self.embed_tokens is not None:
            logits = F.linear(hidden, self.embed_tokens.weight)
        elif hasattr(self, 'lm_head'):
            logits = self.lm_head(hidden)
        else:
            raise RuntimeError(
                "No lm_head or tied embeddings found — cannot compute logits."
            )

        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
