"""
vllm-i64 :: MixtralForCausalLM

Mixtral = Mistral architecture + sparse MoE (8 experts, top-2 routing).
Attention from llama, MoE layer replaces dense MLP.

HF weight naming:
    layers.N.self_attn.{q,k,v,o}_proj.weight  — same as Llama
    layers.N.block_sparse_moe.gate.weight      — router
    layers.N.block_sparse_moe.experts.E.w1.weight  — gate projection
    layers.N.block_sparse_moe.experts.E.w2.weight  — down projection
    layers.N.block_sparse_moe.experts.E.w3.weight  — up projection

INL - 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from vllm_i64.layers.moe import MixtralMoE
from vllm_i64.layers.rmsnorm import RMSNorm
from vllm_i64.models.llama.model import LlamaAttention


# =========================================================================
# Mixtral Decoder Layer
# =========================================================================

class MixtralDecoderLayer(nn.Module):
    """
    Pre-norm transformer layer with sparse MoE:
        h = h + attn(norm1(h))
        h = h + moe(norm2(h))
    """

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_sparse_moe = MixtralMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

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
        hidden = self.block_sparse_moe(self.post_attention_layernorm(hidden))
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
        hidden = self.block_sparse_moe(self.post_attention_layernorm(hidden))
        return residual + hidden


# =========================================================================
# MixtralForCausalLM
# =========================================================================

class MixtralForCausalLM(nn.Module):
    """
    Mixtral sparse MoE causal LM.

    Same interface as LlamaForCausalLM / ComplexityDeepModel:
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
            lambda idx: MixtralDecoderLayer(config),
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
