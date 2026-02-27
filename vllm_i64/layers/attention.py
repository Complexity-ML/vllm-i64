"""
vllm-i64 :: Attention Backends

Abstraction over attention implementations:
  - FlashAttention (GPU, when flash-attn is installed)
  - Naive torch.bmm (CPU fallback, or when flash-attn unavailable)

Integer-first: all indexing (cu_seqlens, cache_seqlens, block_table) is integer.
Only the Q/K/V/output tensors are float.

INL - 2025
"""

import torch
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

# Try to import flash_attn
_FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kv_cache
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass


def is_flash_attn_available() -> bool:
    return _FLASH_ATTN_AVAILABLE


def compute_cu_seqlens(tokens_per_seq: List[int], device: torch.device) -> Tuple[torch.Tensor, int]:
    """
    Compute cumulative sequence lengths for flash_attn_varlen_func.
    Returns (cu_seqlens, max_seqlen) — cu_seqlens is int32.
    """
    cu = torch.zeros(len(tokens_per_seq) + 1, dtype=torch.int32, device=device)
    for i, n in enumerate(tokens_per_seq):
        cu[i + 1] = cu[i] + n
    max_seqlen = max(tokens_per_seq) if tokens_per_seq else 0
    return cu, max_seqlen


# =========================================================================
# Flash Attention (GPU only, requires flash-attn)
# =========================================================================

def flash_prefill_attention(
    q: torch.Tensor,           # (total_tokens, num_heads, head_dim)
    k: torch.Tensor,           # (total_tokens, num_kv_heads, head_dim)
    v: torch.Tensor,           # (total_tokens, num_kv_heads, head_dim)
    tokens_per_seq: List[int],
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    FlashAttention variable-length prefill.
    GQA handled natively by flash_attn (num_heads != num_kv_heads).
    """
    cu_seqlens, max_seqlen = compute_cu_seqlens(tokens_per_seq, q.device)

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
        softmax_scale=softmax_scale,
    )
    return out  # (total_tokens, num_heads, head_dim)


def flash_prefill_with_cache(
    q: torch.Tensor,           # (total_new_tokens, num_heads, head_dim)
    k_all: torch.Tensor,       # (total_cached_tokens, num_kv_heads, head_dim)
    v_all: torch.Tensor,       # (total_cached_tokens, num_kv_heads, head_dim)
    cu_seqlens_q: torch.Tensor,  # (num_seqs + 1,) int32
    cu_seqlens_k: torch.Tensor,  # (num_seqs + 1,) int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """FlashAttention prefill with different Q/K lengths (for cached prefill)."""
    out = flash_attn_varlen_func(
        q, k_all, v_all,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=True,
        softmax_scale=softmax_scale,
    )
    return out


def flash_decode_attention(
    q: torch.Tensor,              # (batch, 1, num_heads, head_dim)
    k_cache: torch.Tensor,        # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,        # same
    cache_seqlens: torch.Tensor,  # (batch,) int32
    block_table: torch.Tensor,    # (batch, max_blocks) int32
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """FlashAttention decode with paged KV cache — zero K/V materialization."""
    out = flash_attn_with_kv_cache(
        q, k_cache, v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=True,
    )
    return out  # (batch, 1, num_heads, head_dim)


# =========================================================================
# Naive Fallback (CPU or no flash_attn)
# =========================================================================

def naive_varlen_attention(
    q: torch.Tensor,           # (total_tokens, num_heads, head_dim)
    k: torch.Tensor,           # (total_tokens, num_kv_heads, head_dim)
    v: torch.Tensor,           # (total_tokens, num_kv_heads, head_dim)
    tokens_per_seq: List[int],
    num_kv_groups: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Naive variable-length attention fallback.
    Processes each sequence independently for correct causal masking.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    # Align dtypes (CPU ops may upcast q to float32 while k/v stay fp16)
    compute_dtype = q.dtype
    if k.dtype != compute_dtype:
        k = k.to(compute_dtype)
    if v.dtype != compute_dtype:
        v = v.to(compute_dtype)

    outputs = []
    offset = 0

    for n in tokens_per_seq:
        q_i = q[offset:offset + n]
        k_i = k[offset:offset + n]
        v_i = v[offset:offset + n]

        # GQA expand
        if num_kv_groups > 1:
            k_i = k_i.repeat_interleave(num_kv_groups, dim=1)
            v_i = v_i.repeat_interleave(num_kv_groups, dim=1)

        # (num_heads, n, head_dim)
        q_t = q_i.transpose(0, 1)
        k_t = k_i.transpose(0, 1)
        v_t = v_i.transpose(0, 1)

        attn = torch.bmm(q_t, k_t.transpose(1, 2)) * softmax_scale

        if n > 1:
            causal_mask = torch.triu(
                torch.full((n, n), float('-inf'), device=q.device, dtype=q.dtype),
                diagonal=1,
            )
            attn = attn + causal_mask.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out_i = torch.bmm(attn, v_t)  # (num_heads, n, head_dim)
        out_i = out_i.transpose(0, 1)  # (n, num_heads, head_dim)
        outputs.append(out_i)
        offset += n

    return torch.cat(outputs, dim=0)


def naive_cached_attention(
    q: torch.Tensor,           # (n_tokens, num_heads, head_dim)
    k_full: torch.Tensor,      # (history_len, num_kv_heads, head_dim)
    v_full: torch.Tensor,      # (history_len, num_kv_heads, head_dim)
    num_kv_groups: int,
    positions: torch.Tensor,   # (n_tokens,) int32
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Naive cached attention fallback for a single request.
    Q attends to all of k_full/v_full with causal masking.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    n = q.shape[0]

    # Align dtypes (CPU ops may upcast q to float32 while cache stays fp16)
    compute_dtype = q.dtype
    if k_full.dtype != compute_dtype:
        k_full = k_full.to(compute_dtype)
    if v_full.dtype != compute_dtype:
        v_full = v_full.to(compute_dtype)

    # GQA expand
    if num_kv_groups > 1:
        k_full = k_full.repeat_interleave(num_kv_groups, dim=1)
        v_full = v_full.repeat_interleave(num_kv_groups, dim=1)

    q_t = q.transpose(0, 1)       # (num_heads, n, head_dim)
    k_t = k_full.transpose(0, 1)  # (num_heads, history, head_dim)
    v_t = v_full.transpose(0, 1)

    attn = torch.bmm(q_t, k_t.transpose(1, 2)) * softmax_scale

    if n > 1:
        total = k_full.shape[0]
        q_pos = positions.unsqueeze(1).to(compute_dtype)
        k_pos = torch.arange(total, device=q.device, dtype=compute_dtype).unsqueeze(0)
        causal = torch.where(k_pos <= q_pos, torch.zeros(1, device=q.device, dtype=compute_dtype),
                             torch.tensor(float('-inf'), device=q.device, dtype=compute_dtype))
        attn = attn + causal.unsqueeze(0)

    attn = F.softmax(attn, dim=-1, dtype=compute_dtype)
    out = torch.bmm(attn, v_t)   # (num_heads, n, head_dim)
    return out.transpose(0, 1)    # (n, num_heads, head_dim)
