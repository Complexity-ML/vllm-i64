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
    import os
    if os.environ.get("VLLM_NO_FLASH_ATTN", "").strip() == "1":
        return False
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
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Naive variable-length attention fallback.
    Processes each sequence independently for correct causal masking.
    Optional sliding_window limits attention span to last N positions.
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
            # Sliding window: mask out positions beyond window size
            if sliding_window is not None:
                sw_mask = torch.tril(
                    torch.full((n, n), float('-inf'), device=q.device, dtype=q.dtype),
                    diagonal=-sliding_window,
                )
                causal_mask = causal_mask + sw_mask
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
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Naive cached attention fallback for a single request.
    Q attends to all of k_full/v_full with causal masking.
    Optional sliding_window limits attention to last N positions.
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

    total = k_full.shape[0]
    q_pos = positions.unsqueeze(1).to(compute_dtype)
    k_pos = torch.arange(total, device=q.device, dtype=compute_dtype).unsqueeze(0)

    # Causal mask (use pre-computed scalars to avoid per-call tensor allocation)
    _zero = torch.zeros(1, device=q.device, dtype=compute_dtype)
    _neginf = torch.full((1,), float('-inf'), device=q.device, dtype=compute_dtype)
    causal = torch.where(k_pos <= q_pos, _zero, _neginf)

    # Sliding window: mask positions outside window
    if sliding_window is not None:
        sw_mask = torch.where(q_pos - k_pos < sliding_window, _zero, _neginf)
        causal = causal + sw_mask

    attn = attn + causal.unsqueeze(0)

    attn = F.softmax(attn, dim=-1, dtype=compute_dtype)
    out = torch.bmm(attn, v_t)   # (num_heads, n, head_dim)
    return out.transpose(0, 1)    # (n, num_heads, head_dim)


def _gpu_softmax_integer(scores: torch.Tensor) -> torch.Tensor:
    """Integer softmax — Triton GPU kernel when available, else CPU LUT."""
    if scores.is_cuda:
        try:
            from vllm_i64.kernels.triton.I64_fused_softmax import triton_fused_softmax_integer
            # Triton expects 2D (N, D) — reshape per-head scores
            orig_shape = scores.shape
            scores_2d = scores.reshape(-1, scores.shape[-1])
            result = triton_fused_softmax_integer(scores_2d)
            if result is not None:
                return result.reshape(orig_shape)
        except ImportError:
            pass
    from vllm_i64.layers.moe import softmax_integer
    return softmax_integer(scores)


def naive_integer_varlen_attention(
    q: torch.Tensor,           # (total_tokens, num_heads, head_dim)
    k: torch.Tensor,           # (total_tokens, num_kv_heads, head_dim)
    v: torch.Tensor,           # (total_tokens, num_kv_heads, head_dim)
    tokens_per_seq: List[int],
    num_kv_groups: int,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Integer attention: INT8 Q@K^T scores + softmax_integer + float V multiply.

    Uses torch._int_mm (GPU SM80+) for true INT8 score computation.
    Falls back to float bmm + softmax_integer on CPU / older GPUs.
    Causal mask uses -1e4 (not -inf) for softmax_integer compatibility.
    """
    from vllm_i64.layers.moe import softmax_integer
    from vllm_i64.core.quantization import quantize_activations_int8, _INT_MM_AVAILABLE

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    compute_dtype = q.dtype
    if k.dtype != compute_dtype:
        k = k.to(compute_dtype)
    if v.dtype != compute_dtype:
        v = v.to(compute_dtype)

    use_int_mm = _INT_MM_AVAILABLE and q.is_cuda
    outputs = []
    offset = 0

    for n in tokens_per_seq:
        q_i = q[offset:offset + n]
        k_i = k[offset:offset + n]
        v_i = v[offset:offset + n]

        if num_kv_groups > 1:
            k_i = k_i.repeat_interleave(num_kv_groups, dim=1)
            v_i = v_i.repeat_interleave(num_kv_groups, dim=1)

        q_t = q_i.transpose(0, 1)  # (num_heads, n, head_dim)
        k_t = k_i.transpose(0, 1)
        v_t = v_i.transpose(0, 1)
        num_heads = q_t.shape[0]

        if use_int_mm and n > 0:
            # INT8 Q@K^T per head — _int_mm is 2D only
            scores = torch.zeros(num_heads, n, n, device=q.device, dtype=torch.float32)
            for h in range(num_heads):
                q_h = q_t[h]  # (n, head_dim)
                k_h = k_t[h]
                q_int8, q_scale = quantize_activations_int8(q_h)
                k_int8, k_scale = quantize_activations_int8(k_h)
                # torch._int_mm requires size(0) > 16 on CUDA — pad small seqs
                if n <= 16:
                    pad = 17 - n
                    q_int8 = torch.nn.functional.pad(q_int8, (0, 0, 0, pad))
                    k_int8 = torch.nn.functional.pad(k_int8, (0, 0, 0, pad))
                    q_scale = torch.nn.functional.pad(q_scale, (0, pad))
                    k_scale = torch.nn.functional.pad(k_scale, (0, pad))
                    s_i32 = torch._int_mm(q_int8, k_int8.t().contiguous())
                    s_i32 = s_i32[:n, :n]
                    q_scale = q_scale[:n]
                    k_scale = k_scale[:n]
                else:
                    s_i32 = torch._int_mm(q_int8, k_int8.t().contiguous())
                scores[h] = s_i32.float() * (q_scale.unsqueeze(1) * k_scale.unsqueeze(0))
            scores = scores * softmax_scale
        else:
            # Float bmm fallback — still use integer softmax
            scores = torch.bmm(q_t.float(), k_t.float().transpose(1, 2)) * softmax_scale

        # Causal mask — -1e4 instead of -inf for softmax_integer (Q7: -1e4*128 clamps to LUT min)
        if n > 1:
            causal_mask = torch.triu(
                torch.full((n, n), -1e4, device=q.device, dtype=torch.float32),
                diagonal=1,
            )
            if sliding_window is not None:
                sw_mask = torch.tril(
                    torch.full((n, n), -1e4, device=q.device, dtype=torch.float32),
                    diagonal=-sliding_window,
                )
                causal_mask = causal_mask + sw_mask
            scores = scores + causal_mask.unsqueeze(0)

        # Integer softmax — Triton GPU kernel when available
        attn = _gpu_softmax_integer(scores)

        # Float V multiply (V values need full precision for output quality)
        out_i = torch.bmm(attn.to(v_t.dtype), v_t)
        out_i = out_i.transpose(0, 1)
        outputs.append(out_i)
        offset += n

    return torch.cat(outputs, dim=0)


def naive_integer_cached_attention(
    q: torch.Tensor,           # (n_tokens, num_heads, head_dim)
    k_full: torch.Tensor,      # (history_len, num_kv_heads, head_dim)
    v_full: torch.Tensor,      # (history_len, num_kv_heads, head_dim)
    num_kv_groups: int,
    positions: torch.Tensor,   # (n_tokens,) int32
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    Integer cached attention: INT8 Q@K^T + softmax_integer for prefill with cache.
    Falls back to float bmm on CPU / no _int_mm. Always uses integer softmax.
    """
    from vllm_i64.layers.moe import softmax_integer
    from vllm_i64.core.quantization import quantize_activations_int8, _INT_MM_AVAILABLE

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    n = q.shape[0]

    compute_dtype = q.dtype
    if k_full.dtype != compute_dtype:
        k_full = k_full.to(compute_dtype)
    if v_full.dtype != compute_dtype:
        v_full = v_full.to(compute_dtype)

    if num_kv_groups > 1:
        k_full = k_full.repeat_interleave(num_kv_groups, dim=1)
        v_full = v_full.repeat_interleave(num_kv_groups, dim=1)

    q_t = q.transpose(0, 1)       # (num_heads, n, head_dim)
    k_t = k_full.transpose(0, 1)  # (num_heads, history, head_dim)
    v_t = v_full.transpose(0, 1)
    num_heads = q_t.shape[0]
    history = k_full.shape[0]

    use_int_mm = _INT_MM_AVAILABLE and q.is_cuda

    if use_int_mm and n > 0 and history > 0:
        scores = torch.zeros(num_heads, n, history, device=q.device, dtype=torch.float32)
        for h in range(num_heads):
            q_int8, q_scale = quantize_activations_int8(q_t[h])
            k_int8, k_scale = quantize_activations_int8(k_t[h])
            # torch._int_mm requires size(0) > 16 on CUDA — pad small seqs
            need_pad_q = n <= 16
            need_pad_k = history <= 16
            q_pad, k_pad = q_int8, k_int8
            qs, ks = q_scale, k_scale
            if need_pad_q:
                pq = 17 - n
                q_pad = torch.nn.functional.pad(q_int8, (0, 0, 0, pq))
                qs = torch.nn.functional.pad(q_scale, (0, pq))
            if need_pad_k:
                pk = 17 - history
                k_pad = torch.nn.functional.pad(k_int8, (0, 0, 0, pk))
                ks = torch.nn.functional.pad(k_scale, (0, pk))
            s_i32 = torch._int_mm(q_pad, k_pad.t().contiguous())
            if need_pad_q or need_pad_k:
                s_i32 = s_i32[:n, :history]
                qs = qs[:n]
                ks = ks[:history]
            scores[h] = s_i32.float() * (qs.unsqueeze(1) * ks.unsqueeze(0))
        scores = scores * softmax_scale
    else:
        scores = torch.bmm(q_t.float(), k_t.float().transpose(1, 2)) * softmax_scale

    # Causal mask — use -1e4 for softmax_integer compatibility
    total = k_full.shape[0]
    q_pos = positions.unsqueeze(1).float()
    k_pos = torch.arange(total, device=q.device, dtype=torch.float32).unsqueeze(0)
    causal = torch.where(k_pos <= q_pos, 0.0, -1e4)

    if sliding_window is not None:
        sw_mask = torch.where(q_pos - k_pos < sliding_window, 0.0, -1e4)
        causal = causal + sw_mask

    scores = scores + causal.unsqueeze(0)

    # Integer softmax — Triton GPU kernel when available
    attn = _gpu_softmax_integer(scores)

    # Float V multiply
    out = torch.bmm(attn.to(v_t.dtype), v_t)
    return out.transpose(0, 1)


def naive_integer_paged_decode_attention(
    q: torch.Tensor,              # (batch, num_heads, head_dim)
    k_cache: torch.Tensor,        # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,        # same
    block_table: torch.Tensor,    # (batch, max_blocks_per_seq) int32
    cache_seqlens: torch.Tensor,  # (batch,) int32
    num_kv_groups: int = 1,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Integer paged decode attention: float Q@K^T + softmax_integer.

    Q@K^T stays float (tiny 1×seq_len matmul, no INT8 benefit).
    softmax_integer replaces F.softmax for deterministic integer weights.
    """
    from vllm_i64.layers.moe import softmax_integer

    batch = q.shape[0]
    bs = k_cache.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    compute_dtype = q.dtype
    outputs = []

    for b in range(batch):
        seq_len = cache_seqlens[b].item()
        if seq_len == 0:
            outputs.append(torch.zeros_like(q[b]))
            continue

        # Gather K/V from blocks
        num_blocks_needed = (seq_len + bs - 1) // bs
        k_parts = []
        v_parts = []
        for blk_idx in range(num_blocks_needed):
            block_id = block_table[b, blk_idx].item()
            if block_id < 0:
                break
            if blk_idx == num_blocks_needed - 1:
                remainder = seq_len - blk_idx * bs
                k_parts.append(k_cache[block_id, :remainder].to(compute_dtype))
                v_parts.append(v_cache[block_id, :remainder].to(compute_dtype))
            else:
                k_parts.append(k_cache[block_id].to(compute_dtype))
                v_parts.append(v_cache[block_id].to(compute_dtype))

        k_seq = torch.cat(k_parts, dim=0)
        v_seq = torch.cat(v_parts, dim=0)

        if num_kv_groups > 1:
            k_seq = k_seq.repeat_interleave(num_kv_groups, dim=1)
            v_seq = v_seq.repeat_interleave(num_kv_groups, dim=1)

        q_b = q[b].unsqueeze(1)           # (num_heads, 1, head_dim)
        k_t = k_seq.transpose(0, 1)       # (num_heads, seq_len, head_dim)
        v_t = v_seq.transpose(0, 1)

        attn = torch.bmm(q_b.float(), k_t.float().transpose(1, 2)) * softmax_scale
        attn = _gpu_softmax_integer(attn)
        out_b = torch.bmm(attn.to(v_t.dtype), v_t).squeeze(1)
        outputs.append(out_b)

    return torch.stack(outputs)


def _tensor_paged_decode_attention(
    q: torch.Tensor,              # (batch, num_heads, head_dim)
    k_cache: torch.Tensor,        # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,
    block_table: torch.Tensor,    # (batch, max_blocks_per_seq) int32
    cache_seqlens: torch.Tensor,  # (batch,) int32
    num_kv_groups: int,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Fully vectorized paged decode attention — CUDA graph compatible.

    No Python loops, no .item() calls. Gathers all K/V blocks at once via
    advanced indexing, masks padding positions with -inf before softmax.

    Used automatically during CUDA graph capture instead of naive_paged_decode_attention.
    """
    batch, num_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    max_blocks = block_table.shape[1]
    max_seq_len = max_blocks * block_size

    # Gather all K/V blocks for all seqs in one shot (no loop over batch or blocks)
    bt = block_table.clamp(min=0).long().view(-1)         # (batch * max_blocks,)
    k_all = k_cache[bt].view(batch, max_seq_len, num_kv_heads, head_dim).to(q.dtype)
    v_all = v_cache[bt].view(batch, max_seq_len, num_kv_heads, head_dim).to(q.dtype)

    # GQA: repeat K/V heads to match Q heads
    if num_kv_groups > 1:
        k_all = k_all.repeat_interleave(num_kv_groups, dim=2)
        v_all = v_all.repeat_interleave(num_kv_groups, dim=2)

    # Attention scores: (batch, num_heads, 1, max_seq_len)
    q_exp = q.unsqueeze(2)                                    # (batch, num_heads, 1, head_dim)
    k_t = k_all.permute(0, 2, 3, 1)                          # (batch, num_heads, head_dim, max_seq_len)
    scores = torch.matmul(q_exp, k_t) * softmax_scale

    # Mask padding positions (positions >= cache_seqlens) with -inf
    pos = torch.arange(max_seq_len, device=q.device)          # (max_seq_len,)
    pad_mask = pos.unsqueeze(0) >= cache_seqlens.unsqueeze(1) # (batch, max_seq_len)
    scores = scores.masked_fill(pad_mask[:, None, None, :], float('-inf'))

    attn = F.softmax(scores, dim=-1)
    v_t = v_all.permute(0, 2, 1, 3)                          # (batch, num_heads, max_seq_len, head_dim)
    return torch.matmul(attn, v_t).squeeze(2)                 # (batch, num_heads, head_dim)


def naive_paged_decode_attention(
    q: torch.Tensor,              # (batch, num_heads, head_dim)
    k_cache: torch.Tensor,        # (num_blocks, block_size, num_kv_heads, head_dim)
    v_cache: torch.Tensor,        # same
    block_table: torch.Tensor,    # (batch, max_blocks_per_seq) int32
    cache_seqlens: torch.Tensor,  # (batch,) int32
    num_kv_groups: int = 1,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Paged decode attention that reads K/V directly from block table.

    For each batch element, gathers K/V from paged blocks and computes
    single-query attention. No explicit gather into contiguous tensor.

    Args:
        q: query for each batch element (batch, num_heads, head_dim)
        k_cache: paged K cache (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: paged V cache (same shape)
        block_table: maps (batch, block_idx) → physical block ID (int32)
        cache_seqlens: sequence length per batch element (int32)
        num_kv_groups: GQA groups (num_heads // num_kv_heads)
        softmax_scale: attention scale factor

    Returns:
        output: (batch, num_heads, head_dim)
    """
    # During CUDA graph capture use the fully vectorized path (no .item(), no Python loops).
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        return _tensor_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens, num_kv_groups,
            softmax_scale or 1.0 / math.sqrt(q.shape[-1]),
        )

    batch = q.shape[0]
    bs = k_cache.shape[1]  # block_size

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    compute_dtype = q.dtype
    outputs = []

    for b in range(batch):
        seq_len = cache_seqlens[b].item()
        if seq_len == 0:
            outputs.append(torch.zeros_like(q[b]))
            continue

        # Gather K/V from blocks
        num_blocks_needed = (seq_len + bs - 1) // bs
        k_parts = []
        v_parts = []
        for blk_idx in range(num_blocks_needed):
            block_id = block_table[b, blk_idx].item()
            if block_id < 0:
                break
            # Last block may be partially filled
            if blk_idx == num_blocks_needed - 1:
                remainder = seq_len - blk_idx * bs
                k_parts.append(k_cache[block_id, :remainder].to(compute_dtype))
                v_parts.append(v_cache[block_id, :remainder].to(compute_dtype))
            else:
                k_parts.append(k_cache[block_id].to(compute_dtype))
                v_parts.append(v_cache[block_id].to(compute_dtype))

        k_seq = torch.cat(k_parts, dim=0)  # (seq_len, num_kv_heads, head_dim)
        v_seq = torch.cat(v_parts, dim=0)

        # GQA expand
        if num_kv_groups > 1:
            k_seq = k_seq.repeat_interleave(num_kv_groups, dim=1)
            v_seq = v_seq.repeat_interleave(num_kv_groups, dim=1)

        # Single-query attention: q_b (num_heads, 1, head_dim) @ k_seq (num_heads, seq_len, head_dim)
        q_b = q[b].unsqueeze(1)           # (num_heads, 1, head_dim)
        k_t = k_seq.transpose(0, 1)       # (num_heads, seq_len, head_dim)
        v_t = v_seq.transpose(0, 1)

        attn = torch.bmm(q_b, k_t.transpose(1, 2)) * softmax_scale  # (num_heads, 1, seq_len)
        attn = F.softmax(attn, dim=-1)
        out_b = torch.bmm(attn, v_t).squeeze(1)  # (num_heads, head_dim)
        outputs.append(out_b)

    return torch.stack(outputs)  # (batch, num_heads, head_dim)
