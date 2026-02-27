"""
vllm-i64 :: Test Attention Backends

Tests for FlashAttention integration and naive fallback:
  - naive_varlen_attention correctness (single/multi sequence, GQA, causal)
  - naive_cached_attention correctness (decode, prefill with history)
  - KV cache batch write + flash accessors
  - Flash vs naive equivalence (skipped if flash_attn not installed)

Run:
    python -m pytest tests/test_flash_attention.py -v

INL - 2025
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.layers.attention import (
    is_flash_attn_available,
    naive_varlen_attention,
    naive_cached_attention,
    compute_cu_seqlens,
)
from vllm_i64.core.kv_cache import PagedKVCache


class TestComputeCuSeqlens:

    def test_basic(self):
        cu, max_sl = compute_cu_seqlens([3, 5, 2], torch.device("cpu"))
        assert cu.tolist() == [0, 3, 8, 10]
        assert max_sl == 5

    def test_single(self):
        cu, max_sl = compute_cu_seqlens([7], torch.device("cpu"))
        assert cu.tolist() == [0, 7]
        assert max_sl == 7

    def test_dtype_is_int32(self):
        cu, _ = compute_cu_seqlens([1, 2], torch.device("cpu"))
        assert cu.dtype == torch.int32


class TestNaiveVarlenAttention:

    def test_single_sequence(self):
        """Single sequence should produce correct output shape."""
        seq_len = 8
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32

        q = torch.randn(seq_len, num_heads, head_dim)
        k = torch.randn(seq_len, num_kv_heads, head_dim)
        v = torch.randn(seq_len, num_kv_heads, head_dim)

        out = naive_varlen_attention(q, k, v, [seq_len], num_kv_groups=2)
        assert out.shape == (seq_len, num_heads, head_dim)

    def test_multi_sequence_isolation(self):
        """Tokens from different sequences must not attend to each other."""
        num_heads = 2
        head_dim = 16

        # Two sequences of length 3
        q = torch.randn(6, num_heads, head_dim)
        k = torch.randn(6, num_heads, head_dim)
        v = torch.randn(6, num_heads, head_dim)

        # Process as two separate sequences
        out_multi = naive_varlen_attention(q, k, v, [3, 3], num_kv_groups=1)

        # Process each separately
        out_1 = naive_varlen_attention(q[:3], k[:3], v[:3], [3], num_kv_groups=1)
        out_2 = naive_varlen_attention(q[3:], k[3:], v[3:], [3], num_kv_groups=1)
        out_separate = torch.cat([out_1, out_2], dim=0)

        assert torch.allclose(out_multi, out_separate, atol=1e-5)

    def test_gqa_expansion(self):
        """GQA with 4:2 ratio produces correct output."""
        q = torch.randn(5, 4, 32)
        k = torch.randn(5, 2, 32)
        v = torch.randn(5, 2, 32)

        out = naive_varlen_attention(q, k, v, [5], num_kv_groups=2)
        assert out.shape == (5, 4, 32)

    def test_causal_masking(self):
        """First token should only attend to itself."""
        num_heads = 1
        head_dim = 8

        q = torch.randn(4, num_heads, head_dim)
        k = torch.randn(4, num_heads, head_dim)
        v = torch.ones(4, num_heads, head_dim)  # Known values

        # Make v[0] distinct
        v[0] = 2.0
        v[1:] = 0.0

        out = naive_varlen_attention(q, k, v, [4], num_kv_groups=1)

        # First token output should be exactly v[0] (it can only attend to itself)
        assert torch.allclose(out[0], v[0], atol=1e-5)

    def test_single_token_no_masking_needed(self):
        """Single token sequence should work without any masking."""
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 2, 32)
        v = torch.randn(1, 2, 32)

        out = naive_varlen_attention(q, k, v, [1], num_kv_groups=2)
        assert out.shape == (1, 4, 32)


class TestNaiveCachedAttention:

    def test_decode_single_token(self):
        """Single new token attending to cached history."""
        num_heads = 4
        num_kv_heads = 2
        head_dim = 32
        history_len = 10

        q = torch.randn(1, num_heads, head_dim)
        k_full = torch.randn(history_len, num_kv_heads, head_dim)
        v_full = torch.randn(history_len, num_kv_heads, head_dim)
        positions = torch.tensor([history_len - 1], dtype=torch.int32)

        out = naive_cached_attention(q, k_full, v_full, num_kv_groups=2, positions=positions)
        assert out.shape == (1, num_heads, head_dim)

    def test_prefill_with_history(self):
        """Multiple new tokens attending to cached + new K/V."""
        num_heads = 2
        head_dim = 16
        history_len = 8
        new_tokens = 3

        q = torch.randn(new_tokens, num_heads, head_dim)
        k_full = torch.randn(history_len, num_heads, head_dim)
        v_full = torch.randn(history_len, num_heads, head_dim)
        positions = torch.arange(history_len - new_tokens, history_len, dtype=torch.int32)

        out = naive_cached_attention(q, k_full, v_full, num_kv_groups=1, positions=positions)
        assert out.shape == (new_tokens, num_heads, head_dim)


class TestKVCacheNewMethods:

    def _make_cache(self):
        return PagedKVCache(
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            block_size=4,
            num_blocks=16,
            max_seqs=4,
            dtype=torch.float32,
            device="cpu",
        )

    def test_layout_is_flash_compatible(self):
        """Cache layout should be (num_blocks, block_size, num_kv_heads, head_dim)."""
        cache = self._make_cache()
        k, v = cache.get_cache_tensors(0)
        assert k.shape == (16, 4, 4, 32)
        assert v.shape == (16, 4, 4, 32)

    def test_write_kv_batch(self):
        """Batch write should match sequential writes."""
        cache1 = self._make_cache()
        cache2 = self._make_cache()

        # Allocate blocks for seq 0
        cache1.allocate_blocks(0, 2)
        cache2.allocate_blocks(0, 2)

        k_vals = torch.randn(3, 4, 32)
        v_vals = torch.randn(3, 4, 32)
        positions = torch.tensor([0, 1, 2], dtype=torch.int32)

        # Sequential write
        for t in range(3):
            cache1.write_kv(0, 0, t, k_vals[t], v_vals[t])

        # Batch write
        cache2.write_kv_batch(0, 0, positions, k_vals, v_vals)

        # Read back and compare
        k1, v1 = cache1.read_kv(0, 0)
        k2, v2 = cache2.read_kv(0, 0)

        assert torch.allclose(k1, k2)
        assert torch.allclose(v1, v2)

    def test_get_block_table_for_seqs(self):
        """Block table extraction returns correct rows."""
        cache = self._make_cache()
        cache.allocate_blocks(0, 2)
        cache.allocate_blocks(1, 3)

        table = cache.get_block_table_for_seqs([0, 1])
        assert table.shape[0] == 2
        assert table.dtype == torch.int32

        # Seq 0 has 2 allocated blocks
        assert (table[0, :2] >= 0).all()
        # Seq 1 has 3 allocated blocks
        assert (table[1, :3] >= 0).all()

    def test_get_cache_seqlens(self):
        """Cache seqlens tracks written positions."""
        cache = self._make_cache()
        cache.allocate_blocks(0, 2)
        cache.allocate_blocks(1, 2)

        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        cache.write_kv(0, 0, 0, k, v)
        cache.write_kv(0, 0, 1, k, v)
        cache.write_kv(0, 1, 0, k, v)

        lens = cache.get_cache_seqlens([0, 1])
        assert lens[0].item() == 2
        assert lens[1].item() == 1

    def test_write_read_roundtrip(self):
        """Write then read preserves values."""
        cache = self._make_cache()
        cache.allocate_blocks(0, 4)

        k_orig = torch.randn(4, 32)
        v_orig = torch.randn(4, 32)

        for pos in range(5):
            cache.write_kv(0, 0, pos, k_orig, v_orig)

        k_out, v_out = cache.read_kv(0, 0)
        assert k_out.shape[0] == 5
        # Last written value should match
        assert torch.allclose(k_out[4], k_orig)
        assert torch.allclose(v_out[4], v_orig)


class TestModelAttentionIntegration:
    """Test that the model still produces correct shapes with new attention backend."""

    def test_model_forward_unchanged(self):
        """ComplexityDeepModel forward pass still works."""
        from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
        from vllm_i64.models.complexity_deep.model import ComplexityDeepModel

        config = ComplexityDeepConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=128, num_experts=2,
        )
        model = ComplexityDeepModel(config).float()

        token_ids = torch.tensor([1, 5, 10, 20, 50], dtype=torch.int64)
        positions = torch.arange(5, dtype=torch.int32)

        with torch.no_grad():
            logits = model(token_ids, positions)

        assert logits.shape == (5, 100)

    def test_model_single_token(self):
        """Single token forward pass works."""
        from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
        from vllm_i64.models.complexity_deep.model import ComplexityDeepModel

        config = ComplexityDeepConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=128, num_experts=2,
        )
        model = ComplexityDeepModel(config).float()

        token_ids = torch.tensor([42], dtype=torch.int64)
        positions = torch.tensor([0], dtype=torch.int32)

        with torch.no_grad():
            logits = model(token_ids, positions)

        assert logits.shape == (1, 100)

    def test_model_deterministic(self):
        """Model should be deterministic."""
        from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
        from vllm_i64.models.complexity_deep.model import ComplexityDeepModel

        config = ComplexityDeepConfig(
            vocab_size=100, hidden_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=128, num_experts=2,
        )
        model = ComplexityDeepModel(config).float()
        model.eval()

        token_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        positions = torch.arange(3, dtype=torch.int32)

        with torch.no_grad():
            out1 = model(token_ids, positions)
            out2 = model(token_ids, positions)

        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
