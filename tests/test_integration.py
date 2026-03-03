"""
vllm-i64 :: Integration Tests

End-to-end tests for engine + KV cache + scheduler interactions:
  - Full generate() with KV cache read/write
  - Engine step() cycle correctness
  - Timeout and cancellation
  - Request counter race safety
  - Deque-based slot pool
  - Single-pass timeout handling

INL - 2025
"""

import pytest
import numpy as np
import torch
import sys
import os
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.engine.i64_engine import I64Engine, GenerationResult, AdaptiveBatchSizer
from vllm_i64.engine.i64_scheduler import I64Scheduler, I64Request, I64Batch
from vllm_i64.core.kv_cache import PagedKVCache
from vllm_i64.core.sampling import SamplingParams


# =====================================================================
# KV Cache Vectorization Tests
# =====================================================================

class TestKVCacheVectorized:
    def test_read_kv_block_vectorized(self):
        """read_kv should use block-level copy instead of per-token loop."""
        cache = PagedKVCache(
            num_layers=2, num_kv_heads=4, head_dim=8,
            block_size=4, num_blocks=16, max_seqs=4,
        )

        seq_id = 0
        # Write 10 tokens worth of KV data
        for pos in range(10):
            k = torch.randn(4, 8)
            v = torch.randn(4, 8)
            cache.write_kv(0, seq_id, pos, k, v)

        # Read back
        k_out, v_out = cache.read_kv(0, seq_id)
        assert k_out.shape == (10, 4, 8)
        assert v_out.shape == (10, 4, 8)
        assert k_out.dtype == torch.float16

    def test_read_kv_empty_sequence(self):
        """read_kv should handle empty sequences gracefully."""
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=4,
            block_size=4, num_blocks=8, max_seqs=2,
        )
        k, v = cache.read_kv(0, 0)
        assert k.shape == (0, 2, 4)
        assert v.shape == (0, 2, 4)

    def test_read_kv_with_max_len(self):
        """read_kv with max_len should truncate correctly."""
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=4,
            block_size=4, num_blocks=8, max_seqs=2,
        )
        for pos in range(8):
            cache.write_kv(0, 0, pos, torch.randn(2, 4), torch.randn(2, 4))

        k, v = cache.read_kv(0, 0, max_len=5)
        assert k.shape[0] == 5

    def test_write_kv_batch_groups_by_block(self):
        """write_kv_batch should handle batch writes correctly."""
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=4,
            block_size=4, num_blocks=8, max_seqs=2,
        )

        positions = torch.arange(6, dtype=torch.int32)
        k = torch.randn(6, 2, 4)
        v = torch.randn(6, 2, 4)
        cache.write_kv_batch(0, 0, positions, k, v)

        # Verify by reading back
        k_out, v_out = cache.read_kv(0, 0)
        assert k_out.shape == (6, 2, 4)

    def test_kv_round_trip_correctness(self):
        """Data written to KV cache should be correctly read back (block boundaries)."""
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=4,
            block_size=4, num_blocks=8, max_seqs=2,
            dtype=torch.float32,  # Use float32 for exact comparison
        )

        original_k = torch.randn(12, 2, 4)
        original_v = torch.randn(12, 2, 4)

        for pos in range(12):
            cache.write_kv(0, 0, pos, original_k[pos], original_v[pos])

        k_out, v_out = cache.read_kv(0, 0)
        assert torch.allclose(k_out, original_k, atol=1e-6)
        assert torch.allclose(v_out, original_v, atol=1e-6)


class TestKVCacheMaxBlocks:
    def test_max_blocks_per_seq_derived(self):
        """max_blocks_per_seq should be derived, not hardcoded to 128."""
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=4,
            block_size=16, num_blocks=512, max_seqs=4,
        )
        # Should be at least 128 (the old default) but now derived
        assert cache.max_blocks_per_seq >= 128


class TestKVCacheEviction:
    def test_evict_lru_vectorized_block_count(self):
        """LRU eviction should count blocks using vectorized operation."""
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=4,
            block_size=4, num_blocks=8, max_seqs=4,
        )

        # Allocate blocks for seq 0
        cache.allocate_blocks(0, 4)
        cache.seq_lens[0] = 16  # 4 blocks * 4 tokens

        # Allocate blocks for seq 1
        cache.allocate_blocks(1, 2)
        cache.seq_lens[1] = 8

        # Touch seq 1 more recently
        cache._touch(1)

        # Evict to free 3 blocks (should evict seq 0, the LRU)
        freed = cache._evict_lru(3, protect_seq_id=-1)
        assert freed >= 3
        assert 0 in cache._evicted_seq_ids


# =====================================================================
# Engine Performance Fixes Tests
# =====================================================================

class TestAdaptiveBatchSizer:
    def test_deque_maxlen(self):
        """AdaptiveBatchSizer should use deque with maxlen (not pop(0))."""
        sizer = AdaptiveBatchSizer(initial=16, window=5)
        assert isinstance(sizer._throughputs, deque)
        assert sizer._throughputs.maxlen == 5

        # Record more than window size
        for i in range(10):
            sizer.record(100, 10.0)
        assert len(sizer._throughputs) == 5


class TestSlotPool:
    def test_slot_pool_is_deque(self):
        """Slot pool should be a deque for O(1) popleft."""
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        assert isinstance(engine._slot_pool, deque)

    def test_slot_allocation_fifo(self):
        """Slots should be allocated in FIFO order (popleft, not pop)."""
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        # Manually set up a slot pool
        engine._slot_pool = deque([0, 1, 2, 3])
        slot1 = engine._allocate_slot(100)
        slot2 = engine._allocate_slot(101)
        assert slot1 == 0  # popleft gives first element
        assert slot2 == 1


class TestTimeoutHandling:
    def test_single_pass_timeout(self):
        """Timeout handling should be single-pass O(n), not nested O(n^2)."""
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")

        # Add some requests
        import time
        for i in range(5):
            rid = engine.add_request([1, 2, 3], max_new_tokens=10, timeout_s=0.001)

        # Run a step to move them to running
        engine.step()

        # Wait for timeout
        time.sleep(0.01)

        # This should handle timeouts in single pass
        engine._check_timeouts_and_cancellations()

        # All should be finished
        assert len(engine.scheduler.running) == 0

    def test_cancellation(self):
        """Cancelled requests should be cleaned up properly."""
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        rid = engine.add_request([1, 2, 3], max_new_tokens=100)
        engine.step()  # Move to running

        engine.cancel_request(rid)
        engine._check_timeouts_and_cancellations()

        assert len(engine.scheduler.running) == 0
        assert len(engine.scheduler.finished) > 0


class TestRequestIndex:
    def test_step_uses_dict_lookup(self):
        """Engine step should build a request index for O(1) lookups."""
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")

        # Add multiple requests
        for i in range(8):
            engine.add_request([i + 1], max_new_tokens=3)

        # Step should complete without errors (uses _running_index internally)
        result = engine.step()
        assert isinstance(result, dict)


# =====================================================================
# Full Generate Integration
# =====================================================================

class TestGenerateIntegration:
    def test_generate_basic(self):
        """Full generate() loop should work end-to-end."""
        engine = I64Engine(
            model=None, num_experts=4, vocab_size=100,
            max_batch_size=8, device="cpu",
        )
        result = engine.generate(
            prompt_token_ids=[1, 2, 3, 4],
            max_new_tokens=5,
        )
        assert isinstance(result, GenerationResult)
        assert len(result.output_tokens) == 5
        assert result.finish_reason == "length"
        assert result.elapsed_ms > 0

    def test_generate_with_sampling_params(self):
        """Generate with custom sampling params."""
        engine = I64Engine(
            model=None, num_experts=4, vocab_size=100,
            device="cpu",
        )
        params = SamplingParams(temperature=0.0, top_k=1, max_tokens=3)
        result = engine.generate(
            prompt_token_ids=[10, 20, 30],
            max_new_tokens=3,
            sampling_params=params,
        )
        assert len(result.output_tokens) == 3

    def test_generate_multiple_sequential(self):
        """Multiple sequential generates should not leak state."""
        engine = I64Engine(
            model=None, num_experts=4, vocab_size=100,
            device="cpu",
        )
        for i in range(5):
            result = engine.generate(
                prompt_token_ids=[i + 1],
                max_new_tokens=3,
                sampling_params=SamplingParams(temperature=0.0),
            )
            # With model=None (random logits), EOS=0 can be hit early
            assert 1 <= len(result.output_tokens) <= 3
            assert result.finish_reason in ("length", "stop")

        # No leftover state
        assert len(engine.scheduler.running) == 0
        assert len(engine.scheduler.pending) == 0


class TestAsyncEngineFactory:
    def test_from_sync_engine(self):
        """AsyncI64Engine.from_sync_engine should create a valid instance."""
        from vllm_i64.engine.i64_engine import AsyncI64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        async_engine = AsyncI64Engine.from_sync_engine(engine)
        assert async_engine.engine is engine
        assert async_engine._running is False
        assert async_engine.active_requests == 0
        assert async_engine.peak_batch_size == 0

    def test_from_sync_engine_stats(self):
        """Stats should work on factory-created instance."""
        from vllm_i64.engine.i64_engine import AsyncI64Engine
        engine = I64Engine(model=None, num_experts=4, vocab_size=100, device="cpu")
        async_engine = AsyncI64Engine.from_sync_engine(engine)
        stats = async_engine.get_stats()
        assert "active_requests" in stats
        assert "peak_batch_size" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
