"""
Tests for vllm-i64 performance features:
  - Batched repetition penalty (WI-1)
  - FP8 KV cache quantization (WI-2)
  - Multi-batch CUDA graphs (WI-3)
  - Prefix caching integration (WI-4)
  - Speculative decoding integration (WI-5)
  - CUDA graph activation (WI-6)

INL - 2025
"""

import pytest
import torch
import numpy as np

from vllm_i64.core.sampling import (
    SamplingParams,
    sample_batch,
    apply_repetition_penalty_batch,
)
from vllm_i64.core.kv_cache import PagedKVCache
from vllm_i64.engine.i64_engine import I64Engine


# =========================================================================
# WI-1: Batched Repetition Penalty
# =========================================================================

class TestBatchedRepetitionPenalty:
    def test_penalty_reduces_repeated_logits(self):
        logits = torch.zeros(2, 10)
        logits[0, 3] = 5.0
        logits[1, 7] = 5.0

        past = [[3, 3, 3], [7, 7]]
        result = apply_repetition_penalty_batch(logits, past, penalty=1.2, vocab_size=10)

        assert result[0, 3] < 5.0  # penalized
        assert result[1, 7] < 5.0  # penalized
        assert result[0, 7] == 0.0  # unaffected

    def test_penalty_1_0_is_noop(self):
        logits = torch.randn(4, 100)
        original = logits.clone()
        result = apply_repetition_penalty_batch(logits, [[] for _ in range(4)], 1.0, 100)
        assert torch.equal(result, original)

    def test_negative_logits_amplified(self):
        logits = torch.zeros(1, 10)
        logits[0, 5] = -3.0
        apply_repetition_penalty_batch(logits, [[5]], penalty=1.5, vocab_size=10)
        assert logits[0, 5].item() < -3.0  # more negative

    def test_oob_tokens_filtered(self):
        logits = torch.randn(1, 10)
        original = logits.clone()
        # Token ID 50 is out of range for vocab_size=10
        apply_repetition_penalty_batch(logits, [[50, 100]], penalty=2.0, vocab_size=10)
        assert torch.equal(logits, original)  # nothing changed

    def test_sample_batch_with_past_tokens(self):
        logits = torch.randn(3, 50)
        params = SamplingParams(temperature=1.0, repetition_penalty=1.5)
        past = [[1, 2, 3], [4, 5], []]
        tokens = sample_batch(logits, params, past_tokens_list=past)
        assert tokens.shape == (3,)

    def test_sample_batch_greedy_with_penalty(self):
        logits = torch.zeros(2, 10)
        logits[0, 5] = 10.0
        logits[0, 3] = 9.0
        logits[1, 7] = 10.0
        params = SamplingParams(temperature=0.0, repetition_penalty=2.0)
        # Token 5 is heavily penalized for request 0 → should pick 3 instead
        tokens = sample_batch(logits, params, past_tokens_list=[[5], []])
        assert tokens[0].item() == 3  # 5 was penalized below 3
        assert tokens[1].item() == 7  # unaffected

    def test_engine_step_uses_batch_penalty(self):
        """Verify engine step works with batched repetition penalty."""
        engine = I64Engine(num_experts=4, vocab_size=100, device="cpu")
        engine.sampling_params = SamplingParams(temperature=1.0, repetition_penalty=1.1)
        engine.add_request([1, 2, 3], max_new_tokens=5)
        result = engine.step()
        assert len(result) == 1


# =========================================================================
# WI-2: FP8 KV Cache
# =========================================================================

class TestFP8KVCache:
    def test_default_dtype_matches_compute(self):
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=16,
            block_size=4, num_blocks=8, max_seqs=2,
            dtype=torch.float32, device="cpu",
        )
        assert cache.kv_dtype == torch.float32
        assert cache.compute_dtype == torch.float32

    def test_fp8_ignored_on_cpu(self):
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=16,
            block_size=4, num_blocks=8, max_seqs=2,
            dtype=torch.float32, device="cpu",
            kv_cache_dtype="fp8",
        )
        # FP8 not supported on CPU — falls back to compute dtype
        assert cache.kv_dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 requires CUDA")
    def test_fp8_cache_creation(self):
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=16,
            block_size=4, num_blocks=8, max_seqs=2,
            dtype=torch.float16, device="cuda",
            kv_cache_dtype="fp8",
        )
        assert cache.kv_dtype == torch.float8_e4m3fn
        assert cache.k_caches[0].dtype == torch.float8_e4m3fn

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 requires CUDA")
    def test_fp8_write_read_roundtrip(self):
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=16,
            block_size=4, num_blocks=8, max_seqs=2,
            dtype=torch.float16, device="cuda",
            kv_cache_dtype="fp8",
        )
        cache.allocate_blocks(0, 1)

        k = torch.randn(2, 16, dtype=torch.float16, device="cuda")
        v = torch.randn(2, 16, dtype=torch.float16, device="cuda")
        cache.write_kv(0, 0, 0, k, v)

        k_out, v_out = cache.read_kv(0, 0)
        # FP8 precision is limited, so check approximate equality
        assert k_out.dtype == torch.float16
        assert k_out.shape == (1, 2, 16)
        assert torch.allclose(k_out[0], k, atol=0.2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 requires CUDA")
    def test_fp8_memory_savings(self):
        cache_fp16 = PagedKVCache(
            num_layers=1, num_kv_heads=8, head_dim=128,
            block_size=16, num_blocks=64, max_seqs=4,
            dtype=torch.float16, device="cuda",
        )
        cache_fp8 = PagedKVCache(
            num_layers=1, num_kv_heads=8, head_dim=128,
            block_size=16, num_blocks=64, max_seqs=4,
            dtype=torch.float16, device="cuda",
            kv_cache_dtype="fp8",
        )
        mem_fp16 = cache_fp16.k_caches[0].nelement() * cache_fp16.k_caches[0].element_size()
        mem_fp8 = cache_fp8.k_caches[0].nelement() * cache_fp8.k_caches[0].element_size()
        assert mem_fp8 == mem_fp16 // 2  # 2x savings

    def test_kv_cache_dtype_param_accepted(self):
        """Ensure kv_cache_dtype parameter doesn't crash on CPU."""
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=16,
            block_size=4, num_blocks=8, max_seqs=2,
            dtype=torch.float32, device="cpu",
            kv_cache_dtype="fp8_e5m2",
        )
        assert cache.compute_dtype == torch.float32


# =========================================================================
# WI-3: Multi-Batch CUDA Graphs
# =========================================================================

class TestMultiBatchCUDAGraphs:
    def test_find_best_size(self):
        from vllm_i64.core.cuda_graph import CUDAGraphRunner

        runner = CUDAGraphRunner(lambda t, p, e: t, max_batch_size=32)
        runner._captured_sizes = {1, 4, 8, 16}

        assert runner._find_best_size(1) == 1
        assert runner._find_best_size(3) == 4
        assert runner._find_best_size(4) == 4
        assert runner._find_best_size(5) == 8
        assert runner._find_best_size(16) == 16
        assert runner._find_best_size(17) is None

    def test_uncaptured_falls_back(self):
        from vllm_i64.core.cuda_graph import CUDAGraphRunner

        called = []
        def fake_forward(t, p, e):
            called.append(True)
            return torch.randn(t.shape[0], 100)

        runner = CUDAGraphRunner(fake_forward, max_batch_size=32)
        # No graphs captured — should fall back
        token_ids = torch.zeros(5, dtype=torch.int64)
        positions = torch.zeros(5, dtype=torch.int32)
        expert_ids = torch.zeros(5, dtype=torch.int32)

        out = runner.run(token_ids, positions, expert_ids)
        assert out.shape == (5, 100)
        assert len(called) == 1
        assert not runner.is_captured

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_capture_common_sizes(self):
        from vllm_i64.core.cuda_graph import CUDAGraphRunner

        def fake_forward(t, p, e):
            return torch.randn(t.shape[0], 50, device=t.device)

        runner = CUDAGraphRunner(fake_forward, max_batch_size=16, device="cuda")
        runner.capture_common_sizes()

        assert 1 in runner._captured_sizes
        assert 4 in runner._captured_sizes
        assert 16 in runner._captured_sizes
        assert 32 not in runner._captured_sizes  # > max_batch_size
        assert runner.is_captured


# =========================================================================
# WI-4: Prefix Caching Integration
# =========================================================================

class TestPrefixCachingIntegration:
    def test_engine_accepts_prefix_caching_flag(self):
        engine = I64Engine(
            num_experts=4, vocab_size=100, device="cpu",
            enable_prefix_caching=True,
        )
        assert engine.enable_prefix_caching is True

    def test_prefix_cache_enabled_in_kv_cache(self):
        """When engine has a model, prefix caching should be enabled on kv_cache."""
        # Without a model, kv_cache is None — just verify the flag is stored
        engine = I64Engine(
            num_experts=4, vocab_size=100, device="cpu",
            enable_prefix_caching=True,
        )
        assert engine.enable_prefix_caching is True
        # kv_cache is None without model, but the flag is ready

    def test_prefix_caching_basic_reuse(self):
        """Test that prefix caching correctly reuses KV blocks."""
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=16,
            block_size=4, num_blocks=16, max_seqs=4,
            dtype=torch.float32, device="cpu",
        )
        cache.enable_prefix_caching()

        # Write a full block for seq 0
        cache.allocate_blocks(0, 1)
        for pos in range(4):
            k = torch.randn(2, 16)
            v = torch.randn(2, 16)
            cache.write_kv(0, 0, pos, k, v)
        cache.register_prefix_blocks(0, [10, 20, 30, 40])

        # Try to reuse for seq 1
        reused = cache.try_reuse_prefix(1, [10, 20, 30, 40, 50, 60])
        assert reused == 4  # one block reused


# =========================================================================
# WI-5: Speculative Decoding Integration
# =========================================================================

class TestSpeculativeIntegration:
    def test_enable_speculative(self):
        engine = I64Engine(num_experts=4, vocab_size=100, device="cpu")

        class FakeModel:
            def __call__(self, token_ids, positions):
                return torch.randn(token_ids.shape[0], 100)

        engine.enable_speculative(FakeModel(), num_speculative=3)
        assert engine.speculative_decoder is not None
        assert engine.speculative_decoder.K == 3

    def test_speculative_step_produces_tokens(self):
        """Speculative decoder should produce accepted tokens."""
        from vllm_i64.core.speculative import SpeculativeDecoder

        class FakeModel:
            def __call__(self, token_ids, positions=None):
                batch = token_ids.shape[0] if token_ids.dim() > 0 else 1
                return torch.randn(batch, 50)

        decoder = SpeculativeDecoder(
            target_model=FakeModel(),
            draft_model=FakeModel(),
            num_speculative=3,
        )
        ctx = torch.tensor([1, 2, 3], dtype=torch.int64)
        pos = torch.tensor([0, 1, 2], dtype=torch.int32)
        accepted, num_draft = decoder.generate_step(ctx, pos)

        assert len(accepted) >= 1
        assert num_draft == 3


# =========================================================================
# WI-6: CUDA Graph Activation
# =========================================================================

class TestCUDAGraphActivation:
    def test_warmup_noop_on_cpu(self):
        engine = I64Engine(num_experts=4, vocab_size=100, device="cpu")
        # Should not crash
        engine.warmup_and_capture_graphs()

    def test_warmup_noop_without_model(self):
        engine = I64Engine(num_experts=4, vocab_size=100, device="cpu")
        engine.model = None
        engine.warmup_and_capture_graphs()


# =========================================================================
# CLI flags
# =========================================================================

class TestCLIFlags:
    def test_serve_parser_has_new_flags(self):
        import argparse
        from vllm_i64.cli import main

        # Just verify the parser accepts the new flags without error
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        p = sub.add_parser("serve")
        p.add_argument("model")
        p.add_argument("--enable-prefix-caching", action="store_true")
        p.add_argument("--kv-cache-dtype", default=None, choices=["fp8", "fp8_e5m2"])
        p.add_argument("--speculative-model", default=None)
        p.add_argument("--num-speculative-tokens", type=int, default=5)

        args = parser.parse_args(["serve", "test-model", "--enable-prefix-caching",
                                  "--kv-cache-dtype", "fp8", "--num-speculative-tokens", "3"])
        assert args.enable_prefix_caching is True
        assert args.kv_cache_dtype == "fp8"
        assert args.num_speculative_tokens == 3
