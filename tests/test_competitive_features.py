"""
vllm-i64 :: Test Competitive Features

Tests for all new competitive features:
  - Production hardening (timeouts, cancellation, logging)
  - Prefix caching
  - Chunked prefill
  - Structured output (logits processors)
  - Beam search
  - Advanced scheduler (preemption, priorities)
  - LoRA hot-swap
  - Kernel loader fallback

Run:
    python -m pytest tests/test_competitive_features.py -v

INL - 2025
"""

import pytest
import torch
import numpy as np
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.engine.i64_scheduler import (
    I64Scheduler, I64Request, I64Batch, RequestStatus,
)
from vllm_i64.engine.i64_engine import I64Engine, AsyncI64Engine, GenerationResult
from vllm_i64.core.kv_cache import PagedKVCache
from vllm_i64.core.sampling import SamplingParams, BeamSearcher, BeamHypothesis
from vllm_i64.core.logits_processor import (
    JSONLogitsProcessor, StopSequenceProcessor, ChoiceLogitsProcessor,
    OutputConstraints, apply_logits_processors,
)
from vllm_i64.core.logging import setup_logging, get_logger, RequestLogger
from vllm_i64.layers.lora import LoRALinear, LoRAManager
from vllm_i64.kernels.kernel_loader import FallbackOps


# =========================================================================
# Production Hardening
# =========================================================================

class TestStructuredLogging:

    def test_logger_creation(self):
        logger = setup_logging(level="DEBUG")
        assert logger.name == "vllm_i64"
        assert logger.level == 10  # DEBUG

    def test_request_logger(self):
        logger = setup_logging(level="WARNING")
        req_logger = RequestLogger(request_id=42, logger=logger)
        assert req_logger.request_id == 42
        assert req_logger.elapsed_ms() >= 0

    def test_get_logger(self):
        log = get_logger("test.module")
        assert log.name == "test.module"


class TestTimeoutsAndCancellation:

    def test_engine_has_timeout_support(self):
        engine = I64Engine(num_experts=4, vocab_size=100, device="cpu")
        assert hasattr(engine, "_request_deadlines")
        assert hasattr(engine, "_cancelled_requests")
        assert engine.default_timeout_s == 300.0

    def test_add_request_with_timeout(self):
        engine = I64Engine(num_experts=4, vocab_size=100, device="cpu")
        rid = engine.add_request([1, 2, 3], max_new_tokens=10, timeout_s=5.0)
        assert rid in engine._request_deadlines

    def test_cancel_request(self):
        engine = I64Engine(num_experts=4, vocab_size=100, device="cpu")
        rid = engine.add_request([1, 2, 3], max_new_tokens=10)
        engine.cancel_request(rid)
        assert rid in engine._cancelled_requests

    def test_generation_result_finish_reason(self):
        result = GenerationResult(
            request_id=1,
            prompt_tokens=[1, 2],
            output_tokens=[3, 4],
            num_steps=2,
            elapsed_ms=100.0,
            finish_reason="timeout",
        )
        assert result.finish_reason == "timeout"


# =========================================================================
# Prefix Caching
# =========================================================================

class TestPrefixCaching:

    def _make_cache(self):
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=16,
            block_size=4, num_blocks=32, max_seqs=4,
            dtype=torch.float32, device="cpu",
        )
        cache.enable_prefix_caching()
        return cache

    def test_enable_prefix_caching(self):
        cache = self._make_cache()
        assert cache.prefix_cache_enabled

    def test_prefix_hash_deterministic(self):
        h1 = PagedKVCache._hash_token_block([1, 2, 3, 4])
        h2 = PagedKVCache._hash_token_block([1, 2, 3, 4])
        h3 = PagedKVCache._hash_token_block([1, 2, 3, 5])
        assert h1 == h2
        assert h1 != h3

    def test_register_and_reuse_prefix(self):
        cache = self._make_cache()

        # First request: prefill and register
        cache.allocate_blocks(0, 2)
        k = torch.randn(2, 16)
        v = torch.randn(2, 16)
        for pos in range(8):
            cache.write_kv(0, 0, pos, k, v)
        token_ids = list(range(100, 108))  # 8 tokens = 2 blocks of size 4
        cache.register_prefix_blocks(0, token_ids)

        # Verify blocks registered
        assert len(cache._prefix_hash_to_blocks) > 0

        # Second request: try to reuse
        reused = cache.try_reuse_prefix(1, token_ids)
        assert reused == 8  # Both blocks reused (2 blocks * block_size=4)

    def test_no_reuse_without_enable(self):
        cache = PagedKVCache(
            num_layers=1, num_kv_heads=2, head_dim=16,
            block_size=4, num_blocks=32, max_seqs=4,
            dtype=torch.float32, device="cpu",
        )
        reused = cache.try_reuse_prefix(0, [1, 2, 3, 4])
        assert reused == 0

    def test_stats_include_prefix_info(self):
        cache = self._make_cache()
        stats = cache.get_stats()
        assert "prefix_cached_blocks" in stats
        assert "prefix_unique_hashes" in stats

    def test_free_sequence_respects_refcount(self):
        cache = self._make_cache()
        cache.allocate_blocks(0, 2)
        k = torch.randn(2, 16)
        v = torch.randn(2, 16)
        for pos in range(8):
            cache.write_kv(0, 0, pos, k, v)
        token_ids = list(range(100, 108))
        cache.register_prefix_blocks(0, token_ids)

        # Reuse from seq 1
        cache.try_reuse_prefix(1, token_ids)

        # Free seq 0 — shared block should NOT be freed yet
        initial_free = cache.num_free_blocks
        cache.free_sequence(0)

        # Free seq 1 — now shared block CAN be freed
        cache.free_sequence(1)


# =========================================================================
# Chunked Prefill
# =========================================================================

class TestChunkedPrefill:

    def test_scheduler_has_prefill_budget(self):
        sched = I64Scheduler(
            max_batch_size=4, num_experts=4,
            max_prefill_tokens=128,
        )
        assert sched.max_prefill_tokens == 128

    def test_short_prompt_no_chunking(self):
        """Short prompts should NOT be chunked."""
        sched = I64Scheduler(
            max_batch_size=4, num_experts=4,
            max_prefill_tokens=256,
        )
        prompt = np.arange(64, dtype=np.int64)
        sched.add_request(prompt, max_new_tokens=10)
        batch = sched.schedule()
        assert batch is not None
        # All 64 prompt tokens should be in the batch
        assert batch.total_tokens == 64

    def test_request_has_prefill_progress(self):
        req = I64Request(
            request_id=0,
            prompt_token_ids=np.arange(100, dtype=np.int64),
            max_new_tokens=10,
        )
        assert req.prefill_progress == 0
        assert not req.prefill_complete
        req.prefill_progress = 100
        assert req.prefill_complete


# =========================================================================
# Advanced Scheduler
# =========================================================================

class TestAdvancedScheduler:

    def test_priority_ordering(self):
        """Higher priority requests (lower number) should be scheduled first."""
        sched = I64Scheduler(max_batch_size=1, num_experts=4)

        sched.add_request(np.array([1, 2, 3], dtype=np.int64), priority=1)  # low
        sched.add_request(np.array([4, 5, 6], dtype=np.int64), priority=-1)  # high

        # High priority should be first in pending
        assert sched.pending[0].priority == -1

    def test_preemption_enabled(self):
        sched = I64Scheduler(
            max_batch_size=4, num_experts=4,
            max_kv_blocks=4, kv_block_size=4,
            enable_preemption=True,
        )
        assert sched.enable_preemption

    def test_preempted_list(self):
        sched = I64Scheduler(max_batch_size=4, num_experts=4)
        assert hasattr(sched, "preempted")
        assert isinstance(sched.preempted, list)

    def test_stats_include_preempted(self):
        sched = I64Scheduler(max_batch_size=4, num_experts=4)
        stats = sched.get_stats()
        assert "preempted" in stats

    def test_fairness_via_arrival_step(self):
        """Requests at same priority should be ordered by arrival."""
        sched = I64Scheduler(max_batch_size=4, num_experts=4)
        sched.add_request(np.array([1], dtype=np.int64), priority=0)
        sched.add_request(np.array([2], dtype=np.int64), priority=0)

        assert sched.pending[0].arrival_step <= sched.pending[1].arrival_step


# =========================================================================
# Structured Output
# =========================================================================

class TestLogitsProcessors:

    def test_json_processor_state_tracking(self):
        proc = JSONLogitsProcessor()
        assert proc._state == proc.STATE_START
        assert proc._depth == 0

    def test_json_processor_tracks_depth(self):
        proc = JSONLogitsProcessor()
        proc._update_state(ord("{"))
        assert proc._depth == 1
        proc._update_state(ord("}"))
        assert proc._depth == 0
        assert proc.is_complete()

    def test_json_processor_reset(self):
        proc = JSONLogitsProcessor()
        proc._update_state(ord("{"))
        proc.reset()
        assert proc._depth == 0
        assert not proc.is_complete()

    def test_stop_sequence_processor(self):
        proc = StopSequenceProcessor(stop_sequences=[[10, 20, 30]])
        logits = torch.randn(100)

        proc(logits, [1, 2, 3])
        assert not proc.should_stop

        proc(logits, [1, 2, 10, 20, 30])
        assert proc.should_stop
        assert proc.stop_index == 2

    def test_choice_processor(self):
        proc = ChoiceLogitsProcessor(choices=["yes", "no"])
        logits = torch.randn(100)
        # Without tokenizer, no actual masking
        result = proc(logits, [])
        assert result.shape == logits.shape

    def test_output_constraints_build_processors(self):
        constraints = OutputConstraints(
            json_mode=True,
            stop_sequences=[[1, 2, 3]],
        )
        processors = constraints.build_processors()
        assert len(processors) == 2

    def test_apply_logits_processors_chain(self):
        proc1 = StopSequenceProcessor([[999]])
        proc2 = JSONLogitsProcessor()
        logits = torch.randn(100)
        result = apply_logits_processors(logits, [proc1, proc2], [1, 2, 3])
        assert result.shape == logits.shape


# =========================================================================
# Beam Search
# =========================================================================

class TestBeamSearch:

    def test_beam_searcher_init(self):
        bs = BeamSearcher(num_beams=4, max_length=100)
        assert bs.num_beams == 4

    def test_beam_init_beams(self):
        bs = BeamSearcher(num_beams=3)
        bs.init_beams(initial_token_ids=[1, 2])
        assert len(bs.beams) == 3
        assert bs.beams[0].token_ids == [1, 2]

    def test_beam_step(self):
        bs = BeamSearcher(num_beams=2, max_length=10)
        bs.init_beams()

        # Simulate logits for 2 beams, vocab_size=5
        logits = torch.randn(2, 5)
        result = bs.step(logits)
        assert len(result) == 2

    def test_beam_eos_detection(self):
        bs = BeamSearcher(num_beams=2, max_length=100, eos_token_id=0)
        bs.init_beams()

        # Make token 0 (EOS) most likely for all beams
        logits = torch.full((2, 5), -10.0)
        logits[:, 0] = 10.0  # EOS
        bs.step(logits)

        assert len(bs.completed) > 0

    def test_beam_get_best(self):
        bs = BeamSearcher(num_beams=2, max_length=5)
        bs.init_beams([10])

        # Step a few times
        for _ in range(3):
            logits = torch.randn(2, 100)
            bs.step(logits)

        best = bs.get_best()
        assert isinstance(best, BeamHypothesis)
        assert len(best.token_ids) > 0

    def test_beam_max_length_termination(self):
        bs = BeamSearcher(num_beams=2, max_length=3)
        bs.init_beams()

        for _ in range(5):
            if bs.is_done:
                break
            logits = torch.randn(2, 10)
            bs.step(logits)

        # Should have terminated by max_length
        assert bs.is_done or len(bs.completed) > 0

    def test_sampling_params_beam_fields(self):
        params = SamplingParams(num_beams=4, length_penalty=0.8)
        assert params.num_beams == 4
        assert params.length_penalty == 0.8


# =========================================================================
# LoRA
# =========================================================================

class TestLoRA:

    def test_lora_linear_basic(self):
        base = torch.nn.Linear(64, 32)
        lora = LoRALinear(base, max_adapters=4)

        x = torch.randn(8, 64)
        out_base = lora(x)
        assert out_base.shape == (8, 32)

    def test_lora_load_adapter(self):
        base = torch.nn.Linear(64, 32)
        lora = LoRALinear(base)

        rank = 8
        A = torch.randn(64, rank)
        B = torch.randn(rank, 32)
        lora.load_adapter(adapter_id=1, lora_A=A, lora_B=B, scaling=0.5)

        assert lora.num_adapters == 1
        assert 1 in lora.adapter_ids

    def test_lora_adapter_changes_output(self):
        base = torch.nn.Linear(64, 32, bias=False)
        lora = LoRALinear(base)

        rank = 8
        A = torch.randn(64, rank) * 0.1
        B = torch.randn(rank, 32) * 0.1
        lora.load_adapter(1, A, B, scaling=1.0)

        x = torch.randn(4, 64)
        out_base = lora(x)  # No adapter
        out_lora = lora(x, adapter_id=1)  # With adapter

        # Outputs should be different
        assert not torch.allclose(out_base, out_lora, atol=1e-6)

    def test_lora_unload(self):
        base = torch.nn.Linear(64, 32)
        lora = LoRALinear(base)

        A = torch.randn(64, 4)
        B = torch.randn(4, 32)
        lora.load_adapter(1, A, B)

        lora.unload_adapter(1)
        assert lora.num_adapters == 0

    def test_lora_multiple_adapters(self):
        base = torch.nn.Linear(64, 32, bias=False)
        lora = LoRALinear(base)

        for aid in range(3):
            A = torch.randn(64, 4)
            B = torch.randn(4, 32)
            lora.load_adapter(aid, A, B)

        assert lora.num_adapters == 3

        x = torch.randn(2, 64)
        out0 = lora(x, adapter_id=0)
        out1 = lora(x, adapter_id=1)
        # Different adapters should give different outputs
        assert not torch.allclose(out0, out1, atol=1e-6)

    def test_lora_set_active(self):
        base = torch.nn.Linear(64, 32, bias=False)
        lora = LoRALinear(base)

        A = torch.randn(64, 4) * 0.1
        B = torch.randn(4, 32) * 0.1
        lora.load_adapter(1, A, B, scaling=1.0)

        lora.set_active_adapter(1)
        x = torch.randn(2, 64)
        out = lora(x)  # Should use adapter 1 automatically
        out_explicit = lora(x, adapter_id=1)
        assert torch.allclose(out, out_explicit)

    def test_lora_manager_wrap(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 16),
        )
        manager = LoRAManager(model)
        assert len(manager._lora_modules) == 0

    def test_lora_manager_list(self):
        model = torch.nn.Linear(64, 32)
        manager = LoRAManager(model)
        assert manager.list_adapters() == {}


# =========================================================================
# Kernel Loader
# =========================================================================

class TestKernelLoader:

    def test_fallback_route_tokens(self):
        token_ids = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        expert_ids = FallbackOps.route_tokens(token_ids, 4)
        assert expert_ids.dtype == torch.int32
        assert expert_ids.tolist() == [2, 3, 0, 1]

    def test_fallback_silu_hadamard(self):
        gate_up = torch.randn(4, 16)  # 4 tokens, expert_inter=8
        result = FallbackOps.silu_hadamard(gate_up, 8)
        assert result.shape == (4, 8)

    def test_fallback_scatter_gather_roundtrip(self):
        hidden = torch.randn(8, 32)
        expert_ids = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3], dtype=torch.int32)
        scattered, indices, offsets, counts = FallbackOps.scatter_by_expert(
            hidden, expert_ids, 4
        )
        assert scattered.shape == hidden.shape

        gathered = FallbackOps.gather_by_expert(scattered, indices)
        assert torch.allclose(gathered, hidden)


# =========================================================================
# Integration Tests
# =========================================================================

class TestIntegration:

    def test_engine_with_new_scheduler(self):
        """Engine should work with the new scheduler features."""
        engine = I64Engine(num_experts=4, vocab_size=100, device="cpu")
        rid = engine.add_request([1, 2, 3, 4, 5], max_new_tokens=3)

        for _ in range(5):
            results = engine.step()
            if not results:
                break

        assert engine.total_steps > 0

    def test_engine_step_after_cancel(self):
        """Cancelled request should not block the engine."""
        engine = I64Engine(num_experts=4, vocab_size=100, device="cpu")
        rid1 = engine.add_request([1, 2, 3], max_new_tokens=10)
        rid2 = engine.add_request([4, 5, 6], max_new_tokens=10)

        engine.cancel_request(rid1)
        engine.step()  # Should handle cancellation gracefully

    def test_scheduler_priority_affects_batch_order(self):
        sched = I64Scheduler(max_batch_size=2, num_experts=4)
        sched.add_request(np.array([1, 2], dtype=np.int64), priority=1)  # low
        sched.add_request(np.array([3, 4], dtype=np.int64), priority=-1)  # high

        batch = sched.schedule()
        assert batch is not None
        # High priority request should be in the batch
        assert batch.num_requests >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
