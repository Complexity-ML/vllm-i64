"""
vllm-i64 :: Tests

Verifies that the entire pipeline is integer-first:
  - Routing produces integer expert IDs
  - Scatter/gather preserve token order via integer indices
  - Engine control flow is integer
  - Only expert MLP compute is float

INL - 2025
"""

import torch
import numpy as np
import pytest
from vllm_i64.kernels.i64_ops import (
    i64_route_tokens,
    i64_scatter,
    i64_gather,
    i64_expert_forward,
    i64_full_pipeline,
)
from vllm_i64.engine.i64_scheduler import I64Scheduler, I64Request
from vllm_i64.engine.i64_engine import I64Engine


# =============================================================================
# Routing tests
# =============================================================================

class TestI64Routing:
    """Test that routing is pure integer, zero float."""

    def test_basic_routing(self):
        token_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
        expert_ids = i64_route_tokens(token_ids, num_experts=4)

        assert expert_ids.dtype == torch.int32
        expected = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32)
        assert torch.equal(expert_ids, expected)

    def test_bit_mask_equals_modulo(self):
        """Verify bit mask == modulo for all tokens."""
        token_ids = torch.arange(100_000, dtype=torch.int64)
        for n_exp in [2, 4, 8, 16]:
            mask_result = i64_route_tokens(token_ids, n_exp)
            mod_result = (token_ids % n_exp).to(torch.int32)
            assert torch.equal(mask_result, mod_result), f"Failed for N={n_exp}"

    def test_uniform_distribution(self):
        """Theorem 4.1: uniform distribution over vocabulary."""
        token_ids = torch.arange(100_000, dtype=torch.int64)
        expert_ids = i64_route_tokens(token_ids, num_experts=4)

        counts = torch.bincount(expert_ids, minlength=4)
        assert torch.all(counts == 25_000)

    def test_output_is_integer(self):
        """Routing output must be integer, not float."""
        token_ids = torch.randint(0, 100_000, (1024,), dtype=torch.int64)
        expert_ids = i64_route_tokens(token_ids, num_experts=4)

        assert expert_ids.dtype == torch.int32
        assert not expert_ids.is_floating_point()


# =============================================================================
# Scatter/Gather tests
# =============================================================================

class TestI64ScatterGather:
    """Test scatter and gather preserve data via integer indexing."""

    def test_scatter_groups_by_expert(self):
        hidden = torch.randn(8, 64)
        expert_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32)

        scattered, indices, offsets, counts = i64_scatter(hidden, expert_ids, 4)

        # Each expert should have 2 tokens
        assert torch.all(counts == 2)
        # Indices must be integer
        assert indices.dtype == torch.int32
        assert offsets.dtype == torch.int32

    def test_gather_restores_order(self):
        """scatter → gather must return original data."""
        hidden = torch.randn(16, 128)
        token_ids = torch.randint(0, 100_000, (16,), dtype=torch.int64)
        expert_ids = i64_route_tokens(token_ids, 4)

        scattered, indices, offsets, counts = i64_scatter(hidden, expert_ids, 4)
        restored = i64_gather(scattered, indices)

        assert torch.allclose(hidden, restored, atol=1e-6)

    def test_roundtrip_with_experts(self):
        """Full pipeline preserves shapes."""
        num_tokens, hidden_dim, num_experts = 32, 768, 4
        expert_inter = hidden_dim // num_experts

        token_ids = torch.randint(0, 100_000, (num_tokens,), dtype=torch.int64)
        hidden = torch.randn(num_tokens, hidden_dim)
        gate_up_w = torch.randn(num_experts, hidden_dim, 2 * expert_inter) * 0.02
        down_w = torch.randn(num_experts, expert_inter, hidden_dim) * 0.02

        output = i64_full_pipeline(token_ids, hidden, gate_up_w, down_w, num_experts)

        assert output.shape == hidden.shape


# =============================================================================
# Scheduler tests
# =============================================================================

class TestI64Scheduler:
    """Test that scheduler uses only integer operations."""

    def test_add_request(self):
        sched = I64Scheduler(num_experts=4)
        req_id = sched.add_request(np.array([1, 2, 3], dtype=np.int64), max_new_tokens=10)

        assert isinstance(req_id, int)
        assert len(sched.pending) == 1

    def test_schedule_prefill(self):
        sched = I64Scheduler(num_experts=4)
        sched.add_request(np.array([10, 20, 30, 40], dtype=np.int64))

        batch = sched.schedule()

        assert batch is not None
        assert batch.token_ids.dtype == np.int64
        assert batch.expert_ids.dtype == np.int32
        assert batch.positions.dtype == np.int32
        assert len(batch.token_ids) == 4  # All prompt tokens

    def test_expert_ids_precomputed(self):
        """Expert IDs are pre-computed in scheduler (integer)."""
        sched = I64Scheduler(num_experts=4)
        tokens = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        sched.add_request(tokens)

        batch = sched.schedule()

        expected = (tokens & 3).astype(np.int32)
        np.testing.assert_array_equal(batch.expert_ids, expected)

    def test_continuous_batching(self):
        """Multiple requests batched together."""
        sched = I64Scheduler(num_experts=4, max_batch_size=4)
        sched.add_request(np.array([1, 2], dtype=np.int64), max_new_tokens=2)
        sched.add_request(np.array([3, 4], dtype=np.int64), max_new_tokens=2)

        batch = sched.schedule()
        assert batch.num_requests == 2

    def test_kv_blocks_integer(self):
        sched = I64Scheduler(num_experts=4, kv_block_size=4)
        sched.add_request(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64))

        batch = sched.schedule()
        assert batch.kv_block_table.dtype == np.int32

    def test_stats_all_integer(self):
        sched = I64Scheduler()
        stats = sched.get_stats()
        for key, val in stats.items():
            assert isinstance(val, int), f"{key} is {type(val)}, expected int"


# =============================================================================
# Engine tests
# =============================================================================

class TestI64Engine:
    """Test engine with dummy model (no GPU required)."""

    def test_generate_no_model(self):
        """Engine works without model (dummy logits)."""
        engine = I64Engine(model=None, num_experts=4, vocab_size=1000)
        result = engine.generate(
            prompt_token_ids=[1, 2, 3, 4, 5],
            max_new_tokens=10,
        )

        assert isinstance(result.request_id, int)
        assert len(result.output_tokens) == 10
        assert all(isinstance(t, int) for t in result.output_tokens)

    def test_stats_all_integer(self):
        engine = I64Engine(model=None, num_experts=4)
        stats = engine.get_stats()
        for key, val in stats.items():
            assert isinstance(val, int), f"{key} is {type(val)}, expected int"

    def test_multiple_requests(self):
        engine = I64Engine(model=None, num_experts=4, vocab_size=1000)

        id1 = engine.add_request([10, 20, 30], max_new_tokens=5)
        id2 = engine.add_request([40, 50, 60], max_new_tokens=5)

        engine.run_continuous()

        assert engine.total_tokens_generated >= 10
        assert len(engine.scheduler.finished) == 2


# =============================================================================
# Integer purity audit
# =============================================================================

class TestIntegerPurity:
    """
    Verify that the ONLY float operations are in expert MLP compute.
    Everything else must be integer.
    """

    def test_routing_no_float(self):
        """Routing must not produce any float tensor."""
        token_ids = torch.randint(0, 100_000, (100,), dtype=torch.int64)
        result = i64_route_tokens(token_ids, 4)
        assert not result.is_floating_point()

    def test_scheduler_no_float(self):
        """Scheduler must not contain any float fields."""
        sched = I64Scheduler()
        sched.add_request(np.array([1, 2, 3], dtype=np.int64))
        batch = sched.schedule()

        for field_name in ['request_ids', 'token_ids', 'expert_ids',
                           'seq_lens', 'positions', 'kv_block_table', 'is_prefill']:
            arr = getattr(batch, field_name)
            assert arr.dtype in (np.int32, np.int64), \
                f"Batch.{field_name} is {arr.dtype}, expected integer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
