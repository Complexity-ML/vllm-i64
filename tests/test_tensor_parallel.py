"""
vllm-i64 :: Test Tensor Parallelism (Single-Rank)

Tests TP primitives with tp_size=1 (no distributed required):
  - ColumnParallelLinear shape and load_full_weight
  - RowParallelLinear shape and load_full_weight
  - shard_expert_weights correctness
  - all_reduce passthrough (tp=1)
  - Model init with TP layers

INL - 2025
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.parallel.tensor_parallel import (
    TPState,
    ColumnParallelLinear,
    RowParallelLinear,
    shard_expert_weights,
    all_reduce,
    get_tp,
)


class TestColumnParallelLinear:
    def test_output_shape(self):
        # tp_size=1 → full output
        layer = ColumnParallelLinear(64, 128)
        x = torch.randn(8, 64)
        out = layer(x)
        assert out.shape == (8, 128)

    def test_load_full_weight(self):
        layer = ColumnParallelLinear(64, 128)
        full_weight = torch.randn(128, 64)
        layer.load_full_weight(full_weight)
        assert layer.linear.weight.shape == (128, 64)
        assert torch.allclose(layer.linear.weight.data, full_weight)

    def test_load_full_weight_with_bias(self):
        layer = ColumnParallelLinear(32, 64, bias=True)
        full_weight = torch.randn(64, 32)
        full_bias = torch.randn(64)
        layer.load_full_weight(full_weight, full_bias)
        assert torch.allclose(layer.linear.weight.data, full_weight)
        assert torch.allclose(layer.linear.bias.data, full_bias)


class TestRowParallelLinear:
    def test_output_shape(self):
        layer = RowParallelLinear(128, 64)
        x = torch.randn(8, 128)
        out = layer(x)
        assert out.shape == (8, 64)

    def test_load_full_weight(self):
        layer = RowParallelLinear(128, 64)
        full_weight = torch.randn(64, 128)
        layer.load_full_weight(full_weight)
        assert layer.linear.weight.shape == (64, 128)
        assert torch.allclose(layer.linear.weight.data, full_weight)

    def test_no_all_reduce_tp1(self):
        """With tp=1, all_reduce is a no-op."""
        layer = RowParallelLinear(64, 32)
        x = torch.randn(4, 64)
        out = layer(x)
        # Should not crash — all_reduce skipped for tp=1
        assert out.shape == (4, 32)


class TestShardExpertWeights:
    def test_passthrough_tp1(self):
        gate_up = torch.randn(4, 64, 128)  # 4 experts
        down = torch.randn(4, 64, 64)
        gu_shard, dn_shard = shard_expert_weights(gate_up, down)
        # tp=1 → no sharding
        assert torch.equal(gu_shard, gate_up)
        assert torch.equal(dn_shard, down)

    def test_shapes_preserved(self):
        gate_up = torch.randn(2, 32, 64)
        down = torch.randn(2, 32, 32)
        gu, dn = shard_expert_weights(gate_up, down)
        assert gu.shape == gate_up.shape
        assert dn.shape == down.shape


class TestAllReduce:
    def test_passthrough_tp1(self):
        x = torch.randn(4, 8)
        out = all_reduce(x)
        assert torch.equal(out, x)


class TestTPState:
    def test_default_state(self):
        tp = get_tp()
        assert tp.tp_size == 1
        assert tp.tp_rank == 0
        assert tp.tp_group is None


class TestModelWithTP:
    def test_model_creates_tp_layers(self):
        from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
        from vllm_i64.models.complexity_deep.model import ComplexityDeepModel

        config = ComplexityDeepConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=128, num_experts=2,
            vocab_size=32,
        )
        model = ComplexityDeepModel(config)

        # Attention uses ColumnParallel/RowParallel
        attn = model.layers[0].self_attn
        assert isinstance(attn.q_proj, ColumnParallelLinear)
        assert isinstance(attn.k_proj, ColumnParallelLinear)
        assert isinstance(attn.v_proj, ColumnParallelLinear)
        assert isinstance(attn.o_proj, RowParallelLinear)

    def test_forward_with_tp_layers(self):
        from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
        from vllm_i64.models.complexity_deep.model import ComplexityDeepModel

        config = ComplexityDeepConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=128, num_experts=2,
            vocab_size=32, max_position_embeddings=32,
        )
        model = ComplexityDeepModel(config)
        model.eval()

        token_ids = torch.randint(0, 32, (8,))
        positions = torch.arange(8, dtype=torch.int32)

        with torch.no_grad():
            logits = model(token_ids, positions)

        assert logits.shape == (8, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
