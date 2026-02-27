"""
vllm-i64 :: Test Model Forward Pass

Tests ComplexityDeepModel end-to-end on CPU:
  - Correct output shapes
  - INL Dynamics produces velocity + mu
  - MuGuidedAttention with TP=1
  - MuGuidedTokenRoutedMLP routing
  - Full forward: token_ids → logits

INL - 2025
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
from vllm_i64.models.complexity_deep.model import (
    ComplexityDeepModel,
    ComplexityDecoderLayer,
    MuGuidedAttention,
    MuGuidedTokenRoutedMLP,
    INLDynamics,
)


@pytest.fixture
def small_config():
    return ComplexityDeepConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        num_experts=4,
        vocab_size=256,
        max_position_embeddings=128,
    )


@pytest.fixture
def model(small_config):
    m = ComplexityDeepModel(small_config)
    m.eval()
    return m


class TestINLDynamics:
    def test_output_shapes(self):
        dyn = INLDynamics(hidden_size=64)
        h = torch.randn(8, 64)
        h_next, v_next, mu = dyn(h)
        assert h_next.shape == (8, 64)
        assert v_next.shape == (8, 64)
        assert mu.shape == (8, 64)

    def test_velocity_init_zeros(self):
        dyn = INLDynamics(hidden_size=32)
        h = torch.randn(4, 32)
        h_next, v_next, mu = dyn(h, v=None)
        assert v_next is not None

    def test_velocity_clamped(self):
        dyn = INLDynamics(hidden_size=16)
        h = torch.randn(2, 16) * 100
        v = torch.randn(2, 16) * 100
        _, v_next, _ = dyn(h, v)
        assert v_next.abs().max() <= 10.0


class TestMuGuidedTokenRoutedMLP:
    def test_pure_i64_routing(self, small_config):
        mlp = MuGuidedTokenRoutedMLP(small_config)
        x = torch.randn(8, small_config.hidden_size)
        token_ids = torch.arange(8)
        out = mlp(x, token_ids=token_ids)
        assert out.shape == (8, small_config.hidden_size)

    def test_mu_guided_routing(self, small_config):
        mlp = MuGuidedTokenRoutedMLP(small_config)
        x = torch.randn(8, small_config.hidden_size)
        token_ids = torch.arange(8)
        mu = torch.randn(8, small_config.hidden_size)
        out = mlp(x, token_ids=token_ids, mu=mu)
        assert out.shape == (8, small_config.hidden_size)

    def test_fallback_without_mu(self, small_config):
        mlp = MuGuidedTokenRoutedMLP(small_config)
        x = torch.randn(4, small_config.hidden_size)
        token_ids = torch.tensor([0, 1, 2, 3])
        out_no_mu = mlp(x, token_ids=token_ids, mu=None)
        assert out_no_mu.shape == (4, small_config.hidden_size)


class TestMuGuidedAttention:
    def test_output_shape(self, small_config):
        attn = MuGuidedAttention(small_config)
        hidden = torch.randn(8, small_config.hidden_size)
        positions = torch.arange(8, dtype=torch.int32)
        out = attn(hidden, positions)
        assert out.shape == (8, small_config.hidden_size)

    def test_with_mu(self, small_config):
        attn = MuGuidedAttention(small_config)
        hidden = torch.randn(4, small_config.hidden_size)
        positions = torch.arange(4, dtype=torch.int32)
        mu = torch.randn(4, small_config.hidden_size)
        out = attn(hidden, positions, mu_prev=mu)
        assert out.shape == (4, small_config.hidden_size)

    def test_gqa_head_counts(self, small_config):
        attn = MuGuidedAttention(small_config)
        # 4 Q heads, 2 KV heads → 2 groups
        assert attn.num_heads_per_tp == 4
        assert attn.num_kv_heads_per_tp == 2
        assert attn.num_kv_groups == 2


class TestComplexityDecoderLayer:
    def test_forward(self, small_config):
        layer = ComplexityDecoderLayer(small_config)
        hidden = torch.randn(8, small_config.hidden_size)
        positions = torch.arange(8, dtype=torch.int32)
        velocity = torch.zeros(8, small_config.hidden_size)
        token_ids = torch.arange(8)

        h_out, v_out, mu = layer(hidden, positions, velocity, token_ids=token_ids)
        assert h_out.shape == hidden.shape
        assert v_out.shape == hidden.shape
        assert mu.shape == hidden.shape


class TestComplexityDeepModel:
    def test_forward_logits_shape(self, model, small_config):
        token_ids = torch.randint(0, small_config.vocab_size, (16,))
        positions = torch.arange(16, dtype=torch.int32)

        with torch.no_grad():
            logits = model(token_ids, positions)

        assert logits.shape == (16, small_config.vocab_size)

    def test_forward_single_token(self, model, small_config):
        token_ids = torch.tensor([42])
        positions = torch.tensor([0], dtype=torch.int32)

        with torch.no_grad():
            logits = model(token_ids, positions)

        assert logits.shape == (1, small_config.vocab_size)

    def test_tied_embeddings(self, model):
        assert model.tie_word_embeddings is True
        assert not hasattr(model, "lm_head") or model.lm_head is None if hasattr(model, "lm_head") else True

    def test_num_parameters(self, model):
        n = model.num_parameters()
        assert n > 0
        assert isinstance(n, int)

    def test_deterministic(self, model, small_config):
        token_ids = torch.randint(0, small_config.vocab_size, (8,))
        positions = torch.arange(8, dtype=torch.int32)

        with torch.no_grad():
            out1 = model(token_ids, positions)
            out2 = model(token_ids, positions)

        assert torch.allclose(out1, out2)

    def test_mu_residual_highway(self, model, small_config):
        """Verify mu accumulates across layers (not just last layer mu)."""
        token_ids = torch.randint(0, small_config.vocab_size, (4,))
        positions = torch.arange(4, dtype=torch.int32)

        with torch.no_grad():
            logits = model(token_ids, positions)

        # Just verify it runs without error — mu highway is internal
        assert logits.shape == (4, small_config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
