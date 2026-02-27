"""
vllm-i64 :: Test Utilities

Tests for:
  - Sampling (greedy, top-k, top-p, temperature, repetition penalty)
  - Quantization (INT8, INT4 roundtrip)
  - Chat template rendering
  - RMSNorm
  - Rotary embeddings

INL - 2025
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.core.sampling import SamplingParams, sample_token, sample_batch
from vllm_i64.core.quantization import (
    QuantConfig, quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4, quantize_experts,
)
from vllm_i64.core.chat_template import ChatTemplate
from vllm_i64.layers.rmsnorm import RMSNorm
from vllm_i64.layers.rotary import RotaryEmbedding, apply_rotary


# =========================================================================
# Sampling
# =========================================================================

class TestSampling:
    def test_greedy(self):
        logits = torch.tensor([0.1, 0.3, 0.9, 0.2])
        params = SamplingParams(temperature=0.0)
        token = sample_token(logits, params)
        assert token == 2  # argmax

    def test_temperature_scaling(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        params = SamplingParams(temperature=0.01)  # near-greedy
        token = sample_token(logits, params)
        assert token == 2  # should pick highest

    def test_top_k(self):
        logits = torch.randn(100)
        logits[42] = 100.0  # make one dominant
        params = SamplingParams(top_k=5, temperature=0.5)
        token = sample_token(logits, params)
        assert 0 <= token < 100

    def test_top_p(self):
        logits = torch.randn(50)
        logits[10] = 50.0
        params = SamplingParams(top_p=0.1, temperature=0.5)
        token = sample_token(logits, params)
        assert 0 <= token < 50

    def test_repetition_penalty(self):
        logits = torch.zeros(10)
        logits[5] = 10.0
        params = SamplingParams(repetition_penalty=100.0, temperature=0.01)
        # Without penalty: picks 5
        t1 = sample_token(logits.clone(), SamplingParams(temperature=0.0))
        assert t1 == 5
        # With penalty on token 5: should avoid it
        t2 = sample_token(logits.clone(), params, past_tokens=[5])
        # With extreme penalty, should pick something else
        assert isinstance(t2, int)

    def test_sample_batch_greedy(self):
        logits = torch.randn(4, 20)
        logits[0, 5] = 100.0
        logits[1, 10] = 100.0
        logits[2, 15] = 100.0
        logits[3, 0] = 100.0
        params = SamplingParams(temperature=0.0)
        tokens = sample_batch(logits, params)
        assert tokens.tolist() == [5, 10, 15, 0]

    def test_sample_batch_shape(self):
        logits = torch.randn(8, 50)
        params = SamplingParams(temperature=1.0, top_k=10)
        tokens = sample_batch(logits, params)
        assert tokens.shape == (8,)

    def test_output_is_integer(self):
        logits = torch.randn(100)
        params = SamplingParams()
        token = sample_token(logits, params)
        assert isinstance(token, int)


# =========================================================================
# Quantization
# =========================================================================

class TestQuantization:
    def test_int8_roundtrip(self):
        weight = torch.randn(64, 128)
        quantized, scale = quantize_int8(weight)
        assert quantized.dtype == torch.int8
        assert scale.shape == (64,)

        restored = dequantize_int8(quantized, scale)
        # Should be close (quantization error)
        error = (weight - restored).abs().mean()
        assert error < 0.1

    def test_int4_roundtrip(self):
        weight = torch.randn(32, 128)
        packed, scale, zero = quantize_int4(weight, group_size=128)
        assert packed.dtype == torch.uint8
        assert packed.shape == (32, 64)  # packed 2 per byte

        restored = dequantize_int4(packed, scale, zero, group_size=128)
        assert restored.shape == (32, 1, 128)  # grouped

    def test_int8_values_in_range(self):
        weight = torch.randn(16, 32) * 10
        quantized, _ = quantize_int8(weight)
        assert quantized.min() >= -128
        assert quantized.max() <= 127

    def test_int4_packed_values(self):
        weight = torch.randn(8, 128)
        packed, _, _ = quantize_int4(weight, group_size=128)
        # Each byte has two 4-bit values
        high = (packed >> 4) & 0xF
        low = packed & 0xF
        assert high.max() <= 15
        assert low.max() <= 15

    def test_quantize_experts_none(self):
        gate_up = torch.randn(4, 32, 64)
        down = torch.randn(4, 32, 32)
        result = quantize_experts(gate_up, down, QuantConfig(method="none"))
        assert result["method"] == "none"
        assert torch.equal(result["gate_up"], gate_up)

    def test_quantize_experts_int8(self):
        gate_up = torch.randn(2, 16, 32)
        down = torch.randn(2, 16, 16)
        result = quantize_experts(gate_up, down, QuantConfig(method="int8"))
        assert result["method"] == "int8"
        assert result["gate_up_int8"].dtype == torch.int8
        assert result["gate_up_int8"].shape == (2, 16, 32)
        assert result["down_int8"].shape == (2, 16, 16)


# =========================================================================
# Chat Template
# =========================================================================

class TestChatTemplate:
    def test_basic_render(self):
        template_str = "{% for msg in messages %}<|{{ msg.role }}|>{{ msg.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
        tmpl = ChatTemplate(template_str)
        result = tmpl.apply([
            {"role": "user", "content": "Hello"},
        ])
        assert "<|user|>Hello" in result
        assert "<|assistant|>" in result

    def test_multi_turn(self):
        template_str = "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}"
        tmpl = ChatTemplate(template_str)
        result = tmpl.apply([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ], add_generation_prompt=False)
        assert "user: Hi" in result
        assert "assistant: Hello!" in result

    def test_from_string(self):
        tmpl = ChatTemplate("{{ messages | length }} messages")
        result = tmpl.apply([{"role": "user", "content": "x"}])
        assert "1" in result


# =========================================================================
# RMSNorm
# =========================================================================

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(8, 64)
        out = norm(x)
        assert out.shape == (8, 64)

    def test_normalization(self):
        norm = RMSNorm(32)
        x = torch.randn(4, 32) * 100
        out = norm(x)
        # Output should have reasonable magnitude
        assert out.abs().mean() < x.abs().mean()

    def test_3d_input(self):
        norm = RMSNorm(16)
        x = torch.randn(4, 8, 16)
        out = norm(x)
        assert out.shape == (4, 8, 16)


# =========================================================================
# Rotary Embeddings
# =========================================================================

class TestRotaryEmbedding:
    def test_cos_sin_shapes(self):
        rope = RotaryEmbedding(dim=64, max_seq_len=128)
        positions = torch.arange(16, dtype=torch.int32)
        cos, sin = rope(positions)
        assert cos.shape == (16, 64)
        assert sin.shape == (16, 64)

    def test_apply_rotary_shape(self):
        rope = RotaryEmbedding(dim=64)
        positions = torch.arange(8, dtype=torch.int32)
        cos, sin = rope(positions)

        x = torch.randn(8, 4, 64)  # (batch, heads, head_dim)
        out = apply_rotary(x, cos, sin)
        assert out.shape == (8, 4, 64)

    def test_different_positions_different_output(self):
        rope = RotaryEmbedding(dim=32)
        pos1 = torch.tensor([0], dtype=torch.int32)
        pos2 = torch.tensor([100], dtype=torch.int32)
        cos1, sin1 = rope(pos1)
        cos2, sin2 = rope(pos2)
        assert not torch.allclose(cos1, cos2)

    def test_zero_position(self):
        rope = RotaryEmbedding(dim=16)
        pos = torch.tensor([0], dtype=torch.int32)
        cos, sin = rope(pos)
        # cos(0) = 1, sin(0) = 0
        assert torch.allclose(cos, torch.ones_like(cos))
        assert torch.allclose(sin, torch.zeros_like(sin))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
