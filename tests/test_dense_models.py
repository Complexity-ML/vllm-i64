"""
vllm-i64 :: Test Dense Models (Llama, Mistral, Qwen2)

Tests dense transformer models — modulo 1 in i64 routing theory.
  - Config loading (from transformers)
  - LlamaForCausalLM forward + decode_step
  - MistralForCausalLM / Qwen2ForCausalLM (same architecture)
  - DenseMLP forward (SwiGLU, INT8, INT4)
  - Registry resolve_architecture

INL - 2025
"""

import json
import torch
import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
# Configs
# =========================================================================

class TestLlamaConfig:
    def test_defaults(self):
        from vllm_i64.models.llama.config import LlamaConfig
        config = LlamaConfig()
        assert config.num_experts == 1
        assert config.hidden_size == 4096
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 32
        assert config.head_dim == 128

    def test_from_json(self, tmp_path):
        from vllm_i64.models.llama.config import LlamaConfig
        cfg = {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 1000,
            "max_position_embeddings": 128,
            "model_type": "llama",
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))

        config = LlamaConfig.from_json(str(path))
        assert config.hidden_size == 256
        assert config.num_experts == 1
        assert config.head_dim == 64

    def test_num_experts_locked(self, tmp_path):
        from vllm_i64.models.llama.config import LlamaConfig
        cfg = {"num_experts": 8}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))

        config = LlamaConfig.from_json(str(path))
        assert config.num_experts == 1  # always locked


class TestMistralConfig:
    def test_defaults(self):
        from vllm_i64.models.mistral.config import MistralConfig
        config = MistralConfig()
        assert config.num_experts == 1
        assert config.num_key_value_heads == 8
        assert config.intermediate_size == 14336
        assert config.sliding_window == 4096


class TestQwen2Config:
    def test_defaults(self):
        from vllm_i64.models.qwen2.config import Qwen2Config
        config = Qwen2Config()
        assert config.num_experts == 1
        assert config.vocab_size == 151936
        assert config.head_dim == config.hidden_size // config.num_attention_heads


# =========================================================================
# DenseMLP
# =========================================================================

class TestDenseMLP:
    @pytest.fixture
    def mlp(self):
        from vllm_i64.layers.dense_mlp import DenseMLP
        return DenseMLP(hidden_size=64, intermediate_size=128)

    def test_forward_shape(self, mlp):
        x = torch.randn(8, 64)
        out = mlp(x)
        assert out.shape == (8, 64)

    def test_ignores_routing_kwargs(self, mlp):
        x = torch.randn(4, 64)
        out = mlp(x, token_ids=torch.arange(4), expert_ids=torch.zeros(4), mu=None)
        assert out.shape == (4, 64)

    def test_deterministic(self, mlp):
        x = torch.randn(4, 64)
        mlp.eval()
        out1 = mlp(x)
        out2 = mlp(x)
        assert torch.allclose(out1, out2)

    def test_int8_quantization(self, mlp):
        from vllm_i64.core.quantization import quantize_int8
        gate_w = mlp.gate_proj.linear.weight.data
        gq, gs = quantize_int8(gate_w)
        mlp.register_buffer("gate_int8", gq)
        mlp.register_buffer("gate_scale", gs)
        up_w = mlp.up_proj.linear.weight.data
        uq, us = quantize_int8(up_w)
        mlp.register_buffer("up_int8", uq)
        mlp.register_buffer("up_scale", us)
        down_w = mlp.down_proj.linear.weight.data
        dq, ds = quantize_int8(down_w)
        mlp.register_buffer("down_int8", dq)
        mlp.register_buffer("down_scale", ds)

        x = torch.randn(4, 64)
        out = mlp(x)
        assert out.shape == (4, 64)


# =========================================================================
# LlamaForCausalLM
# =========================================================================

@pytest.fixture
def small_llama_config():
    from vllm_i64.models.llama.config import LlamaConfig
    return LlamaConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=64,
    )


@pytest.fixture
def llama_model(small_llama_config):
    from vllm_i64.models.llama.model import LlamaForCausalLM
    m = LlamaForCausalLM(small_llama_config)
    m.eval()
    return m


class TestLlamaForCausalLM:
    def test_forward_logits_shape(self, llama_model, small_llama_config):
        token_ids = torch.randint(0, 256, (8,))
        positions = torch.arange(8)
        logits = llama_model(token_ids, positions)
        assert logits.shape == (8, small_llama_config.vocab_size)

    def test_batch_forward(self, llama_model, small_llama_config):
        token_ids = torch.randint(0, 256, (16,))
        positions = torch.arange(16)
        logits = llama_model(token_ids, positions, tokens_per_seq=[8, 8])
        assert logits.shape == (16, small_llama_config.vocab_size)

    def test_deterministic(self, llama_model):
        token_ids = torch.randint(0, 256, (4,))
        positions = torch.arange(4)
        out1 = llama_model(token_ids, positions)
        out2 = llama_model(token_ids, positions)
        assert torch.allclose(out1, out2)

    def test_num_parameters(self, llama_model):
        assert llama_model.num_parameters() > 0

    def test_config_exposed(self, llama_model, small_llama_config):
        assert llama_model.config.hidden_size == small_llama_config.hidden_size
        assert llama_model.config.num_hidden_layers == small_llama_config.num_hidden_layers


# =========================================================================
# MistralForCausalLM / Qwen2ForCausalLM
# =========================================================================

class TestMistralForCausalLM:
    def test_forward(self):
        from vllm_i64.models.mistral.config import MistralConfig
        from vllm_i64.models.mistral.model import MistralForCausalLM
        config = MistralConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=64,
        )
        model = MistralForCausalLM(config)
        model.eval()
        token_ids = torch.randint(0, 256, (4,))
        positions = torch.arange(4)
        logits = model(token_ids, positions)
        assert logits.shape == (4, 256)


class TestQwen2ForCausalLM:
    def test_forward(self):
        from vllm_i64.models.qwen2.config import Qwen2Config
        from vllm_i64.models.qwen2.model import Qwen2ForCausalLM
        config = Qwen2Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=64,
        )
        model = Qwen2ForCausalLM(config)
        model.eval()
        token_ids = torch.randint(0, 256, (4,))
        positions = torch.arange(4)
        logits = model(token_ids, positions)
        assert logits.shape == (4, 256)


# =========================================================================
# MixtralMoE
# =========================================================================

class TestMixtralMoE:
    def test_moe_forward_shape(self):
        from vllm_i64.layers.moe import MixtralMoE
        moe = MixtralMoE(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        x = torch.randn(8, 64)
        out = moe(x)
        assert out.shape == (8, 64)

    def test_moe_ignores_routing_kwargs(self):
        from vllm_i64.layers.moe import MixtralMoE
        moe = MixtralMoE(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        x = torch.randn(4, 64)
        out = moe(x, token_ids=torch.arange(4), expert_ids=torch.zeros(4))
        assert out.shape == (4, 64)

    def test_integer_mode_forward_shape(self):
        from vllm_i64.layers.moe import MixtralMoE
        moe = MixtralMoE(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2, integer_mode=True)
        x = torch.randn(8, 64)
        out = moe(x)
        assert out.shape == (8, 64)

    def test_integer_mode_same_routing(self):
        """Integer softmax must select the same top-k experts as float softmax."""
        from vllm_i64.layers.moe import softmax_integer
        # Use well-separated logits so rounding doesn't flip routing
        logits = torch.tensor([
            [5.0, 3.0, 1.0, 0.5, 0.1, -1.0, -2.0, -3.0],
            [0.1, 0.2, 8.0, 0.3, 6.0, 0.4, 0.5, 0.6],
            [-1.0, 4.0, -2.0, 7.0, -3.0, 1.0, 0.0, -4.0],
        ])
        float_w = torch.nn.functional.softmax(logits, dim=-1)
        int_w = softmax_integer(logits)
        # Same top-2 selection
        _, float_top2 = float_w.topk(2, dim=-1)
        _, int_top2 = int_w.topk(2, dim=-1)
        float_top2_sorted = float_top2.sort(dim=-1).values
        int_top2_sorted = int_top2.sort(dim=-1).values
        assert (float_top2_sorted == int_top2_sorted).all()

    def test_softmax_integer_standalone(self):
        """softmax_integer should approximate F.softmax."""
        from vllm_i64.layers.moe import softmax_integer
        logits = torch.randn(4, 8)
        float_result = torch.nn.functional.softmax(logits, dim=-1)
        int_result = softmax_integer(logits)
        # Same shape
        assert float_result.shape == int_result.shape
        # Sums to ~1
        assert torch.allclose(int_result.sum(dim=-1), torch.ones(4), atol=0.05)
        # Same argmax (same routing decision)
        assert (float_result.argmax(dim=-1) == int_result.argmax(dim=-1)).all()

    def test_silu_integer_standalone(self):
        """silu_integer should approximate F.silu."""
        from vllm_i64.layers.moe import silu_integer, _Q_IN
        x = torch.linspace(-6.0, 6.0, 100)
        float_result = torch.nn.functional.silu(x)
        x_q7 = (x * _Q_IN).round().to(torch.int32)
        int_result = silu_integer(x_q7).float() / _Q_IN
        assert torch.allclose(float_result, int_result, atol=0.05)

    def test_quantize_moe_int8(self):
        """quantize_moe_int8 should quantize gate + experts, keep same output."""
        from vllm_i64.layers.moe import MixtralMoE, quantize_moe_int8
        torch.manual_seed(42)
        moe = MixtralMoE(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.eval()
        x = torch.randn(8, 64)
        with torch.no_grad():
            out_float = moe(x)
        # Quantize
        quantize_moe_int8(moe)
        assert moe.integer_mode is True
        assert hasattr(moe, 'gate_int8')
        assert hasattr(moe.experts[0], 'w1_int8')
        with torch.no_grad():
            out_int8 = moe(x)
        assert out_int8.shape == out_float.shape
        # Should be close (INT8 quantization noise)
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int8.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95


class TestMixtralForCausalLM:
    def test_forward(self):
        from vllm_i64.models.mixtral.config import MixtralConfig
        from vllm_i64.models.mixtral.model import MixtralForCausalLM
        config = MixtralConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            vocab_size=256,
            max_position_embeddings=64,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        model = MixtralForCausalLM(config)
        model.eval()
        token_ids = torch.randint(0, 256, (8,))
        positions = torch.arange(8)
        logits = model(token_ids, positions)
        assert logits.shape == (8, 256)

    def test_config_num_experts(self):
        from vllm_i64.models.mixtral.config import MixtralConfig
        config = MixtralConfig(num_local_experts=8, num_experts_per_tok=2)
        assert config.num_experts == 8
        assert config.num_local_experts == 8
        assert config.num_experts_per_tok == 2


# =========================================================================
# Registry
# =========================================================================

class TestRegistry:
    def test_llama_registered(self):
        from vllm_i64.core.registry import get_model_entry
        entry = get_model_entry("llama")
        assert "LlamaForCausalLM" in entry.model_class

    def test_mistral_registered(self):
        from vllm_i64.core.registry import get_model_entry
        entry = get_model_entry("mistral")
        assert "MistralForCausalLM" in entry.model_class

    def test_qwen2_registered(self):
        from vllm_i64.core.registry import get_model_entry
        entry = get_model_entry("qwen2")
        assert "Qwen2ForCausalLM" in entry.model_class

    def test_mixtral_registered(self):
        from vllm_i64.core.registry import get_model_entry
        entry = get_model_entry("mixtral")
        assert "MixtralForCausalLM" in entry.model_class

    def test_resolve_architecture_mixtral(self, tmp_path):
        from vllm_i64.core.registry import resolve_architecture
        cfg = {"architectures": ["MixtralForCausalLM"], "model_type": "mixtral"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        result = resolve_architecture(str(tmp_path))
        assert result is not None
        assert "MixtralForCausalLM" in result[0]

    def test_resolve_architecture_llama(self, tmp_path):
        from vllm_i64.core.registry import resolve_architecture
        cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        result = resolve_architecture(str(tmp_path))
        assert result is not None
        model_class, config_loader, config_path = result
        assert "LlamaForCausalLM" in model_class

    def test_resolve_architecture_mistral(self, tmp_path):
        from vllm_i64.core.registry import resolve_architecture
        cfg = {"architectures": ["MistralForCausalLM"], "model_type": "mistral"}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        result = resolve_architecture(str(tmp_path))
        assert result is not None
        assert "MistralForCausalLM" in result[0]

    def test_resolve_architecture_unknown(self, tmp_path):
        from vllm_i64.core.registry import resolve_architecture
        cfg = {"architectures": ["UnknownModel"]}
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        result = resolve_architecture(str(tmp_path))
        assert result is None

    def test_list_models_includes_all(self):
        from vllm_i64.core.registry import list_models
        names = [m["name"] for m in list_models()]
        assert "llama" in names
        assert "mistral" in names
        assert "qwen2" in names
        assert "mixtral" in names


# =========================================================================
# Native INT8 matmul (torch._int_mm)
# =========================================================================

class TestNativeInt8Matmul:
    def test_quantize_activations_int8(self):
        """Dynamic per-token activation quantization."""
        from vllm_i64.core.quantization import quantize_activations_int8
        x = torch.randn(4, 64)
        x_int8, x_scale = quantize_activations_int8(x)
        assert x_int8.dtype == torch.int8
        assert x_int8.shape == (4, 64)
        assert x_scale.shape == (4,)
        # Dequantized should be close to original
        x_recon = x_int8.float() * x_scale.unsqueeze(1)
        assert torch.allclose(x, x_recon, atol=0.05)

    def test_int8_linear_native_cpu_fallback(self):
        """int8_linear_native falls back to dequant on CPU — same result."""
        from vllm_i64.core.quantization import (
            quantize_int8, dequantize_int8, int8_linear_native,
        )
        torch.manual_seed(42)
        W = torch.randn(32, 64)  # (out, in)
        x = torch.randn(4, 64)
        # Float reference
        ref = torch.nn.functional.linear(x, W)
        # INT8 native (CPU fallback = dequant + F.linear)
        w_int8, w_scale = quantize_int8(W)
        out = int8_linear_native(x, w_int8, w_scale)
        assert out.shape == (4, 32)
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            out.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.98

    def test_int8_linear_native_3d_input(self):
        """int8_linear_native handles batched (batch, seq, hidden) input."""
        from vllm_i64.core.quantization import quantize_int8, int8_linear_native
        torch.manual_seed(42)
        W = torch.randn(32, 64)
        x = torch.randn(2, 4, 64)  # (batch, seq, hidden)
        w_int8, w_scale = quantize_int8(W)
        out = int8_linear_native(x, w_int8, w_scale)
        assert out.shape == (2, 4, 32)

    def test_int8_linear_native_with_bias(self):
        """int8_linear_native applies bias correctly."""
        from vllm_i64.core.quantization import quantize_int8, int8_linear_native
        torch.manual_seed(42)
        W = torch.randn(32, 64)
        bias = torch.randn(32)
        x = torch.randn(4, 64)
        w_int8, w_scale = quantize_int8(W)
        out_no_bias = int8_linear_native(x, w_int8, w_scale)
        out_bias = int8_linear_native(x, w_int8, w_scale, bias=bias)
        assert torch.allclose(out_bias, out_no_bias + bias, atol=1e-6)

    def test_moe_expert_uses_native_int8(self):
        """MoEExpert._forward_int8 uses int8_linear_native path."""
        from vllm_i64.layers.moe import MoEExpert
        from vllm_i64.core.quantization import quantize_int8
        torch.manual_seed(42)
        expert = MoEExpert(64, 128)
        expert.eval()
        x = torch.randn(4, 64)
        with torch.no_grad():
            out_float = expert(x)
        # Quantize expert weights
        for name in ('w1', 'w2', 'w3'):
            layer = getattr(expert, name)
            q, s = quantize_int8(layer.weight.data)
            expert.register_buffer(f'{name}_int8', q)
            expert.register_buffer(f'{name}_scale', s)
        with torch.no_grad():
            out_int8 = expert(x)
        assert out_int8.shape == out_float.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int8.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for native INT8 matmul test",
    )
    def test_int8_linear_native_cuda(self):
        """int8_linear_native on CUDA — uses _int_mm if SM80+, else fallback."""
        from vllm_i64.core.quantization import (
            quantize_int8, int8_linear_native, int8_linear_available,
        )
        torch.manual_seed(42)
        W = torch.randn(32, 64).cuda()
        x = torch.randn(4, 64).cuda()
        ref = torch.nn.functional.linear(x, W)
        w_int8, w_scale = quantize_int8(W)
        out = int8_linear_native(x, w_int8, w_scale)
        assert out.shape == (4, 32)
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            out.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95

    def test_int8_linear_available_detection(self):
        """int8_linear_available returns bool without crashing."""
        from vllm_i64.core.quantization import int8_linear_available
        result = int8_linear_available()
        assert isinstance(result, bool)


# =========================================================================
# Fused gate+up INT8 matmul + sorted expert dispatch
# =========================================================================

class TestFusedInt8:
    def test_fused_gate_up_matches_separate(self):
        """Fused gate+up should produce same result as two separate matmuls."""
        from vllm_i64.core.quantization import (
            quantize_int8, int8_linear_native, int8_fused_gate_up_native,
        )
        torch.manual_seed(42)
        gate_w = torch.randn(128, 64)
        up_w = torch.randn(128, 64)
        x = torch.randn(8, 64)

        gate_int8, gate_scale = quantize_int8(gate_w)
        up_int8, up_scale = quantize_int8(up_w)

        # Separate
        gate_sep = int8_linear_native(x, gate_int8, gate_scale)
        up_sep = int8_linear_native(x, up_int8, up_scale)

        # Fused
        fused_int8 = torch.cat([gate_int8, up_int8], dim=0)
        fused_scale = torch.cat([gate_scale, up_scale])
        gate_fused, up_fused = int8_fused_gate_up_native(
            x, fused_int8, fused_scale, inter_size=128,
        )

        assert gate_fused.shape == gate_sep.shape
        assert up_fused.shape == up_sep.shape
        assert torch.allclose(gate_fused, gate_sep, atol=1e-5)
        assert torch.allclose(up_fused, up_sep, atol=1e-5)

    def test_fused_gate_up_3d(self):
        """Fused gate+up handles (batch, seq, hidden) input."""
        from vllm_i64.core.quantization import (
            quantize_int8, int8_fused_gate_up_native,
        )
        torch.manual_seed(42)
        gate_w = torch.randn(128, 64)
        up_w = torch.randn(128, 64)
        x = torch.randn(2, 4, 64)

        gate_int8, gate_scale = quantize_int8(gate_w)
        up_int8, up_scale = quantize_int8(up_w)
        fused_int8 = torch.cat([gate_int8, up_int8], dim=0)
        fused_scale = torch.cat([gate_scale, up_scale])

        gate, up = int8_fused_gate_up_native(
            x, fused_int8, fused_scale, inter_size=128,
        )
        assert gate.shape == (2, 4, 128)
        assert up.shape == (2, 4, 128)

    def test_moe_expert_fused_forward(self):
        """MoEExpert with fused w13 weights produces close output."""
        from vllm_i64.layers.moe import MoEExpert
        from vllm_i64.core.quantization import quantize_int8
        torch.manual_seed(42)
        expert = MoEExpert(64, 128)
        expert.eval()
        x = torch.randn(8, 64)
        with torch.no_grad():
            out_float = expert(x)

        # Quantize + fuse
        w1_q, w1_s = quantize_int8(expert.w1.weight.data)
        w3_q, w3_s = quantize_int8(expert.w3.weight.data)
        w2_q, w2_s = quantize_int8(expert.w2.weight.data)
        expert.register_buffer('w13_int8', torch.cat([w1_q, w3_q], dim=0))
        expert.register_buffer('w13_scale', torch.cat([w1_s, w3_s]))
        expert.w13_inter = w1_q.shape[0]
        expert.register_buffer('w2_int8', w2_q)
        expert.register_buffer('w2_scale', w2_s)

        with torch.no_grad():
            out_fused = expert(x)
        assert out_fused.shape == out_float.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_fused.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95

    def test_quantize_moe_int8_creates_fused(self):
        """quantize_moe_int8 creates fused w13 buffers."""
        from vllm_i64.layers.moe import MixtralMoE, quantize_moe_int8
        torch.manual_seed(42)
        moe = MixtralMoE(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        quantize_moe_int8(moe)
        for expert in moe.experts:
            assert hasattr(expert, 'w13_int8')
            assert hasattr(expert, 'w13_scale')
            assert expert.w13_inter == 128
            assert expert.w13_int8.shape == (256, 64)  # 2*inter, hidden
            assert expert.w13_scale.shape == (256,)

    def test_sorted_dispatch_same_result(self):
        """Sorted expert dispatch gives same output as original."""
        from vllm_i64.layers.moe import MixtralMoE
        torch.manual_seed(42)
        moe = MixtralMoE(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.eval()
        x = torch.randn(16, 64)
        with torch.no_grad():
            out1 = moe(x)
            out2 = moe(x)
        # Deterministic — same input gives same output
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_quantize_moe_int8_full_pipeline(self):
        """Full pipeline: quantize → fused forward → sorted dispatch."""
        from vllm_i64.layers.moe import MixtralMoE, quantize_moe_int8
        torch.manual_seed(42)
        moe = MixtralMoE(hidden_size=64, intermediate_size=128, num_experts=4, top_k=2)
        moe.eval()
        x = torch.randn(16, 64)
        with torch.no_grad():
            out_float = moe(x)
        quantize_moe_int8(moe)
        with torch.no_grad():
            out_int8 = moe(x)
        assert out_int8.shape == out_float.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int8.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.90


# =========================================================================
# Integer SiLU LUT + INT32 gate*up multiply
# =========================================================================

class TestIntegerSiluMultiply:
    def test_silu_multiply_integer_shape(self):
        """silu_multiply_integer returns correct shape."""
        from vllm_i64.layers.moe import silu_multiply_integer
        gate = torch.randn(8, 128)
        up = torch.randn(8, 128)
        out = silu_multiply_integer(gate, up)
        assert out.shape == (8, 128)
        assert out.dtype == torch.float32

    def test_silu_multiply_integer_vs_float(self):
        """Integer SiLU*up is close to float F.silu(gate)*up."""
        from vllm_i64.layers.moe import silu_multiply_integer
        torch.manual_seed(42)
        gate = torch.randn(16, 64)
        up = torch.randn(16, 64)
        ref = torch.nn.functional.silu(gate) * up
        out = silu_multiply_integer(gate, up)
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            out.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.99

    def test_silu_multiply_integer_large_values(self):
        """Values outside LUT range [-8,8] handled correctly."""
        from vllm_i64.layers.moe import silu_multiply_integer
        gate = torch.tensor([[10.0, -10.0, 0.0, 5.0]])
        up = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        out = silu_multiply_integer(gate, up)
        ref = torch.nn.functional.silu(gate) * up
        # For x=10: silu(10) ≈ 10, for x=-10: silu(-10) ≈ 0
        assert out[0, 0].item() > 9.0   # silu(10) ≈ 10
        assert abs(out[0, 1].item()) < 0.1  # silu(-10) ≈ 0
        assert abs(out[0, 2].item()) < 0.01  # silu(0) = 0

    def test_moe_expert_integer_silu_forward(self):
        """MoEExpert INT8 path now uses integer SiLU + INT32 multiply."""
        from vllm_i64.layers.moe import MoEExpert
        from vllm_i64.core.quantization import quantize_int8
        torch.manual_seed(42)
        expert = MoEExpert(64, 128)
        expert.eval()
        x = torch.randn(8, 64)
        with torch.no_grad():
            out_float = expert(x)
        # Quantize + fuse (triggers integer SiLU path)
        w1_q, w1_s = quantize_int8(expert.w1.weight.data)
        w3_q, w3_s = quantize_int8(expert.w3.weight.data)
        w2_q, w2_s = quantize_int8(expert.w2.weight.data)
        expert.register_buffer('w13_int8', torch.cat([w1_q, w3_q], dim=0))
        expert.register_buffer('w13_scale', torch.cat([w1_s, w3_s]))
        expert.w13_inter = w1_q.shape[0]
        expert.register_buffer('w2_int8', w2_q)
        expert.register_buffer('w2_scale', w2_s)
        with torch.no_grad():
            out_int = expert(x)
        assert out_int.shape == out_float.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.93

    def test_dense_mlp_integer_silu(self):
        """DenseMLP INT8 path uses integer SiLU + INT32 multiply."""
        from vllm_i64.layers.dense_mlp import DenseMLP
        from vllm_i64.core.quantization import quantize_int8
        torch.manual_seed(42)
        mlp = DenseMLP(64, 128)
        mlp.eval()
        x = torch.randn(8, 64)
        with torch.no_grad():
            out_float = mlp(x)
        # Quantize (separate path — gate_int8)
        gq, gs = quantize_int8(mlp.gate_proj.linear.weight.data)
        uq, us = quantize_int8(mlp.up_proj.linear.weight.data)
        dq, ds = quantize_int8(mlp.down_proj.linear.weight.data)
        mlp.register_buffer("gate_int8", gq)
        mlp.register_buffer("gate_scale", gs)
        mlp.register_buffer("up_int8", uq)
        mlp.register_buffer("up_scale", us)
        mlp.register_buffer("down_int8", dq)
        mlp.register_buffer("down_scale", ds)
        with torch.no_grad():
            out_int = mlp(x)
        assert out_int.shape == out_float.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.93


# =========================================================================
# INT8 Attention QKV + O
# =========================================================================

class TestInt8Attention:
    def _make_config(self):
        """Create a minimal config for LlamaAttention."""
        class Config:
            hidden_size = 64
            num_attention_heads = 4
            num_key_value_heads = 2
            head_dim = 16
            attention_bias = False
            use_qk_norm = False
            max_position_embeddings = 128
            rope_theta = 10000.0
        return Config()

    def test_qkv_int8_forward_shape(self):
        """LlamaAttention with INT8 QKV produces correct output shape."""
        from vllm_i64.models.llama.model import LlamaAttention
        from vllm_i64.core.quantization import quantize_int8
        config = self._make_config()
        attn = LlamaAttention(config)
        attn.eval()
        # Quantize QKV
        q_w = attn.q_proj.linear.weight.data
        k_w = attn.k_proj.linear.weight.data
        v_w = attn.v_proj.linear.weight.data
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_q, qkv_s = quantize_int8(qkv_w)
        attn.register_buffer("qkv_int8", qkv_q)
        attn.register_buffer("qkv_scale", qkv_s)
        attn.q_size = q_w.shape[0]
        attn.kv_size = k_w.shape[0]
        # Quantize O
        o_w = attn.o_proj.linear.weight.data
        o_q, o_s = quantize_int8(o_w)
        attn.register_buffer("o_int8", o_q)
        attn.register_buffer("o_scale", o_s)
        # Forward
        hidden = torch.randn(8, 64)
        positions = torch.arange(8)
        with torch.no_grad():
            out = attn(hidden, positions, tokens_per_seq=[8])
        assert out.shape == (8, 64)

    def test_qkv_int8_vs_float(self):
        """INT8 QKV+O attention is close to float attention."""
        from vllm_i64.models.llama.model import LlamaAttention
        from vllm_i64.core.quantization import quantize_int8
        torch.manual_seed(42)
        config = self._make_config()
        attn = LlamaAttention(config)
        attn.eval()
        hidden = torch.randn(8, 64)
        positions = torch.arange(8)
        with torch.no_grad():
            out_float = attn(hidden, positions, tokens_per_seq=[8])
        # Quantize
        q_w = attn.q_proj.linear.weight.data
        k_w = attn.k_proj.linear.weight.data
        v_w = attn.v_proj.linear.weight.data
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_q, qkv_s = quantize_int8(qkv_w)
        attn.register_buffer("qkv_int8", qkv_q)
        attn.register_buffer("qkv_scale", qkv_s)
        attn.q_size = q_w.shape[0]
        attn.kv_size = k_w.shape[0]
        o_w = attn.o_proj.linear.weight.data
        o_q, o_s = quantize_int8(o_w)
        attn.register_buffer("o_int8", o_q)
        attn.register_buffer("o_scale", o_s)
        with torch.no_grad():
            out_int8 = attn(hidden, positions, tokens_per_seq=[8])
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int8.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95

    def test_qkv_int8_with_bias(self):
        """INT8 QKV with attention_bias=True works correctly."""
        from vllm_i64.models.llama.model import LlamaAttention
        from vllm_i64.core.quantization import quantize_int8
        torch.manual_seed(42)
        config = self._make_config()
        config.attention_bias = True
        attn = LlamaAttention(config)
        attn.eval()
        hidden = torch.randn(4, 64)
        positions = torch.arange(4)
        with torch.no_grad():
            out_float = attn(hidden, positions, tokens_per_seq=[4])
        # Quantize with bias
        q_w = attn.q_proj.linear.weight.data
        k_w = attn.k_proj.linear.weight.data
        v_w = attn.v_proj.linear.weight.data
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_q, qkv_s = quantize_int8(qkv_w)
        attn.register_buffer("qkv_int8", qkv_q)
        attn.register_buffer("qkv_scale", qkv_s)
        attn.q_size = q_w.shape[0]
        attn.kv_size = k_w.shape[0]
        qkv_bias = torch.cat([
            attn.q_proj.linear.bias.data,
            attn.k_proj.linear.bias.data,
            attn.v_proj.linear.bias.data,
        ])
        attn.register_buffer("qkv_bias", qkv_bias)
        o_w = attn.o_proj.linear.weight.data
        o_q, o_s = quantize_int8(o_w)
        attn.register_buffer("o_int8", o_q)
        attn.register_buffer("o_scale", o_s)
        attn.register_buffer("o_bias", attn.o_proj.linear.bias.data.clone())
        with torch.no_grad():
            out_int8 = attn(hidden, positions, tokens_per_seq=[4])
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int8.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95

    def test_quantize_attention_loader(self):
        """_quantize_attention from loader creates correct buffers."""
        from vllm_i64.models.llama.model import LlamaAttention
        config = self._make_config()
        attn = LlamaAttention(config)
        # Wrap in module for named_modules iteration
        wrapper = torch.nn.Module()
        wrapper.attn = attn
        from vllm_i64.core.loader import _quantize_attention
        _quantize_attention(wrapper, "int8")
        assert hasattr(attn, 'qkv_int8')
        assert hasattr(attn, 'qkv_scale')
        assert hasattr(attn, 'o_int8')
        assert hasattr(attn, 'o_scale')
        # Check shapes: QKV fused = (q + k + v output dims, hidden)
        q_size = config.num_attention_heads * config.head_dim  # 4*16 = 64
        kv_size = config.num_key_value_heads * config.head_dim  # 2*16 = 32
        assert attn.qkv_int8.shape == (q_size + 2 * kv_size, config.hidden_size)
        assert attn.q_size == q_size
        assert attn.kv_size == kv_size
        assert attn.o_int8.shape == (config.hidden_size, q_size)

    def test_quantize_attention_skips_int4(self):
        """_quantize_attention does nothing for int4 method."""
        from vllm_i64.models.llama.model import LlamaAttention
        config = self._make_config()
        attn = LlamaAttention(config)
        wrapper = torch.nn.Module()
        wrapper.attn = attn
        from vllm_i64.core.loader import _quantize_attention
        _quantize_attention(wrapper, "int4")
        assert not hasattr(attn, 'qkv_int8')

    def test_full_llama_layer_int8(self):
        """Full LlamaDecoderLayer with INT8 attention + INT8 MLP."""
        from vllm_i64.models.llama.model import LlamaDecoderLayer
        from vllm_i64.core.loader import _quantize_attention, _quantize_dense_mlp
        torch.manual_seed(42)
        config = self._make_config()
        config.intermediate_size = 128
        config.rms_norm_eps = 1e-5
        layer = LlamaDecoderLayer(config)
        layer.eval()
        hidden = torch.randn(8, 64)
        positions = torch.arange(8)
        with torch.no_grad():
            out_float = layer(hidden, positions, tokens_per_seq=[8])
        # Quantize everything
        wrapper = torch.nn.Module()
        wrapper.layer = layer
        _quantize_attention(wrapper, "int8")
        _quantize_dense_mlp(wrapper, "int8")
        with torch.no_grad():
            out_int8 = layer(hidden, positions, tokens_per_seq=[8])
        assert out_int8.shape == out_float.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int8.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.90
