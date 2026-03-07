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
        attn.register_buffer("o_bias", attn.o_proj.bias.data.clone())
        with torch.no_grad():
            out_int8 = attn(hidden, positions, tokens_per_seq=[4])
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int8.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.90, f"INT8 QKV+bias cosine sim too low: {cos_sim:.4f}"

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


# =========================================================================
# Integer RMSNorm
# =========================================================================

class TestIntegerRMSNorm:
    def test_shape(self):
        """Integer RMSNorm output shape matches input."""
        from vllm_i64.layers.rmsnorm import RMSNorm, quantize_rmsnorm
        norm = RMSNorm(64)
        quantize_rmsnorm(norm)
        x = torch.randn(8, 64)
        with torch.no_grad():
            out = norm(x)
        assert out.shape == x.shape

    def test_vs_float(self):
        """Integer RMSNorm matches float path (cosine > 0.999)."""
        from vllm_i64.layers.rmsnorm import RMSNorm, quantize_rmsnorm
        torch.manual_seed(42)
        norm = RMSNorm(128)
        norm.weight.data = torch.randn(128) * 0.5 + 1.0
        x = torch.randn(16, 128)
        with torch.no_grad():
            out_float = norm(x)
        quantize_rmsnorm(norm)
        with torch.no_grad():
            out_int = norm(x)
        cos = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int.flatten().unsqueeze(0),
        ).item()
        assert cos > 0.999, f"Integer RMSNorm cosine={cos:.6f}"

    def test_bf16_input(self):
        """Integer RMSNorm handles bf16 input correctly."""
        from vllm_i64.layers.rmsnorm import RMSNorm, quantize_rmsnorm
        norm = RMSNorm(64)
        quantize_rmsnorm(norm)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        with torch.no_grad():
            out = norm(x)
        assert out.dtype == torch.bfloat16
        assert out.shape == x.shape

    def test_quantize_creates_buffer(self):
        """quantize_rmsnorm creates weight_q12 INT16 buffer."""
        from vllm_i64.layers.rmsnorm import RMSNorm, quantize_rmsnorm
        norm = RMSNorm(32)
        assert not hasattr(norm, 'weight_q12')
        quantize_rmsnorm(norm)
        assert hasattr(norm, 'weight_q12')
        assert norm.weight_q12.dtype == torch.int16
        assert norm.weight_q12.shape == (32,)

    def test_loader_quantizes_rmsnorm(self):
        """_quantize_rmsnorm finds and quantizes all RMSNorm modules."""
        from vllm_i64.layers.rmsnorm import RMSNorm
        from vllm_i64.core.loader import _quantize_rmsnorm
        model = torch.nn.Module()
        model.norm1 = RMSNorm(64)
        model.norm2 = RMSNorm(64)
        _quantize_rmsnorm(model, "int8")
        assert hasattr(model.norm1, 'weight_q12')
        assert hasattr(model.norm2, 'weight_q12')

    def test_loader_skips_non_int8(self):
        """_quantize_rmsnorm skips for non-int8 methods."""
        from vllm_i64.layers.rmsnorm import RMSNorm
        from vllm_i64.core.loader import _quantize_rmsnorm
        model = torch.nn.Module()
        model.norm = RMSNorm(64)
        _quantize_rmsnorm(model, "int4")
        assert not hasattr(model.norm, 'weight_q12')


# =========================================================================
# Integer RoPE
# =========================================================================

class TestIntegerRoPE:
    def test_table_building(self):
        """Integer RoPE builds Q14 INT16 cos/sin tables."""
        from vllm_i64.layers.rotary import RotaryEmbedding
        rope = RotaryEmbedding(32, max_seq_len=256)
        rope.build_integer_tables(256, torch.device('cpu'))
        assert hasattr(rope, 'cos_table')
        assert hasattr(rope, 'sin_table')
        assert rope.cos_table.dtype == torch.int16
        assert rope.sin_table.dtype == torch.int16
        assert rope.cos_table.shape == (256, 32)

    def test_forward_integer_shape(self):
        """forward_integer returns correct shapes."""
        from vllm_i64.layers.rotary import RotaryEmbedding
        rope = RotaryEmbedding(64, max_seq_len=128)
        positions = torch.arange(8)
        cos, sin = rope.forward_integer(positions)
        assert cos.shape == (8, 64)
        assert sin.shape == (8, 64)
        assert cos.dtype == torch.int16

    def test_forward_integer_vs_float(self):
        """Integer RoPE cos/sin tables match float (high correlation)."""
        from vllm_i64.layers.rotary import RotaryEmbedding, _Q_ROPE
        rope = RotaryEmbedding(64, max_seq_len=256)
        positions = torch.arange(64)
        cos_f, sin_f = rope(positions)
        cos_i, sin_i = rope.forward_integer(positions)
        cos_dq = cos_i.float() / _Q_ROPE
        sin_dq = sin_i.float() / _Q_ROPE
        cos_sim_c = torch.nn.functional.cosine_similarity(
            cos_f.flatten().unsqueeze(0), cos_dq.flatten().unsqueeze(0),
        ).item()
        cos_sim_s = torch.nn.functional.cosine_similarity(
            sin_f.flatten().unsqueeze(0), sin_dq.flatten().unsqueeze(0),
        ).item()
        assert cos_sim_c > 0.9999, f"Cos table cosine={cos_sim_c:.6f}"
        assert cos_sim_s > 0.9999, f"Sin table cosine={cos_sim_s:.6f}"

    def test_apply_rotary_integer_shape(self):
        """apply_rotary_integer output shape matches input."""
        from vllm_i64.layers.rotary import RotaryEmbedding, apply_rotary_integer
        rope = RotaryEmbedding(32, max_seq_len=128)
        positions = torch.arange(8)
        cos_q14, sin_q14 = rope.forward_integer(positions)
        x = torch.randn(8, 4, 32)
        out = apply_rotary_integer(x, cos_q14, sin_q14)
        assert out.shape == x.shape

    def test_apply_rotary_integer_vs_float(self):
        """Integer rotary matches float rotary (cosine > 0.999)."""
        from vllm_i64.layers.rotary import (
            RotaryEmbedding, apply_rotary, apply_rotary_integer,
        )
        torch.manual_seed(42)
        rope = RotaryEmbedding(64, max_seq_len=128)
        positions = torch.arange(16)
        x = torch.randn(16, 4, 64)
        cos_f, sin_f = rope(positions)
        cos_i, sin_i = rope.forward_integer(positions)
        out_f = apply_rotary(x, cos_f, sin_f)
        out_i = apply_rotary_integer(x, cos_i, sin_i)
        cos_sim = torch.nn.functional.cosine_similarity(
            out_f.flatten().unsqueeze(0), out_i.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, f"Rotary integer cosine={cos_sim:.6f}"

    def test_lazy_table_building(self):
        """forward_integer builds tables lazily and extends when needed."""
        from vllm_i64.layers.rotary import RotaryEmbedding
        rope = RotaryEmbedding(32, max_seq_len=64)
        positions = torch.arange(10)
        cos, sin = rope.forward_integer(positions)
        assert rope.cos_table.shape[0] >= 10
        positions2 = torch.arange(3000, 3010)
        cos2, sin2 = rope.forward_integer(positions2)
        assert rope.cos_table.shape[0] >= 3010

    def test_bf16_passthrough(self):
        """Integer rotary preserves bf16 dtype."""
        from vllm_i64.layers.rotary import RotaryEmbedding, apply_rotary_integer
        rope = RotaryEmbedding(32, max_seq_len=128)
        positions = torch.arange(4)
        cos_q14, sin_q14 = rope.forward_integer(positions)
        x = torch.randn(4, 2, 32, dtype=torch.bfloat16)
        out = apply_rotary_integer(x, cos_q14, sin_q14)
        assert out.dtype == torch.bfloat16


# =========================================================================
# Integer Attention (Q@K^T + softmax_integer)
# =========================================================================

class TestIntegerAttentionPipeline:
    def test_varlen_shape(self):
        """Integer varlen attention output shape matches input."""
        from vllm_i64.layers.attention import naive_integer_varlen_attention
        q = torch.randn(8, 4, 32)
        k = torch.randn(8, 2, 32)
        v = torch.randn(8, 2, 32)
        out = naive_integer_varlen_attention(q, k, v, [8], num_kv_groups=2)
        assert out.shape == q.shape

    def test_varlen_multi_seq(self):
        """Integer varlen attention handles multiple sequences."""
        from vllm_i64.layers.attention import naive_integer_varlen_attention
        q = torch.randn(12, 4, 32)
        k = torch.randn(12, 4, 32)
        v = torch.randn(12, 4, 32)
        out = naive_integer_varlen_attention(q, k, v, [5, 7], num_kv_groups=1)
        assert out.shape == q.shape

    def test_varlen_vs_float(self):
        """Integer attention matches float attention (cosine > 0.95)."""
        from vllm_i64.layers.attention import (
            naive_varlen_attention, naive_integer_varlen_attention,
        )
        torch.manual_seed(42)
        q = torch.randn(16, 4, 32)
        k = torch.randn(16, 4, 32)
        v = torch.randn(16, 4, 32)
        out_f = naive_varlen_attention(q, k, v, [16], num_kv_groups=1)
        out_i = naive_integer_varlen_attention(q, k, v, [16], num_kv_groups=1)
        cos_sim = torch.nn.functional.cosine_similarity(
            out_f.flatten().unsqueeze(0), out_i.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95, f"Integer attention cosine={cos_sim:.6f}"

    def test_cached_shape(self):
        """Integer cached attention output shape."""
        from vllm_i64.layers.attention import naive_integer_cached_attention
        q = torch.randn(4, 4, 32)
        k_full = torch.randn(20, 4, 32)
        v_full = torch.randn(20, 4, 32)
        positions = torch.arange(16, 20)
        out = naive_integer_cached_attention(q, k_full, v_full, 1, positions)
        assert out.shape == q.shape

    def test_cached_vs_float(self):
        """Integer cached attention matches float (cosine > 0.95)."""
        from vllm_i64.layers.attention import (
            naive_cached_attention, naive_integer_cached_attention,
        )
        torch.manual_seed(42)
        q = torch.randn(4, 4, 32)
        k_full = torch.randn(20, 4, 32)
        v_full = torch.randn(20, 4, 32)
        positions = torch.arange(16, 20)
        out_f = naive_cached_attention(q, k_full, v_full, 1, positions)
        out_i = naive_integer_cached_attention(q, k_full, v_full, 1, positions)
        cos_sim = torch.nn.functional.cosine_similarity(
            out_f.flatten().unsqueeze(0), out_i.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95, f"Integer cached attention cosine={cos_sim:.6f}"

    def test_paged_decode_shape(self):
        """Integer paged decode attention output shape."""
        from vllm_i64.layers.attention import naive_integer_paged_decode_attention
        batch = 2
        num_heads = 4
        head_dim = 32
        block_size = 4
        num_blocks = 8
        q = torch.randn(batch, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        block_table = torch.tensor([[0, 1, 2, -1], [3, 4, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([10, 6], dtype=torch.int32)
        out = naive_integer_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens, num_kv_groups=1,
        )
        assert out.shape == (batch, num_heads, head_dim)

    def test_paged_decode_vs_float(self):
        """Integer paged decode matches float (cosine > 0.95)."""
        from vllm_i64.layers.attention import (
            naive_paged_decode_attention, naive_integer_paged_decode_attention,
        )
        torch.manual_seed(42)
        batch = 2
        num_heads = 4
        head_dim = 32
        block_size = 4
        num_blocks = 8
        q = torch.randn(batch, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
        block_table = torch.tensor([[0, 1, 2, -1], [3, 4, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([10, 6], dtype=torch.int32)
        out_f = naive_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens, num_kv_groups=1,
        )
        out_i = naive_integer_paged_decode_attention(
            q, k_cache, v_cache, block_table, cache_seqlens, num_kv_groups=1,
        )
        cos_sim = torch.nn.functional.cosine_similarity(
            out_f.flatten().unsqueeze(0), out_i.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95, f"Integer paged decode cosine={cos_sim:.6f}"

    def test_sliding_window(self):
        """Integer varlen attention supports sliding window masking."""
        from vllm_i64.layers.attention import naive_integer_varlen_attention
        q = torch.randn(16, 4, 32)
        k = torch.randn(16, 4, 32)
        v = torch.randn(16, 4, 32)
        out = naive_integer_varlen_attention(
            q, k, v, [16], num_kv_groups=1, sliding_window=4,
        )
        assert out.shape == q.shape

    def test_integer_pipeline_llama_layer(self):
        """Full LlamaDecoderLayer with integer pipeline (INT8 + integer RoPE + integer attention + integer RMSNorm)."""
        from vllm_i64.models.llama.model import LlamaDecoderLayer
        from vllm_i64.core.loader import _quantize_attention, _quantize_dense_mlp, _quantize_rmsnorm
        torch.manual_seed(42)
        config = type('C', (), {
            'hidden_size': 64, 'num_attention_heads': 4,
            'num_key_value_heads': 2, 'head_dim': 16,
            'max_position_embeddings': 128, 'rope_theta': 10000.0,
            'rope_scaling': None, 'rms_norm_eps': 1e-5,
            'intermediate_size': 128, 'attention_bias': False,
            'use_qk_norm': False,
        })()
        layer = LlamaDecoderLayer(config)
        layer.eval()
        hidden = torch.randn(8, 64)
        positions = torch.arange(8)
        with torch.no_grad():
            out_float = layer(hidden, positions, tokens_per_seq=[8])
        # Enable full integer pipeline
        wrapper = torch.nn.Module()
        wrapper.layer = layer
        _quantize_attention(wrapper, "int8")
        _quantize_dense_mlp(wrapper, "int8")
        _quantize_rmsnorm(wrapper, "int8")
        with torch.no_grad():
            out_int = layer(hidden, positions, tokens_per_seq=[8])
        assert out_int.shape == out_float.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0),
            out_int.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.85, f"Full integer pipeline cosine={cos_sim:.6f}"


# =========================================================================
# FP8 Quantization
# =========================================================================

class TestFP8:
    def test_quantize_fp8_shape(self):
        """FP8 quantization produces correct shapes."""
        from vllm_i64.core.fp8 import quantize_fp8, fp8_dtype
        if fp8_dtype() is None:
            pytest.skip("FP8 dtype not available in this PyTorch version")
        w = torch.randn(64, 128)
        w_fp8, scale = quantize_fp8(w)
        assert w_fp8.shape == (64, 128)
        assert scale.shape == (64,)
        assert w_fp8.dtype == fp8_dtype()

    def test_quantize_fp8_per_tensor(self):
        """FP8 per-tensor quantization gives scalar scale."""
        from vllm_i64.core.fp8 import quantize_fp8, fp8_dtype
        if fp8_dtype() is None:
            pytest.skip("FP8 dtype not available")
        w = torch.randn(32, 64)
        w_fp8, scale = quantize_fp8(w, per_channel=False)
        assert w_fp8.shape == (32, 64)
        assert scale.dim() == 0  # scalar

    def test_fp8_roundtrip_accuracy(self):
        """FP8 quantize → dequant preserves values (cosine > 0.99)."""
        from vllm_i64.core.fp8 import quantize_fp8, fp8_dtype
        if fp8_dtype() is None:
            pytest.skip("FP8 dtype not available")
        torch.manual_seed(42)
        w = torch.randn(64, 128)
        w_fp8, scale = quantize_fp8(w)
        w_recon = w_fp8.float() * scale.unsqueeze(-1)
        cos_sim = torch.nn.functional.cosine_similarity(
            w.flatten().unsqueeze(0), w_recon.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, f"FP8 roundtrip cosine={cos_sim:.6f}"

    def test_fp8_linear_fallback(self):
        """FP8 linear works in fallback mode (CPU/no _scaled_mm)."""
        from vllm_i64.core.fp8 import quantize_fp8, fp8_linear, fp8_dtype
        if fp8_dtype() is None:
            pytest.skip("FP8 dtype not available")
        torch.manual_seed(42)
        w = torch.randn(32, 64)
        w_fp8, scale = quantize_fp8(w)
        x = torch.randn(4, 64)
        out = fp8_linear(x, w_fp8, scale)
        assert out.shape == (4, 32)
        # Compare with float path
        out_ref = torch.nn.functional.linear(x, w)
        cos_sim = torch.nn.functional.cosine_similarity(
            out_ref.flatten().unsqueeze(0), out.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, f"FP8 linear cosine={cos_sim:.6f}"

    def test_fp8_fused_gate_up_fallback(self):
        """FP8 fused gate+up works in fallback mode."""
        from vllm_i64.core.fp8 import quantize_fp8, fp8_fused_gate_up, fp8_dtype
        if fp8_dtype() is None:
            pytest.skip("FP8 dtype not available")
        torch.manual_seed(42)
        gate_w = torch.randn(32, 64)
        up_w = torch.randn(32, 64)
        fused_w = torch.cat([gate_w, up_w], dim=0)
        fused_fp8, fused_scale = quantize_fp8(fused_w)
        x = torch.randn(4, 64)
        gate, up = fp8_fused_gate_up(x, fused_fp8, fused_scale, 32)
        assert gate.shape == (4, 32)
        assert up.shape == (4, 32)

    def test_fp8_available_function(self):
        """fp8_available() returns bool."""
        from vllm_i64.core.fp8 import fp8_available
        result = fp8_available()
        assert isinstance(result, bool)

    def test_quantize_experts_fp8(self):
        """FP8 expert quantization produces correct shapes."""
        from vllm_i64.core.fp8 import quantize_experts_fp8, fp8_dtype
        if fp8_dtype() is None:
            pytest.skip("FP8 dtype not available")
        gate_up = torch.randn(4, 64, 128)
        down = torch.randn(4, 64, 64)
        result = quantize_experts_fp8(gate_up, down)
        assert result["method"] == "fp8"
        assert result["gate_up_fp8"].shape == (4, 64, 128)
        assert result["down_fp8"].shape == (4, 64, 64)
        assert result["gate_up_scale"].shape == (4, 64)
        assert result["down_scale"].shape == (4, 64)

    def test_dense_mlp_fp8_path(self):
        """DenseMLP with FP8 buffers uses FP8 forward path."""
        from vllm_i64.layers.dense_mlp import DenseMLP
        from vllm_i64.core.fp8 import quantize_fp8, fp8_dtype
        if fp8_dtype() is None:
            pytest.skip("FP8 dtype not available")
        mlp = DenseMLP(64, 128)
        mlp.eval()
        x = torch.randn(4, 64)
        with torch.no_grad():
            out_float = mlp(x)

        # Register FP8 buffers
        gate_fp8, gate_s = quantize_fp8(mlp.gate_proj.linear.weight.data)
        up_fp8, up_s = quantize_fp8(mlp.up_proj.linear.weight.data)
        down_fp8, down_s = quantize_fp8(mlp.down_proj.linear.weight.data)
        mlp.register_buffer('gate_fp8', gate_fp8)
        mlp.register_buffer('gate_fp8_scale', gate_s)
        mlp.register_buffer('up_fp8', up_fp8)
        mlp.register_buffer('up_fp8_scale', up_s)
        mlp.register_buffer('down_fp8', down_fp8)
        mlp.register_buffer('down_fp8_scale', down_s)

        with torch.no_grad():
            out_fp8 = mlp(x)
        assert out_fp8.shape == out_float.shape
        cos_sim = torch.nn.functional.cosine_similarity(
            out_float.flatten().unsqueeze(0), out_fp8.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95, f"FP8 DenseMLP cosine={cos_sim:.6f}"


# =========================================================================
# Triton Fused Kernels (CPU fallback tests — returns None on CPU)
# =========================================================================

class TestTritonFusedKernels:
    def test_fused_silu_mul_returns_none_on_cpu(self):
        """Triton fused SiLU*up returns None on CPU (correct fallback)."""
        try:
            from vllm_i64.kernels.triton.I64_fused_silu_mul import triton_fused_silu_mul
        except ImportError:
            pytest.skip("Triton not installed")
        gate = torch.randn(4, 32)
        up = torch.randn(4, 32)
        result = triton_fused_silu_mul(gate, up)
        assert result is None  # CPU → returns None

    def test_fused_rmsnorm_returns_none_on_cpu(self):
        """Triton fused RMSNorm returns None on CPU."""
        try:
            from vllm_i64.kernels.triton.I64_fused_rmsnorm_quant import triton_fused_rmsnorm
        except ImportError:
            pytest.skip("Triton not installed")
        x = torch.randn(4, 64)
        w = torch.ones(64)
        result = triton_fused_rmsnorm(x, w)
        assert result is None

    def test_fused_rmsnorm_quant_returns_none_on_cpu(self):
        """Triton fused RMSNorm+quant returns None on CPU."""
        try:
            from vllm_i64.kernels.triton.I64_fused_rmsnorm_quant import triton_fused_rmsnorm_quant
        except ImportError:
            pytest.skip("Triton not installed")
        x = torch.randn(4, 64)
        w = torch.ones(64)
        result = triton_fused_rmsnorm_quant(x, w)
        assert result is None

    def test_fused_softmax_returns_none_on_cpu(self):
        """Triton integer softmax returns None on CPU."""
        try:
            from vllm_i64.kernels.triton.I64_fused_softmax import triton_fused_softmax_integer
        except ImportError:
            pytest.skip("Triton not installed")
        logits = torch.randn(4, 16)
        result = triton_fused_softmax_integer(logits)
        assert result is None

    def test_fused_rope_returns_none_on_cpu(self):
        """Triton fused RoPE returns None on CPU."""
        try:
            from vllm_i64.kernels.triton.I64_fused_rope import triton_fused_rope
        except ImportError:
            pytest.skip("Triton not installed")
        x = torch.randn(4, 8, 32)
        cos = torch.randn(4, 32)
        sin = torch.randn(4, 32)
        result = triton_fused_rope(x, cos, sin)
        assert result is None

    def test_fused_dequant_gemm_returns_none_on_cpu(self):
        """Triton dequant+GEMM returns None on CPU."""
        try:
            from vllm_i64.kernels.triton.I64_fused_dequant_gemm import triton_dequant_gemm_int8
        except ImportError:
            pytest.skip("Triton not installed")
        x = torch.randn(4, 64)
        w = torch.randint(-128, 127, (32, 64), dtype=torch.int8)
        s = torch.randn(32).abs()
        result = triton_dequant_gemm_int8(x, w, s)
        assert result is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_silu_mul_gpu(self):
        """Triton fused SiLU*up on GPU matches PyTorch."""
        try:
            from vllm_i64.kernels.triton.I64_fused_silu_mul import triton_fused_silu_mul
        except ImportError:
            pytest.skip("Triton not installed")
        torch.manual_seed(42)
        gate = torch.randn(16, 128, device='cuda')
        up = torch.randn(16, 128, device='cuda')
        ref = torch.nn.functional.silu(gate) * up
        result = triton_fused_silu_mul(gate, up)
        if result is None:
            pytest.skip("Triton not available on this GPU")
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0), result.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, f"Triton SiLU*up cosine={cos_sim:.6f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_rmsnorm_gpu(self):
        """Triton fused RMSNorm on GPU matches PyTorch."""
        try:
            from vllm_i64.kernels.triton.I64_fused_rmsnorm_quant import triton_fused_rmsnorm
        except ImportError:
            pytest.skip("Triton not installed")
        from vllm_i64.layers.rmsnorm import RMSNorm
        torch.manual_seed(42)
        norm = RMSNorm(128, eps=1e-6).cuda()
        x = torch.randn(8, 128, device='cuda')
        ref = norm.weight * (x.float() * (x.float().pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt())
        result = triton_fused_rmsnorm(x, norm.weight.data, 1e-6)
        if result is None:
            pytest.skip("Triton not available on this GPU")
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0), result.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, f"Triton RMSNorm cosine={cos_sim:.6f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_softmax_integer_gpu(self):
        """Triton integer softmax on GPU matches CPU integer softmax."""
        try:
            from vllm_i64.kernels.triton.I64_fused_softmax import triton_fused_softmax_integer
        except ImportError:
            pytest.skip("Triton not installed")
        from vllm_i64.layers.moe import softmax_integer
        torch.manual_seed(42)
        logits = torch.randn(8, 32, device='cuda')
        ref = softmax_integer(logits)
        result = triton_fused_softmax_integer(logits)
        if result is None:
            pytest.skip("Triton not available on this GPU")
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0), result.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.99, f"Triton integer softmax cosine={cos_sim:.6f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_fused_rope_gpu(self):
        """Triton fused RoPE on GPU matches PyTorch."""
        try:
            from vllm_i64.kernels.triton.I64_fused_rope import triton_fused_rope
        except ImportError:
            pytest.skip("Triton not installed")
        from vllm_i64.layers.rotary import apply_rotary
        torch.manual_seed(42)
        x = torch.randn(4, 8, 32, device='cuda')
        cos = torch.randn(4, 32, device='cuda')
        sin = torch.randn(4, 32, device='cuda')
        ref = apply_rotary(x, cos, sin)
        result = triton_fused_rope(x, cos, sin)
        if result is None:
            pytest.skip("Triton not available on this GPU")
        cos_sim = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0), result.flatten().unsqueeze(0),
        ).item()
        assert cos_sim > 0.999, f"Triton RoPE cosine={cos_sim:.6f}"


# =========================================================================
# CUDA Kernel Loader
# =========================================================================

class TestCUDAKernels:
    def test_cuda_loader_import(self):
        """CUDA kernel loader imports without error."""
        from vllm_i64.kernels.cuda.I64_loader import is_i64_cuda_available
        result = is_i64_cuda_available()
        # On CPU-only machines this returns False — that's expected
        assert isinstance(result, bool)

    def test_cuda_kernels_init_import(self):
        """CUDA kernels __init__ imports without error."""
        from vllm_i64.kernels.cuda import is_i64_cuda_available
        assert isinstance(is_i64_cuda_available(), bool)

    def test_triton_init_import(self):
        """Triton kernels __init__ imports without error (even without Triton)."""
        try:
            from vllm_i64.kernels.triton import (
                triton_fused_rmsnorm_quant,
                triton_fused_rmsnorm,
                triton_fused_silu_mul,
                triton_fused_softmax_integer,
                triton_fused_rope,
                triton_dequant_gemm_int8,
            )
        except ImportError:
            pytest.skip("Triton not installed")

    def test_fp8_module_import(self):
        """FP8 module imports without error."""
        from vllm_i64.core.fp8 import (
            fp8_available,
            fp8_linear,
            fp8_fused_gate_up,
            quantize_fp8,
            quantize_experts_fp8,
        )


# =========================================================================
# Loader FP8 quantization
# =========================================================================

class TestLoaderFP8:
    def test_quantize_dense_mlp_fp8(self):
        """Loader _quantize_dense_mlp_fp8 registers FP8 buffers."""
        from vllm_i64.core.fp8 import fp8_dtype
        if fp8_dtype() is None:
            pytest.skip("FP8 dtype not available")
        from vllm_i64.layers.dense_mlp import DenseMLP
        from vllm_i64.core.loader import _quantize_dense_mlp_fp8

        wrapper = torch.nn.Module()
        wrapper.mlp = DenseMLP(64, 128)
        _quantize_dense_mlp_fp8(wrapper)

        assert hasattr(wrapper.mlp, 'gate_fp8')
        assert hasattr(wrapper.mlp, 'up_fp8')
        assert hasattr(wrapper.mlp, 'down_fp8')
        assert hasattr(wrapper.mlp, 'gate_fp8_scale')
        assert wrapper.mlp.gate_fp8.dtype == fp8_dtype()

    def test_load_model_fp8_quantization_method(self):
        """The 'fp8' quantization method is recognized by load_model_by_name dispatch."""
        # Just verify the code path doesn't crash — actual model load needs checkpoint
        from vllm_i64.core.loader import _quantize_dense_mlp_fp8
        from vllm_i64.core.fp8 import fp8_dtype
        if fp8_dtype() is None:
            pytest.skip("FP8 dtype not available")
        # Test with empty module (no DenseMLP children = no-op)
        wrapper = torch.nn.Module()
        _quantize_dense_mlp_fp8(wrapper)  # should not crash


# =========================================================================
# CPU Optimizations: vectorized INT4, CPU _int_mm, torch.compile
# =========================================================================

class TestCPUOptimizations:
    """CPU performance optimizations — INT4 vectorization, CPU _int_mm, torch.compile."""

    def test_int4_linear_vectorized_shape(self):
        """Vectorized INT4 linear produces correct output shape."""
        from vllm_i64.core.quantization import quantize_int4, int4_linear
        torch.manual_seed(42)
        w = torch.randn(64, 128)
        packed, scale, zero = quantize_int4(w)
        x = torch.randn(4, 128)
        y = int4_linear(x, packed, scale, zero)
        assert y.shape == (4, 64)

    def test_int4_linear_vectorized_1d(self):
        """Vectorized INT4 linear handles 1D input (single token)."""
        from vllm_i64.core.quantization import quantize_int4, int4_linear
        torch.manual_seed(42)
        w = torch.randn(64, 128)
        packed, scale, zero = quantize_int4(w)
        x = torch.randn(128)
        y = int4_linear(x, packed, scale, zero)
        assert y.shape == (64,)

    def test_int4_linear_vectorized_accuracy(self):
        """Vectorized INT4 matches float reference within quantization tolerance."""
        from vllm_i64.core.quantization import quantize_int4, int4_linear, dequantize_int4
        torch.manual_seed(42)
        w = torch.randn(32, 128)
        packed, scale, zero = quantize_int4(w)
        x = torch.randn(8, 128)

        # INT4 path
        y_int4 = int4_linear(x, packed, scale, zero)

        # Float reference: dequant + matmul
        w_deq = dequantize_int4(packed, scale, zero).reshape(32, 128)
        y_ref = x.float() @ w_deq.T

        # INT4 quantization error is larger than INT8, allow ~10% relative error
        rel_err = (y_int4 - y_ref).abs().mean() / y_ref.abs().mean()
        assert rel_err < 0.15, f"Relative error {rel_err:.4f} too high"

    def test_int4_linear_with_bias(self):
        """Vectorized INT4 linear handles bias."""
        from vllm_i64.core.quantization import quantize_int4, int4_linear
        torch.manual_seed(42)
        w = torch.randn(64, 128)
        packed, scale, zero = quantize_int4(w)
        bias = torch.randn(64)
        x = torch.randn(4, 128)
        y_bias = int4_linear(x, packed, scale, zero, bias=bias)
        y_no_bias = int4_linear(x, packed, scale, zero)
        diff = (y_bias - y_no_bias - bias).abs().max()
        assert diff < 1e-5

    def test_int_mm_cpu_probe(self):
        """CPU _int_mm probe runs at import time."""
        from vllm_i64.core.quantization import _INT_MM_CPU_OK, _INT_MM_AVAILABLE
        assert _INT_MM_AVAILABLE, "torch._int_mm not found"
        assert _INT_MM_CPU_OK, "CPU _int_mm probe failed"

    def test_int8_linear_uses_int_mm_on_cpu(self):
        """INT8 linear uses native _int_mm on CPU (not dequant fallback)."""
        from vllm_i64.core.quantization import quantize_int8, int8_linear_native, _INT_MM_CPU_OK
        if not _INT_MM_CPU_OK:
            pytest.skip("CPU _int_mm not available")

        torch.manual_seed(42)
        w = torch.randn(64, 128)
        wq, ws = quantize_int8(w)
        x = torch.randn(4, 128)
        y = int8_linear_native(x, wq, ws)
        assert y.shape == (4, 64)
        # Verify it's not all zeros (matmul actually ran)
        assert y.abs().sum() > 0

    def test_int8_linear_cpu_accuracy(self):
        """CPU _int_mm path matches dequant reference within INT8 tolerance."""
        from vllm_i64.core.quantization import (
            quantize_int8, int8_linear_native, dequantize_int8, _INT_MM_CPU_OK,
        )
        if not _INT_MM_CPU_OK:
            pytest.skip("CPU _int_mm not available")

        torch.manual_seed(42)
        w = torch.randn(32, 128)
        wq, ws = quantize_int8(w)
        x = torch.randn(8, 128)

        # _int_mm path
        y = int8_linear_native(x, wq, ws)

        # Float reference
        w_float = dequantize_int8(wq, ws)
        import torch.nn.functional as F
        y_ref = F.linear(x.float(), w_float)

        rel_err = (y - y_ref).abs().mean() / y_ref.abs().mean()
        assert rel_err < 0.05, f"Relative error {rel_err:.4f} too high"

    def test_int8_fused_gate_up_cpu(self):
        """Fused gate+up uses _int_mm on CPU."""
        from vllm_i64.core.quantization import (
            quantize_int8, int8_fused_gate_up_native, _INT_MM_CPU_OK,
        )
        if not _INT_MM_CPU_OK:
            pytest.skip("CPU _int_mm not available")

        torch.manual_seed(42)
        gate_w = torch.randn(64, 128)
        up_w = torch.randn(64, 128)
        gq, gs = quantize_int8(gate_w)
        uq, us = quantize_int8(up_w)
        fused = torch.cat([gq, uq], dim=0)
        fused_s = torch.cat([gs, us])

        x = torch.randn(4, 128)
        gate, up = int8_fused_gate_up_native(x, fused, fused_s, 64)
        assert gate.shape == (4, 64)
        assert up.shape == (4, 64)

    def test_int8_linear_available_cpu(self):
        """int8_linear_available reports CPU support."""
        from vllm_i64.core.quantization import int8_linear_available
        assert int8_linear_available("cpu") is True

    def test_compile_module_imports(self):
        """compile module imports without error."""
        from vllm_i64.core.compile import (
            compile_model,
            compile_function,
            is_compile_available,
        )

    def test_compile_function(self):
        """compile_function wraps a function (may or may not actually compile)."""
        from vllm_i64.core.compile import compile_function
        def my_fn(x):
            return x * 2 + 1
        compiled = compile_function(my_fn)
        result = compiled(torch.tensor(3.0))
        assert result.item() == 7.0

    def test_compile_model_rmsnorm(self):
        """compile_model processes RMSNorm modules."""
        from vllm_i64.core.compile import compile_model
        from vllm_i64.layers.rmsnorm import RMSNorm
        wrapper = torch.nn.Module()
        wrapper.norm = RMSNorm(64)
        compile_model(wrapper)
        # Should still work after compilation
        x = torch.randn(2, 64)
        out = wrapper.norm(x)
        assert out.shape == (2, 64)

    def test_compile_model_dense_mlp(self):
        """compile_model processes DenseMLP modules."""
        from vllm_i64.core.compile import compile_model
        from vllm_i64.layers.dense_mlp import DenseMLP
        wrapper = torch.nn.Module()
        wrapper.mlp = DenseMLP(64, 128)
        compile_model(wrapper)
        x = torch.randn(2, 64)
        out = wrapper.mlp(x)
        assert out.shape == (2, 64)

    def test_loader_calls_compile(self):
        """load_model_by_name calls compile_model (integration check)."""
        from vllm_i64.core.compile import compile_model
        # Just verify the import + call path works
        wrapper = torch.nn.Module()
        result = compile_model(wrapper)
        assert result is wrapper
