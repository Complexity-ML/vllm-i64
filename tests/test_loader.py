"""
vllm-i64 :: Test Weight Loader + Config

Tests:
  - ComplexityDeepConfig.from_json() parsing
  - ComplexityDeepConfig defaults
  - load_checkpoint with synthetic state_dict
  - TP-aware loading (ColumnParallel/RowParallel detection)
  - Tied embedding handling
  - Registry lookup

INL - 2025
"""

import torch
import pytest
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
from vllm_i64.models.complexity_deep.model import ComplexityDeepModel
from vllm_i64.core.loader import load_checkpoint, _get_module_for_param
from vllm_i64.parallel.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class TestComplexityDeepConfig:
    def test_defaults(self):
        config = ComplexityDeepConfig()
        assert config.vocab_size == 32000
        assert config.hidden_size == 2048
        assert config.num_experts == 4
        assert config.head_dim == 128  # 2048 / 16

    def test_head_dim_property(self):
        config = ComplexityDeepConfig(hidden_size=512, num_attention_heads=8)
        assert config.head_dim == 64

    def test_expert_intermediate_property(self):
        config = ComplexityDeepConfig(intermediate_size=5632, num_experts=4)
        assert config.expert_intermediate_size == 1408

    def test_from_json(self):
        data = {
            "model_type": "complexity-deep",
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "num_experts": 2,
            "parameters": "skip_this",
            "innovations": "skip_this_too",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            config = ComplexityDeepConfig.from_json(path)
            assert config.vocab_size == 1000
            assert config.hidden_size == 256
            assert config.num_hidden_layers == 4
            assert config.num_experts == 2
            # "parameters" and "innovations" should be skipped
        finally:
            os.unlink(path)

    def test_from_json_ignores_unknown_keys(self):
        data = {"vocab_size": 500, "some_future_field": True}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            config = ComplexityDeepConfig.from_json(path)
            assert config.vocab_size == 500
            assert not hasattr(config, "some_future_field")
        finally:
            os.unlink(path)


class TestGetModuleForParam:
    def test_finds_linear(self):
        config = ComplexityDeepConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=128, num_experts=2, vocab_size=32,
        )
        model = ComplexityDeepModel(config)
        # q_proj is ColumnParallelLinear
        module = _get_module_for_param(model, "layers.0.self_attn.q_proj.linear.weight")
        assert module is not None

    def test_returns_none_for_missing(self):
        config = ComplexityDeepConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=128, num_experts=2, vocab_size=32,
        )
        model = ComplexityDeepModel(config)
        module = _get_module_for_param(model, "nonexistent.layer.weight")
        assert module is None


class TestLoadCheckpoint:
    def _make_small_model(self):
        config = ComplexityDeepConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, intermediate_size=128, num_experts=2,
            vocab_size=32, max_position_embeddings=32,
        )
        return ComplexityDeepModel(config), config

    def test_load_synthetic_checkpoint(self):
        model, config = self._make_small_model()

        # Create a synthetic state_dict from the model itself
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            path = f.name

        try:
            stats = load_checkpoint(model, path, dtype=torch.float32)
            assert stats["loaded"] > 0
            assert stats["tp_size"] == 1
        finally:
            os.unlink(path)

    def test_tied_embeddings(self):
        model, config = self._make_small_model()

        # Simulate checkpoint with lm_head.weight (tied to embed_tokens)
        state_dict = {"lm_head.weight": torch.randn(config.vocab_size, config.hidden_size)}

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            path = f.name

        try:
            stats = load_checkpoint(model, path, dtype=torch.float32)
            # lm_head.weight should be loaded into embed_tokens.weight
            assert "lm_head.weight" not in stats.get("missing", set())
        finally:
            os.unlink(path)

    def test_missing_checkpoint_raises(self):
        model, _ = self._make_small_model()
        with pytest.raises(FileNotFoundError):
            load_checkpoint(model, "/nonexistent/path/model.pt")

    def test_skips_rotary_inv_freq(self):
        model, _ = self._make_small_model()

        state_dict = {
            "layers.0.self_attn.rope.rotary_emb.inv_freq": torch.randn(16),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            path = f.name

        try:
            stats = load_checkpoint(model, path, dtype=torch.float32)
            assert stats["skipped"] >= 1
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
