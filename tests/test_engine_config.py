"""
Tests for EngineConfig, I64Sampler, and cmd_estimate.
"""

import pytest
import torch
import numpy as np
import json
import sys
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from pathlib import Path

from vllm_i64.engine.config import EngineConfig
from vllm_i64.engine.sampler import I64Sampler
from vllm_i64.core.sampling import SamplingParams


# =========================================================================
# EngineConfig
# =========================================================================

class TestEngineConfig:
    def test_defaults(self):
        cfg = EngineConfig()
        assert cfg.num_experts == 4
        assert cfg.hidden_dim == 768
        assert cfg.vocab_size == 100_000
        assert cfg.max_batch_size == 32
        assert cfg.max_seq_len == 2048
        assert cfg.max_prefill_tokens == 512
        assert cfg.max_kv_blocks == 0
        assert cfg.enable_prefix_caching is True
        assert cfg.kv_cache_dtype is None
        assert cfg.device == "cuda"
        assert cfg.default_timeout_s == 300.0
        assert cfg.enable_swap is False
        assert cfg.enable_merge is False

    def test_custom_values(self):
        cfg = EngineConfig(
            num_experts=8,
            hidden_dim=4096,
            max_batch_size=64,
            device="cpu",
            enable_prefix_caching=False,
        )
        assert cfg.num_experts == 8
        assert cfg.hidden_dim == 4096
        assert cfg.max_batch_size == 64
        assert cfg.device == "cpu"
        assert cfg.enable_prefix_caching is False

    def test_resolve_kv_blocks_auto(self):
        cfg = EngineConfig(max_kv_blocks=0, max_batch_size=32)
        assert cfg.resolve_kv_blocks() == 256  # max(256, 32*8=256)

        cfg2 = EngineConfig(max_kv_blocks=0, max_batch_size=64)
        assert cfg2.resolve_kv_blocks() == 512  # max(256, 64*8=512)

        cfg3 = EngineConfig(max_kv_blocks=0, max_batch_size=16)
        assert cfg3.resolve_kv_blocks() == 256  # max(256, 16*8=128) = 256

    def test_resolve_kv_blocks_explicit(self):
        cfg = EngineConfig(max_kv_blocks=1024)
        assert cfg.resolve_kv_blocks() == 1024

    def test_resolve_kv_blocks_negative_treated_as_auto(self):
        cfg = EngineConfig(max_kv_blocks=-1, max_batch_size=32)
        assert cfg.resolve_kv_blocks() == 256


# =========================================================================
# I64Sampler
# =========================================================================

class TestI64Sampler:
    def test_default_params_greedy(self):
        sampler = I64Sampler()
        assert sampler.default_params.temperature == 0.0

    def test_custom_default_params(self):
        params = SamplingParams(temperature=0.8, top_k=50)
        sampler = I64Sampler(default_params=params)
        assert sampler.default_params.temperature == 0.8
        assert sampler.default_params.top_k == 50

    def test_sample_greedy(self):
        sampler = I64Sampler()
        # Create logits where token 42 has highest score
        logits = torch.zeros(1, 100)
        logits[0, 42] = 10.0
        result = sampler.sample(logits)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int64
        assert result[0] == 42

    def test_sample_batch(self):
        sampler = I64Sampler()
        batch_size = 4
        vocab = 100
        logits = torch.zeros(batch_size, vocab)
        # Each item in batch should pick a different token
        for i in range(batch_size):
            logits[i, i * 10] = 10.0
        result = sampler.sample(logits)
        assert result.shape == (batch_size,)
        for i in range(batch_size):
            assert result[i] == i * 10

    def test_sample_with_override_params(self):
        sampler = I64Sampler()  # default greedy
        logits = torch.zeros(1, 100)
        logits[0, 42] = 10.0
        # Override with greedy too — should still pick 42
        override = SamplingParams(temperature=0.0)
        result = sampler.sample(logits, params=override)
        assert result[0] == 42

    def test_sample_output_is_i64(self):
        sampler = I64Sampler()
        logits = torch.randn(3, 50)
        result = sampler.sample(logits)
        assert result.dtype == np.int64
        assert result.shape == (3,)
        # All token IDs should be valid indices
        assert all(0 <= t < 50 for t in result)

    def test_sample_with_logprobs(self):
        sampler = I64Sampler()
        logits = torch.zeros(1, 100)
        logits[0, 42] = 10.0
        result = sampler.sample_with_logprobs(logits)
        # Should return something with token_ids
        assert hasattr(result, 'token_ids') or isinstance(result, tuple)


# =========================================================================
# cmd_estimate
# =========================================================================

class TestCmdEstimate:
    def test_estimate_with_config_file(self, tmp_path):
        """Test estimate reads config.json and produces output."""
        config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "vocab_size": 32000,
            "intermediate_size": 3072,
            "num_experts": 1,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        args = SimpleNamespace(
            checkpoint=str(tmp_path),
            model="llama",
            dtype="float16",
            kv_dtype=None,
            max_batch_size=8,
            max_seq_len=2048,
        )

        from vllm_i64.cli import cmd_estimate
        # Should run without error and print output
        with patch('builtins.print') as mock_print:
            cmd_estimate(args)
        # Verify it printed something about memory
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "GB" in output
        assert "Weights" in output

    def test_estimate_no_config(self, tmp_path):
        """Test estimate exits when no config found."""
        args = SimpleNamespace(
            checkpoint=str(tmp_path / "nonexistent"),
            model="llama",
            dtype="float16",
            kv_dtype=None,
            max_batch_size=8,
            max_seq_len=2048,
        )
        from vllm_i64.cli import cmd_estimate
        with pytest.raises(SystemExit):
            cmd_estimate(args)

    def test_estimate_moe_model(self, tmp_path):
        """Test estimate with MoE model (num_experts > 1)."""
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 100000,
            "intermediate_size": 14336,
            "num_experts": 8,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        args = SimpleNamespace(
            checkpoint=str(tmp_path),
            model="pacific-prime",
            dtype="float16",
            kv_dtype="fp8",
            max_batch_size=32,
            max_seq_len=4096,
        )

        from vllm_i64.cli import cmd_estimate
        with patch('builtins.print') as mock_print:
            cmd_estimate(args)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Experts" in output
        assert "8" in output
