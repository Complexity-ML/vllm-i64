"""
vllm-i64 :: Test Safetensors Loading

Verifies multi-format checkpoint loading:
  - .safetensors single file
  - .safetensors sharded with index.json
  - .pt PyTorch files
  - Auto-detection from directory
  - Backward compatibility with existing load_checkpoint

Run:
    python -m pytest tests/test_safetensors_loader.py -v

INL - 2025
"""

import pytest
import torch
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import save_file

from vllm_i64.core.loader import (
    _load_safetensors_file,
    _load_pytorch_file,
    _load_sharded_safetensors,
    _load_from_directory,
    _load_state_dict,
)


@pytest.fixture
def sample_state_dict():
    """Small state dict for testing."""
    return {
        "embed_tokens.weight": torch.randn(100, 64),
        "layers.0.input_layernorm.weight": torch.randn(64),
        "layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        "layers.0.self_attn.k_proj.weight": torch.randn(32, 64),
        "layers.0.self_attn.v_proj.weight": torch.randn(32, 64),
        "layers.0.self_attn.o_proj.weight": torch.randn(64, 64),
        "norm.weight": torch.randn(64),
    }


class TestSafetensorsFile:

    def test_load_single_safetensors(self, tmp_path, sample_state_dict):
        """Load a single .safetensors file."""
        path = tmp_path / "model.safetensors"
        save_file(sample_state_dict, str(path))

        loaded = _load_safetensors_file(str(path))
        assert set(loaded.keys()) == set(sample_state_dict.keys())
        for key in sample_state_dict:
            assert torch.allclose(loaded[key], sample_state_dict[key])

    def test_load_pytorch_file(self, tmp_path, sample_state_dict):
        """Load a .pt file."""
        path = tmp_path / "model.pt"
        torch.save(sample_state_dict, str(path))

        loaded = _load_pytorch_file(str(path))
        assert set(loaded.keys()) == set(sample_state_dict.keys())

    def test_load_pytorch_unwraps_nested(self, tmp_path, sample_state_dict):
        """PyTorch loader unwraps nested 'model' key."""
        path = tmp_path / "ckpt.pt"
        torch.save({"model": sample_state_dict}, str(path))

        loaded = _load_pytorch_file(str(path))
        assert "embed_tokens.weight" in loaded

    def test_load_pytorch_unwraps_state_dict(self, tmp_path, sample_state_dict):
        """PyTorch loader unwraps nested 'state_dict' key."""
        path = tmp_path / "ckpt.pt"
        torch.save({"state_dict": sample_state_dict}, str(path))

        loaded = _load_pytorch_file(str(path))
        assert "embed_tokens.weight" in loaded


class TestShardedSafetensors:

    def test_load_sharded(self, tmp_path, sample_state_dict):
        """Load sharded safetensors with index.json."""
        # Split into 2 shards
        keys = list(sample_state_dict.keys())
        mid = len(keys) // 2
        shard1_keys = keys[:mid]
        shard2_keys = keys[mid:]

        shard1 = {k: sample_state_dict[k] for k in shard1_keys}
        shard2 = {k: sample_state_dict[k] for k in shard2_keys}

        save_file(shard1, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(tmp_path / "model-00002-of-00002.safetensors"))

        # Create index
        weight_map = {}
        for k in shard1_keys:
            weight_map[k] = "model-00001-of-00002.safetensors"
        for k in shard2_keys:
            weight_map[k] = "model-00002-of-00002.safetensors"

        index = {"metadata": {}, "weight_map": weight_map}
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        loaded = _load_sharded_safetensors(tmp_path)
        assert set(loaded.keys()) == set(sample_state_dict.keys())
        for key in sample_state_dict:
            assert torch.allclose(loaded[key], sample_state_dict[key])

    def test_missing_shard_raises(self, tmp_path):
        """Missing shard file raises FileNotFoundError."""
        index = {"weight_map": {"w": "missing.safetensors"}}
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        with pytest.raises(FileNotFoundError):
            _load_sharded_safetensors(tmp_path)


class TestDirectoryLoading:

    def test_directory_prefers_sharded_index(self, tmp_path, sample_state_dict):
        """Directory loading prioritizes index.json."""
        # Create both single and sharded
        save_file(sample_state_dict, str(tmp_path / "model.safetensors"))

        # Also create sharded with index
        save_file(sample_state_dict, str(tmp_path / "model-00001-of-00001.safetensors"))
        weight_map = {k: "model-00001-of-00001.safetensors" for k in sample_state_dict}
        index = {"weight_map": weight_map}
        with open(tmp_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        loaded = _load_from_directory(tmp_path)
        assert set(loaded.keys()) == set(sample_state_dict.keys())

    def test_directory_single_safetensors(self, tmp_path, sample_state_dict):
        """Directory loading finds model.safetensors."""
        save_file(sample_state_dict, str(tmp_path / "model.safetensors"))

        loaded = _load_from_directory(tmp_path)
        assert set(loaded.keys()) == set(sample_state_dict.keys())

    def test_directory_glob_safetensors(self, tmp_path, sample_state_dict):
        """Directory loading globs *.safetensors."""
        save_file(sample_state_dict, str(tmp_path / "weights.safetensors"))

        loaded = _load_from_directory(tmp_path)
        assert set(loaded.keys()) == set(sample_state_dict.keys())

    def test_directory_pytorch_fallback(self, tmp_path, sample_state_dict):
        """Directory loading falls back to .pt files."""
        torch.save(sample_state_dict, str(tmp_path / "model.pt"))

        loaded = _load_from_directory(tmp_path)
        assert set(loaded.keys()) == set(sample_state_dict.keys())

    def test_empty_directory_raises(self, tmp_path):
        """Empty directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_from_directory(tmp_path)


class TestAutoDetect:

    def test_auto_detect_safetensors(self, tmp_path, sample_state_dict):
        """Auto-detect .safetensors extension."""
        path = tmp_path / "model.safetensors"
        save_file(sample_state_dict, str(path))

        loaded = _load_state_dict(str(path))
        assert "embed_tokens.weight" in loaded

    def test_auto_detect_pt(self, tmp_path, sample_state_dict):
        """Auto-detect .pt extension."""
        path = tmp_path / "model.pt"
        torch.save(sample_state_dict, str(path))

        loaded = _load_state_dict(str(path))
        assert "embed_tokens.weight" in loaded

    def test_auto_detect_directory(self, tmp_path, sample_state_dict):
        """Auto-detect directory."""
        save_file(sample_state_dict, str(tmp_path / "model.safetensors"))

        loaded = _load_state_dict(str(tmp_path))
        assert "embed_tokens.weight" in loaded

    def test_missing_file_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_state_dict("/nonexistent/path/model.safetensors")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
