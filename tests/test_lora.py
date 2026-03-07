"""
Tests for LoRA hot-swap: LoRALinear and LoRAManager.
"""

import pytest
import torch
import torch.nn as nn

from vllm_i64.layers.lora import LoRALinear, LoRAManager


# =========================================================================
# LoRALinear
# =========================================================================

class TestLoRALinear:
    def _make(self, in_f=64, out_f=32, max_adapters=8):
        base = nn.Linear(in_f, out_f, bias=False)
        return LoRALinear(base, max_adapters=max_adapters)

    def test_forward_no_adapter(self):
        lora = self._make()
        x = torch.randn(2, 64)
        out = lora(x)
        expected = lora.base(x)
        assert torch.allclose(out, expected)

    def test_load_and_activate_adapter(self):
        lora = self._make(64, 32)
        rank = 4
        A = torch.randn(64, rank)
        B = torch.randn(rank, 32)
        lora.load_adapter(adapter_id=1, lora_A=A, lora_B=B, scaling=1.0)

        assert lora.num_adapters == 1
        assert 1 in lora.adapter_ids

        # Without activation, output == base
        x = torch.randn(2, 64)
        out_base = lora(x)
        assert torch.allclose(out_base, lora.base(x))

        # With activation, output includes LoRA delta
        lora.set_active_adapter(1)
        out_lora = lora(x)
        assert not torch.allclose(out_lora, out_base), "LoRA should change the output"

    def test_per_request_adapter_id(self):
        lora = self._make(64, 32)
        A = torch.randn(64, 4)
        B = torch.randn(4, 32)
        lora.load_adapter(1, A, B, scaling=1.0)

        x = torch.randn(1, 64)
        out_base = lora(x, adapter_id=None)
        # Nonexistent adapter_id falls back to base
        out_unknown = lora(x, adapter_id=999)
        assert torch.allclose(out_base, out_unknown)

        # Existing adapter_id uses the adapter
        out_lora = lora(x, adapter_id=1)
        assert not torch.allclose(out_base, out_lora)

    def test_multiple_adapters(self):
        lora = self._make(64, 32)
        A1 = torch.randn(64, 4)
        B1 = torch.randn(4, 32)
        A2 = torch.randn(64, 8)
        B2 = torch.randn(8, 32)

        lora.load_adapter(1, A1, B1, scaling=1.0)
        lora.load_adapter(2, A2, B2, scaling=1.0)
        assert lora.num_adapters == 2

        x = torch.randn(1, 64)
        out1 = lora(x, adapter_id=1)
        out2 = lora(x, adapter_id=2)
        assert not torch.allclose(out1, out2), "Different adapters should produce different outputs"

    def test_unload_adapter(self):
        lora = self._make(64, 32)
        A = torch.randn(64, 4)
        B = torch.randn(4, 32)
        lora.load_adapter(1, A, B, scaling=1.0)
        lora.set_active_adapter(1)
        assert lora.num_adapters == 1

        lora.unload_adapter(1)
        assert lora.num_adapters == 0
        assert lora._active_adapter_id is None

        # After unload, output == base
        x = torch.randn(1, 64)
        assert torch.allclose(lora(x), lora.base(x))

    def test_max_adapters_limit(self):
        lora = self._make(64, 32, max_adapters=2)
        for i in range(2):
            lora.load_adapter(i, torch.randn(64, 4), torch.randn(4, 32))

        with pytest.raises(RuntimeError, match="Max adapters"):
            lora.load_adapter(99, torch.randn(64, 4), torch.randn(4, 32))

    def test_rank_mismatch_raises(self):
        lora = self._make(64, 32)
        A = torch.randn(64, 4)
        B = torch.randn(8, 32)  # rank 8 != 4
        with pytest.raises(ValueError, match="rank mismatch"):
            lora.load_adapter(1, A, B)

    def test_scaling_factor(self):
        lora = self._make(64, 32)
        A = torch.randn(64, 4)
        B = torch.randn(4, 32)

        lora.load_adapter(1, A, B, scaling=0.5)
        lora.load_adapter(2, A.clone(), B.clone(), scaling=2.0)

        x = torch.randn(1, 64)
        base_out = lora.base(x)
        delta = (x @ A) @ B

        out_half = lora(x, adapter_id=1)
        out_double = lora(x, adapter_id=2)

        assert torch.allclose(out_half, base_out + 0.5 * delta, atol=1e-5)
        assert torch.allclose(out_double, base_out + 2.0 * delta, atol=1e-5)


# =========================================================================
# LoRAManager
# =========================================================================

class TestLoRAManager:
    def _make_model(self):
        """Create a simple model with q_proj and v_proj linear layers."""
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.k_proj = nn.Linear(64, 64, bias=False)
                self.v_proj = nn.Linear(64, 64, bias=False)
                self.o_proj = nn.Linear(64, 64, bias=False)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = Layer()

        return Model()

    def test_auto_wrap(self):
        model = self._make_model()
        manager = LoRAManager(model)
        manager.auto_wrap()

        # Should have wrapped q_proj, k_proj, v_proj, o_proj
        assert len(manager._lora_modules) == 4
        assert isinstance(model.layer0.attn.q_proj, LoRALinear)
        assert isinstance(model.layer0.attn.v_proj, LoRALinear)

    def test_auto_wrap_custom_targets(self):
        model = self._make_model()
        manager = LoRAManager(model)
        manager.auto_wrap(target_names=["q_proj", "v_proj"])
        assert len(manager._lora_modules) == 2

    def test_load_and_swap_adapter(self):
        model = self._make_model()
        manager = LoRAManager(model)
        manager.auto_wrap(target_names=["q_proj"])

        # Build adapter weights matching module path
        module_path = list(manager._lora_modules.keys())[0]
        weights = {
            f"{module_path}.lora_A": torch.randn(64, 4),
            f"{module_path}.lora_B": torch.randn(4, 64),
        }

        manager.load_adapter(adapter_id=1, adapter_name="adapter-v1", adapter_weights=weights)
        assert manager.list_adapters() == {1: "adapter-v1"}

        # Hot-swap: load a second adapter
        weights2 = {
            f"{module_path}.lora_A": torch.randn(64, 8),
            f"{module_path}.lora_B": torch.randn(8, 64),
        }
        manager.load_adapter(adapter_id=2, adapter_name="adapter-v2", adapter_weights=weights2)
        assert len(manager.list_adapters()) == 2

        # Switch between adapters
        manager.set_active_adapter(1)
        x = torch.randn(1, 64)
        out1 = model.layer0.attn.q_proj(x)

        manager.set_active_adapter(2)
        out2 = model.layer0.attn.q_proj(x)

        assert not torch.allclose(out1, out2)

    def test_unload_adapter(self):
        model = self._make_model()
        manager = LoRAManager(model)
        manager.auto_wrap(target_names=["q_proj"])

        module_path = list(manager._lora_modules.keys())[0]
        weights = {
            f"{module_path}.lora_A": torch.randn(64, 4),
            f"{module_path}.lora_B": torch.randn(4, 64),
        }
        manager.load_adapter(1, "test", weights)
        assert 1 in manager.list_adapters()

        manager.unload_adapter(1)
        assert 1 not in manager.list_adapters()

    def test_set_no_adapter(self):
        model = self._make_model()
        manager = LoRAManager(model)
        manager.auto_wrap(target_names=["q_proj"])

        module_path = list(manager._lora_modules.keys())[0]
        weights = {
            f"{module_path}.lora_A": torch.randn(64, 4),
            f"{module_path}.lora_B": torch.randn(4, 64),
        }
        manager.load_adapter(1, "test", weights)
        manager.set_active_adapter(1)

        x = torch.randn(1, 64)
        out_lora = model.layer0.attn.q_proj(x)

        # Deactivate
        manager.set_active_adapter(None)
        out_base = model.layer0.attn.q_proj(x)

        base_linear = manager._lora_modules[module_path].base
        assert torch.allclose(out_base, base_linear(x))
