"""
vllm-i64 :: LoRA (Low-Rank Adaptation) Layer

Supports hot-swapping LoRA adapters at runtime:
  - Multiple LoRA adapters loaded simultaneously
  - Per-request adapter selection
  - Zero-copy base weights (shared across adapters)
  - Adapter loading from safetensors/PyTorch files

Integer-first: adapter selection is integer (adapter_id).
Only the low-rank matrices A, B are float.

INL - 2025
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class LoRALinear(nn.Module):
    """
    Linear layer with optional LoRA adapter.

    forward(x) = base_linear(x) + scaling * (x @ A @ B)

    where A: (in_features, rank), B: (rank, out_features) are low-rank.

    Multiple adapters can be loaded; selection is by integer adapter_id.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        max_adapters: int = 8,
    ):
        super().__init__()
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.max_adapters = max_adapters

        # Adapter storage: id → (A, B, scaling)
        self._adapters: Dict[int, Tuple[torch.Tensor, torch.Tensor, float]] = {}
        self._active_adapter_id: Optional[int] = None

    def load_adapter(
        self,
        adapter_id: int,
        lora_A: torch.Tensor,  # (in_features, rank)
        lora_B: torch.Tensor,  # (rank, out_features)
        scaling: float = 1.0,
    ):
        """Load a LoRA adapter. Integer adapter_id for selection."""
        if len(self._adapters) >= self.max_adapters:
            raise RuntimeError(f"Max adapters ({self.max_adapters}) reached")

        # Move to same device as base weights
        device = self.base.weight.device
        dtype = self.base.weight.dtype
        self._adapters[adapter_id] = (
            lora_A.to(device=device, dtype=dtype),
            lora_B.to(device=device, dtype=dtype),
            scaling,
        )

    def unload_adapter(self, adapter_id: int):
        """Unload a LoRA adapter."""
        self._adapters.pop(adapter_id, None)
        if self._active_adapter_id == adapter_id:
            self._active_adapter_id = None

    def set_active_adapter(self, adapter_id: Optional[int]):
        """Set active adapter by integer ID. None = base model only."""
        self._active_adapter_id = adapter_id

    def forward(self, x: torch.Tensor, adapter_id: Optional[int] = None) -> torch.Tensor:
        """Forward with optional LoRA."""
        out = self.base(x)

        aid = adapter_id if adapter_id is not None else self._active_adapter_id
        if aid is not None and aid in self._adapters:
            A, B, scaling = self._adapters[aid]
            lora_out = (x @ A) @ B
            out = out + scaling * lora_out

        return out

    @property
    def num_adapters(self) -> int:
        return len(self._adapters)

    @property
    def adapter_ids(self):
        return list(self._adapters.keys())


class LoRAManager:
    """
    Manages LoRA adapters across a model.

    Wraps target modules (attention Q/K/V/O projections) with LoRALinear
    and provides hot-swap capability.
    """

    def __init__(self, model: nn.Module, max_adapters: int = 8):
        self.model = model
        self.max_adapters = max_adapters
        self._lora_modules: Dict[str, LoRALinear] = {}
        self._loaded_adapters: Dict[int, str] = {}  # adapter_id → name

    def wrap_module(self, module_path: str, target_linear: nn.Linear) -> LoRALinear:
        """Wrap a Linear layer with LoRA capability."""
        lora_linear = LoRALinear(target_linear, max_adapters=self.max_adapters)
        self._lora_modules[module_path] = lora_linear
        return lora_linear

    def auto_wrap(self, target_names: Optional[list] = None):
        """
        Automatically wrap target linear layers in the model.

        Default targets: q_proj, k_proj, v_proj, o_proj (attention layers).
        """
        if target_names is None:
            target_names = ["q_proj", "k_proj", "v_proj", "o_proj"]

        for name, module in self.model.named_modules():
            for target in target_names:
                if name.endswith(target) and isinstance(module, nn.Linear):
                    # Replace with LoRALinear
                    lora_mod = self.wrap_module(name, module)
                    # Set on parent
                    parts = name.rsplit(".", 1)
                    if len(parts) == 2:
                        parent = dict(self.model.named_modules())[parts[0]]
                        setattr(parent, parts[1], lora_mod)
                    break

    def load_adapter(
        self,
        adapter_id: int,
        adapter_name: str,
        adapter_weights: Dict[str, torch.Tensor],
        scaling: float = 1.0,
    ):
        """
        Load a LoRA adapter from a weight dict.

        Expected keys: "{module_path}.lora_A", "{module_path}.lora_B"
        """
        for module_path, lora_module in self._lora_modules.items():
            a_key = f"{module_path}.lora_A"
            b_key = f"{module_path}.lora_B"

            # Try alternate naming conventions
            for a_k in [a_key, a_key.replace(".", "/"), f"base_model.model.{a_key}"]:
                if a_k in adapter_weights:
                    a_key = a_k
                    break
            for b_k in [b_key, b_key.replace(".", "/"), f"base_model.model.{b_key}"]:
                if b_k in adapter_weights:
                    b_key = b_k
                    break

            if a_key in adapter_weights and b_key in adapter_weights:
                lora_module.load_adapter(
                    adapter_id,
                    adapter_weights[a_key],
                    adapter_weights[b_key],
                    scaling=scaling,
                )

        self._loaded_adapters[adapter_id] = adapter_name

    def unload_adapter(self, adapter_id: int):
        """Unload an adapter from all modules."""
        for lora_module in self._lora_modules.values():
            lora_module.unload_adapter(adapter_id)
        self._loaded_adapters.pop(adapter_id, None)

    def set_active_adapter(self, adapter_id: Optional[int]):
        """Set active adapter across all modules."""
        for lora_module in self._lora_modules.values():
            lora_module.set_active_adapter(adapter_id)

    def list_adapters(self) -> Dict[int, str]:
        """List loaded adapters."""
        return dict(self._loaded_adapters)
