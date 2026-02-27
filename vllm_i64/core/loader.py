"""
vllm-i64 :: Weight Loader

Load checkpoint weights into models.
TP-aware: detects ColumnParallelLinear / RowParallelLinear and loads
only the current rank's shard from the full checkpoint.

Handles:
  - Tied embeddings (lm_head.weight → embed_tokens.weight)
  - Token-routed MLP expert weights (gate_up_proj, down_proj) — TP sharded
  - Attention Q/K/V (ColumnParallel) — TP sharded on heads
  - Attention O (RowParallel) — TP sharded on input dim
  - token_to_expert buffer (replicated)
  - dtype conversion

INL - 2025
"""

import json as _json
import torch
import torch.nn as nn
from typing import Optional, Dict
from pathlib import Path

from vllm_i64.core.registry import get_model_entry
from vllm_i64.parallel.tensor_parallel import (
    get_tp, ColumnParallelLinear, RowParallelLinear,
)


# =========================================================================
# Multi-format state_dict loading (safetensors + PyTorch)
# =========================================================================

def _load_safetensors_file(filepath: str) -> Dict[str, torch.Tensor]:
    """Load a single .safetensors file."""
    from safetensors.torch import load_file
    return load_file(filepath)


def _load_pytorch_file(filepath: str) -> Dict[str, torch.Tensor]:
    """Load a PyTorch checkpoint file and unwrap nested state dicts."""
    state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict):
        if "model" in state_dict and "model.layers.0.input_layernorm.weight" not in state_dict:
            state_dict = state_dict["model"]
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    return state_dict


def _load_sharded_safetensors(directory: Path) -> Dict[str, torch.Tensor]:
    """Load sharded safetensors from a HuggingFace model directory."""
    index_path = directory / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        index = _json.load(f)

    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))

    state_dict = {}
    for shard_name in shard_files:
        shard_path = directory / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        state_dict.update(_load_safetensors_file(str(shard_path)))
    return state_dict


def _load_from_directory(dir_path: Path) -> Dict[str, torch.Tensor]:
    """
    Load from a model directory. Priority:
      1. model.safetensors.index.json (sharded)
      2. model.safetensors (single file)
      3. *.safetensors (glob)
      4. *.pt / *.pth / *.bin (PyTorch)
    """
    if (dir_path / "model.safetensors.index.json").exists():
        return _load_sharded_safetensors(dir_path)

    single_st = dir_path / "model.safetensors"
    if single_st.exists():
        return _load_safetensors_file(str(single_st))

    st_files = sorted(dir_path.glob("*.safetensors"))
    if st_files:
        state_dict = {}
        for f in st_files:
            state_dict.update(_load_safetensors_file(str(f)))
        return state_dict

    pt_files = sorted(dir_path.glob("*.pt")) + sorted(dir_path.glob("*.pth")) + sorted(dir_path.glob("*.bin"))
    if pt_files:
        state_dict = {}
        for f in pt_files:
            state_dict.update(_load_pytorch_file(str(f)))
        return state_dict

    raise FileNotFoundError(f"No checkpoint files found in {dir_path}")


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Auto-detect and load state dict from any supported format.

    Supports:
      - .pt / .pth / .bin (PyTorch)
      - .safetensors (single or sharded)
      - Directories (HuggingFace model dirs)
    """
    path = Path(checkpoint_path)

    if path.is_dir():
        return _load_from_directory(path)
    elif path.suffix == ".safetensors":
        return _load_safetensors_file(str(path))
    elif path.suffix in (".pt", ".pth", ".bin"):
        return _load_pytorch_file(str(path))
    else:
        try:
            return _load_pytorch_file(str(path))
        except Exception:
            return _load_safetensors_file(str(path))


def _get_module_for_param(model: nn.Module, param_name: str) -> Optional[nn.Module]:
    """Walk the module tree to find the module owning a parameter."""
    parts = param_name.split(".")
    # The last part is the actual parameter name (e.g. "weight", "bias")
    # Walk up to the parent module
    module = model
    for part in parts[:-1]:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    strict: bool = False,
) -> dict:
    """
    TP-aware checkpoint loading.

    For ColumnParallelLinear / RowParallelLinear layers, loads the full
    weight from checkpoint and calls load_full_weight() which takes the
    current rank's shard.

    For TokenRoutedMLP expert weights, calls load_full_weights() which
    shards gate_up and down projections.

    Other params are loaded directly (replicated).

    Args:
        model: the model to load into
        checkpoint_path: path to .pt file or directory
        dtype: target dtype for weights
        device: target device
        strict: if True, require all keys match

    Returns:
        dict with loading stats
    """
    tp = get_tp()
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if tp.tp_rank == 0:
        print(f"Loading checkpoint: {checkpoint_path}")

    state_dict = _load_state_dict(checkpoint_path)

    # Build lookup of model parameters and their parent modules
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    loaded = set()
    skipped = set()
    missing = set()

    # Collect expert weight pairs for MLP sharding
    expert_gate_up = {}  # layer_prefix → tensor
    expert_down = {}     # layer_prefix → tensor

    for name, weight in state_dict.items():
        # Skip rotary inv_freq (computed at init)
        if "rotary_emb.inv_freq" in name:
            skipped.add(name)
            continue

        # Tied embeddings: lm_head → embed_tokens
        if name == "lm_head.weight":
            embed_name = "embed_tokens.weight"
            if embed_name in params:
                params[embed_name].data.copy_(weight.to(dtype))
                loaded.add(name)
                loaded.add(embed_name)
            continue

        # Skip embed_tokens if already loaded via lm_head
        if name in ("embed_tokens.weight", "model.embed_tokens.weight"):
            if name in loaded:
                continue

        # Buffer (token_to_expert — replicated)
        if "token_to_expert" in name:
            buf_name = name.replace("model.", "", 1) if name not in buffers else name
            if buf_name in buffers:
                buffers[buf_name].copy_(weight)
                loaded.add(name)
            continue

        # Resolve actual param name (strip "model." prefix if needed)
        resolved_name = name
        if name not in params:
            resolved_name = name.replace("model.", "", 1)

        if resolved_name not in params:
            missing.add(name)
            continue

        # Check if this belongs to a TP-sharded module
        parent_module = _get_module_for_param(model, resolved_name)
        param_leaf = resolved_name.split(".")[-1]

        if isinstance(parent_module, ColumnParallelLinear) and param_leaf == "weight":
            # Full weight → take TP shard
            parent_module.load_full_weight(weight.to(dtype))
            loaded.add(name)

        elif isinstance(parent_module, RowParallelLinear) and param_leaf == "weight":
            # Full weight → take TP shard
            parent_module.load_full_weight(weight.to(dtype))
            loaded.add(name)

        elif "gate_up_proj" in resolved_name and "mlp" in resolved_name:
            # Expert MLP gate_up — collect for paired sharding
            prefix = resolved_name.rsplit("gate_up_proj", 1)[0]
            expert_gate_up[prefix] = weight
            loaded.add(name)

        elif "down_proj" in resolved_name and "mlp" in resolved_name:
            # Expert MLP down — collect for paired sharding
            prefix = resolved_name.rsplit("down_proj", 1)[0]
            expert_down[prefix] = weight
            loaded.add(name)

        else:
            # Regular parameter (replicated) — direct copy
            params[resolved_name].data.copy_(weight.to(dtype))
            loaded.add(name)

    # Load expert MLP weight pairs with TP sharding
    for prefix in expert_gate_up:
        if prefix in expert_down:
            module_path = prefix.rstrip(".")
            mlp_module = _get_module_for_param(model, module_path + ".gate_up_proj")
            if mlp_module is None:
                # Try navigating to the MLP module directly
                parts = module_path.split(".")
                mlp = model
                for p in parts:
                    if hasattr(mlp, p):
                        mlp = getattr(mlp, p)
                mlp_module = mlp

            if hasattr(mlp_module, "load_full_weights"):
                mlp_module.load_full_weights(
                    expert_gate_up[prefix].to(dtype),
                    expert_down[prefix].to(dtype),
                )

    # Check for unloaded model params
    model_params = set(params.keys())
    unloaded = model_params - loaded
    # Filter out params that don't need checkpoint (freshly initialized)
    unloaded = {p for p in unloaded if "mu_router" not in p and "mu_to_" not in p}

    stats = {
        "loaded": len(loaded),
        "skipped": len(skipped),
        "missing_in_model": len(missing),
        "unloaded_params": len(unloaded),
        "tp_rank": tp.tp_rank,
        "tp_size": tp.tp_size,
    }

    if tp.tp_rank == 0:
        print(f"  Loaded: {stats['loaded']} tensors (TP={tp.tp_size})")
        if stats['skipped']:
            print(f"  Skipped: {stats['skipped']} (rotary inv_freq, etc.)")
        if stats['missing_in_model']:
            print(f"  Not in model: {stats['missing_in_model']}")
        if stats['unloaded_params']:
            print(f"  Unloaded params: {stats['unloaded_params']}")
            if strict:
                raise RuntimeError(f"Missing weights: {unloaded}")

    return stats


def load_model_by_name(
    model_name: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    checkpoint_override: Optional[str] = None,
    quantization: Optional[str] = None,
) -> nn.Module:
    """
    Load a registered model by name with TP support.

    Each rank creates the model (with TP-aware layers), then
    load_checkpoint takes the appropriate shard from the full checkpoint.

    Args:
        model_name: registered name (e.g. "pacific-prime-chat")
        dtype: weight dtype
        device: target device
        checkpoint_override: override checkpoint path
        quantization: optional quantization method ("int8", "int4", or None)

    Returns:
        loaded model on the correct device
    """
    import importlib

    entry = get_model_entry(model_name)
    tp = get_tp()

    # Import model class dynamically
    module_path, class_name = entry.model_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    # Import config class dynamically
    config_module_path, config_class_name = entry.config_loader.rsplit(".", 1)
    config_module = importlib.import_module(config_module_path)
    config_cls = getattr(config_module, config_class_name)

    # Load config
    config = config_cls.from_json(entry.config_path)

    # Create model — TP layers auto-detect tp_size from global state
    model = model_cls(config)
    model = model.to(dtype)

    # Load weights — TP-aware sharding
    ckpt_path = checkpoint_override or entry.checkpoint
    if ckpt_path:
        load_checkpoint(model, ckpt_path, dtype=dtype, device=device)

    # Post-load quantization of expert weights
    if quantization and quantization != "none":
        _quantize_experts(model, quantization)
        if tp.tp_rank == 0:
            print(f"  Quantized expert weights: {quantization}")

    return model.to(device)


def _quantize_experts(model: nn.Module, method: str):
    """Apply post-load quantization to expert MLP weights."""
    from vllm_i64.core.quantization import QuantConfig, quantize_int8, dequantize_int8

    if method not in ("int8", "int4"):
        return

    for name, module in model.named_modules():
        # Look for TokenRoutedMLP or modules with gate_up_proj + down_proj
        if hasattr(module, 'gate_up_proj') and hasattr(module, 'down_proj'):
            gate_up = module.gate_up_proj
            down = module.down_proj

            if method == "int8" and hasattr(gate_up, 'weight'):
                q, s = quantize_int8(gate_up.weight.data)
                # Store quantized weights and scales as buffers
                module.register_buffer("gate_up_int8", q)
                module.register_buffer("gate_up_scale", s)
                q, s = quantize_int8(down.weight.data)
                module.register_buffer("down_int8", q)
                module.register_buffer("down_scale", s)
