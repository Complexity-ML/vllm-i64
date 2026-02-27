"""
vllm-i64 :: Weight Loader

Load checkpoint weights into models.
Handles:
  - Tied embeddings (lm_head.weight → embed_tokens.weight)
  - Token-routed MLP expert weights (gate_up_proj, down_proj)
  - token_to_expert buffer
  - dtype conversion

INL - 2025
"""

import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path

from vllm_i64.core.registry import get_model_entry


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    strict: bool = False,
) -> dict:
    """
    Load a checkpoint into a model.

    Handles Pacific-Prime/Complexity Deep specifics:
      - lm_head.weight → embed_tokens.weight (tied)
      - token_to_expert → buffer (not parameter)
      - rotary_emb.inv_freq → skip (computed at init)

    Args:
        model: the model to load into
        checkpoint_path: path to .pt file
        dtype: target dtype for weights
        device: target device
        strict: if True, require all keys match

    Returns:
        dict with loading stats
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # If checkpoint wraps in "model" key
    if "model" in state_dict and "model.layers.0.input_layernorm.weight" not in state_dict:
        state_dict = state_dict["model"]

    # If checkpoint wraps in "state_dict" key
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    loaded = set()
    skipped = set()
    missing = set()

    for name, weight in state_dict.items():
        # Skip rotary inv_freq (computed)
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
        if name == "embed_tokens.weight" or name == "model.embed_tokens.weight":
            if name in loaded:
                continue

        # Buffer (token_to_expert)
        if "token_to_expert" in name:
            if name in buffers:
                buffers[name].copy_(weight)
                loaded.add(name)
            continue

        # Regular parameter
        if name in params:
            params[name].data.copy_(weight.to(dtype))
            loaded.add(name)
        else:
            # Try with "model." prefix stripped
            stripped = name.replace("model.", "", 1)
            if stripped in params:
                params[stripped].data.copy_(weight.to(dtype))
                loaded.add(name)
            else:
                missing.add(name)

    # Check for unloaded model params
    model_params = set(params.keys())
    unloaded = model_params - loaded
    # Filter out params that don't need checkpoint (e.g. freshly initialized mu_router)
    unloaded = {p for p in unloaded if "mu_router" not in p}

    stats = {
        "loaded": len(loaded),
        "skipped": len(skipped),
        "missing_in_model": len(missing),
        "unloaded_params": len(unloaded),
    }

    print(f"  Loaded: {stats['loaded']} tensors")
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
) -> nn.Module:
    """
    Load a registered model by name.

    Args:
        model_name: registered name (e.g. "pacific-prime-chat")
        dtype: weight dtype
        device: target device
        checkpoint_override: override checkpoint path

    Returns:
        loaded model
    """
    import importlib

    entry = get_model_entry(model_name)

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

    # Create model
    model = model_cls(config)
    model = model.to(dtype)

    # Load weights
    ckpt_path = checkpoint_override or entry.checkpoint
    if ckpt_path:
        load_checkpoint(model, ckpt_path, dtype=dtype, device=device)

    return model.to(device)
