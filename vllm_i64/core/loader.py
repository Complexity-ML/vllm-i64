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
import logging
import torch
import torch.nn as nn
from typing import Optional, Dict
from pathlib import Path

from vllm_i64.core.registry import get_model_entry
from vllm_i64.parallel.tensor_parallel import (
    get_tp, ColumnParallelLinear, RowParallelLinear,
)

logger = logging.getLogger("vllm_i64.loader")


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
    with open(index_path, "r", encoding="utf-8") as f:
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
        except (OSError, RuntimeError, KeyError):
            return _load_safetensors_file(str(path))


def _convert_framework_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert complexity-framework checkpoint format to vllm-i64 format.

    Handles:
    1. Separate expert weights (experts.0.gate_proj + experts.0.up_proj → fused gate_up_proj)
    2. Dense MLP (mlp.gate_proj + mlp.up_proj → fused gate_up_proj with 1 expert)
    3. INL dynamics controller format (controller.0/2 → controller_in/out)
    4. token_to_expert buffer
    """
    import re

    # Detect framework format: look for "mlp.experts.0.gate_proj" or "mlp.gate_proj"
    has_separate_experts = any("mlp.experts." in k and "gate_proj" in k for k in state_dict)
    has_dense_mlp = any(re.match(r"layers\.\d+\.mlp\.gate_proj", k) for k in state_dict)

    if not has_separate_experts and not has_dense_mlp:
        return state_dict  # Already in vllm-i64 format or unknown

    logger.info("Detected complexity-framework format, converting weights...")
    converted = {}

    # Collect expert weights per layer
    expert_weights = {}  # (layer_idx, expert_id) → {gate_proj, up_proj, down_proj}

    for name, tensor in state_dict.items():
        # Expert format: layers.X.mlp.experts.E.{gate,up,down}_proj.weight
        m = re.match(r"(layers\.\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name)
        if m:
            layer_prefix, expert_id, proj_type = m.group(1), int(m.group(2)), m.group(3)
            key = (layer_prefix, expert_id)
            if key not in expert_weights:
                expert_weights[key] = {}
            expert_weights[key][proj_type] = tensor
            continue

        # Dense MLP: layers.X.mlp.{gate,up,down}_proj.weight
        m = re.match(r"(layers\.\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight", name)
        if m:
            layer_prefix, proj_type = m.group(1), m.group(2)
            key = (layer_prefix, 0)  # Single expert = expert 0
            if key not in expert_weights:
                expert_weights[key] = {}
            expert_weights[key][proj_type] = tensor
            continue

        # token_to_expert buffer
        if "token_to_expert" in name:
            converted[name] = tensor
            continue

        # INL dynamics: controller.0 → controller_in, controller.2 → controller_out
        m = re.match(r"(layers\.\d+\.dynamics)\.controller\.0\.(weight|bias)", name)
        if m:
            converted[f"{m.group(1)}.controller_in.{m.group(2)}"] = tensor
            continue
        m = re.match(r"(layers\.\d+\.dynamics)\.controller\.2\.(weight|bias)", name)
        if m:
            converted[f"{m.group(1)}.controller_out.{m.group(2)}"] = tensor
            continue

        # Everything else passes through as-is
        converted[name] = tensor

    # Fuse expert gate+up into gate_up_proj [num_experts, hidden, 2*inter]
    # and stack down_proj [num_experts, inter, hidden]
    from collections import defaultdict
    layers_experts = defaultdict(dict)
    for (layer_prefix, expert_id), weights in expert_weights.items():
        layers_experts[layer_prefix][expert_id] = weights

    for layer_prefix, experts in layers_experts.items():
        num_experts = max(experts.keys()) + 1

        gate_up_list = []
        down_list = []
        for eid in range(num_experts):
            w = experts[eid]
            gate = w["gate_proj"]  # [inter, hidden]
            up = w["up_proj"]      # [inter, hidden]
            down = w["down_proj"]  # [hidden, inter]

            # Fuse gate+up: [hidden, 2*inter] (transposed for bmm format)
            gate_up = torch.cat([gate, up], dim=0).t()  # [hidden, 2*inter]
            gate_up_list.append(gate_up)
            down_list.append(down.t())  # [inter, hidden]

        # Stack: [num_experts, hidden, 2*inter]
        gate_up_stacked = torch.stack(gate_up_list, dim=0)
        down_stacked = torch.stack(down_list, dim=0)

        converted[f"{layer_prefix}.mlp.gate_up_proj"] = gate_up_stacked
        converted[f"{layer_prefix}.mlp.down_proj"] = down_stacked

        # Create token_to_expert buffer if not present
        tok_expert_key = f"{layer_prefix}.mlp.token_to_expert"
        if tok_expert_key not in converted:
            vocab_size = state_dict.get("embed_tokens.weight", torch.zeros(32000, 1)).shape[0]
            converted[tok_expert_key] = torch.arange(vocab_size, dtype=torch.long) % num_experts

    logger.info("Converted %d layers (%d expert groups)", len(layers_experts), len(expert_weights))
    return converted


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
        logger.info("Loading checkpoint: %s", checkpoint_path)

    state_dict = _load_state_dict(checkpoint_path)

    # Auto-detect complexity-framework format and convert to vllm-i64 format
    state_dict = _convert_framework_weights(state_dict)

    # Build lookup of model parameters and their parent modules
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    loaded = set()           # checkpoint keys that were loaded
    loaded_params = set()    # model param names that received data
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

        # TP-wrapped layers: checkpoint has "q_proj.weight" but model has
        # "q_proj.linear.weight" (ColumnParallelLinear / RowParallelLinear)
        if resolved_name not in params and resolved_name.endswith(".weight"):
            tp_name = resolved_name[:-len(".weight")] + ".linear.weight"
            if tp_name in params:
                resolved_name = tp_name

        if resolved_name not in params:
            missing.add(name)
            continue

        # Check if this belongs to a TP-sharded module
        parent_module = _get_module_for_param(model, resolved_name)
        param_leaf = resolved_name.split(".")[-1]

        # For TP-wrapped names like "q_proj.linear.weight", check the grandparent
        tp_parent = None
        if param_leaf == "weight" and ".linear.weight" in resolved_name:
            gp_name = resolved_name.rsplit(".linear.weight", 1)[0] + ".dummy"
            tp_parent = _get_module_for_param(model, gp_name)

        if isinstance(tp_parent or parent_module, ColumnParallelLinear) and param_leaf == "weight":
            # Full weight → take TP shard
            tp_mod = tp_parent if isinstance(tp_parent, ColumnParallelLinear) else parent_module
            tp_mod.load_full_weight(weight.to(dtype))
            loaded.add(name)
            loaded_params.add(resolved_name)

        elif isinstance(tp_parent or parent_module, RowParallelLinear) and param_leaf == "weight":
            # Full weight → take TP shard
            tp_mod = tp_parent if isinstance(tp_parent, RowParallelLinear) else parent_module
            tp_mod.load_full_weight(weight.to(dtype))
            loaded.add(name)
            loaded_params.add(resolved_name)

        elif "gate_up_proj" in resolved_name and "mlp" in resolved_name:
            # Expert MLP gate_up — collect for paired sharding
            prefix = resolved_name.rsplit("gate_up_proj", 1)[0]
            expert_gate_up[prefix] = weight
            loaded.add(name)
            loaded_params.add(resolved_name)

        elif "down_proj" in resolved_name and "mlp" in resolved_name:
            # Expert MLP down — collect for paired sharding
            prefix = resolved_name.rsplit("down_proj", 1)[0]
            expert_down[prefix] = weight
            loaded.add(name)
            loaded_params.add(resolved_name)

        else:
            # Regular parameter (replicated) — direct copy
            params[resolved_name].data.copy_(weight.to(dtype))
            loaded.add(name)
            loaded_params.add(resolved_name)

    # Load expert MLP weight pairs with TP sharding
    unpaired_gu = set(expert_gate_up.keys()) - set(expert_down.keys())
    unpaired_dn = set(expert_down.keys()) - set(expert_gate_up.keys())
    if unpaired_gu or unpaired_dn:
        _logger = logging.getLogger("vllm_i64.loader")
        for p in unpaired_gu:
            _logger.warning("Expert gate_up_proj without matching down_proj: %s", p)
        for p in unpaired_dn:
            _logger.warning("Expert down_proj without matching gate_up_proj: %s", p)

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
    unloaded = model_params - loaded_params

    stats = {
        "loaded": len(loaded),
        "skipped": len(skipped),
        "missing_in_model": len(missing),
        "unloaded_params": len(unloaded),
        "tp_rank": tp.tp_rank,
        "tp_size": tp.tp_size,
    }

    if tp.tp_rank == 0:
        logger.info("Loaded: %d tensors (TP=%d)", stats['loaded'], tp.tp_size)
        if stats['skipped']:
            logger.info("Skipped: %d (rotary inv_freq, etc.)", stats['skipped'])
        if stats['missing_in_model']:
            logger.warning("Not in model: %d", stats['missing_in_model'])
        if stats['unloaded_params']:
            logger.warning("Unloaded params: %d", stats['unloaded_params'])
            if strict:
                raise RuntimeError(f"Missing weights: {unloaded}")

    return stats


def _detect_checkpoint_quantization(checkpoint_path: str) -> Optional[str]:
    """
    Auto-detect AWQ/GPTQ quantization from checkpoint config.json.

    Returns "awq", "gptq", or None.
    """
    from vllm_i64.core.awq_gptq import detect_quant_config

    result = detect_quant_config(checkpoint_path)
    if result is not None:
        return result[0]  # "awq" or "gptq"
    return None


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

    Supports pre-quantized checkpoints:
      - "awq": load AWQ-quantized checkpoint (auto-detected or explicit)
      - "gptq": load GPTQ-quantized checkpoint (auto-detected or explicit)
      - "int8"/"int4": post-load quantization of float checkpoints
      - "fp8": FP8 quantization for Hopper GPUs

    Args:
        model_name: registered name (e.g. "pacific-prime-chat")
        dtype: weight dtype
        device: target device
        checkpoint_override: override checkpoint path
        quantization: optional quantization method ("int8", "int4", "awq", "gptq", or None)

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

    # Load config — use checkpoint dir's config.json if override provided
    config_path = entry.config_path
    if checkpoint_override:
        import os
        override_config = os.path.join(checkpoint_override, "config.json")
        if os.path.exists(override_config):
            config_path = override_config
    config = config_cls.from_json(config_path)

    # Create model — TP layers auto-detect tp_size from global state
    model = model_cls(config)
    model = model.to(dtype)

    # Resolve checkpoint path
    ckpt_path = checkpoint_override or entry.checkpoint

    # Auto-detect AWQ/GPTQ from checkpoint if quantization not explicitly set
    effective_quant = quantization
    if ckpt_path and effective_quant in (None, "none"):
        detected = _detect_checkpoint_quantization(ckpt_path)
        if detected is not None:
            effective_quant = detected
            if tp.tp_rank == 0:
                logger.info("Auto-detected quantization: %s", detected)

    # Load weights — dispatch by quantization format
    if ckpt_path and effective_quant in ("awq", "gptq"):
        # Pre-quantized checkpoint: use AWQ/GPTQ loader
        from vllm_i64.core.awq_gptq import load_awq_checkpoint, load_gptq_checkpoint

        if effective_quant == "awq":
            stats = load_awq_checkpoint(model, ckpt_path, device=device, dtype=dtype)
        else:
            stats = load_gptq_checkpoint(model, ckpt_path, device=device, dtype=dtype)

        if tp.tp_rank == 0:
            logger.info("Loaded %s checkpoint: %d quantized layers, %d non-quantized weights",
                        effective_quant.upper(), stats['quant_layers'], stats['non_quant_loaded'])
    elif ckpt_path:
        # Standard float checkpoint
        load_checkpoint(model, ckpt_path, dtype=dtype, device=device)

    # Post-load quantization (only for float checkpoints, not pre-quantized)
    if effective_quant and effective_quant not in ("none", "awq", "gptq"):
        if effective_quant == "fp8":
            _quantize_dense_mlp_fp8(model)
        else:
            _quantize_experts(model, effective_quant)
            _quantize_dense_mlp(model, effective_quant)
            _quantize_attention(model, effective_quant)
            _quantize_mu_attention(model, effective_quant)
            _quantize_i64_dynamics(model, effective_quant)
            _quantize_lm_head(model, effective_quant)
            _quantize_rmsnorm(model, effective_quant)
        if tp.tp_rank == 0:
            logger.info("Quantized weights: %s", effective_quant)

    # Disable grad tracking — inference only, avoids autograd view conflicts
    # with quantized buffers that are views of original parameters.
    model.requires_grad_(False)

    return model.to(device)


def _quantize_dense_mlp(model: nn.Module, method: str):
    """Apply post-load quantization to DenseMLP layers (2D weights, no expert dim)."""
    from vllm_i64.layers.dense_mlp import DenseMLP
    from vllm_i64.core.quantization import quantize_int8, quantize_int4

    if method not in ("int8", "int4"):
        return

    for name, module in model.named_modules():
        if not isinstance(module, DenseMLP):
            continue

        gate_w = module.gate_proj.linear.weight.data
        up_w = module.up_proj.linear.weight.data
        down_w = module.down_proj.linear.weight.data

        if method == "int8":
            gq, gs = quantize_int8(gate_w)
            uq, us = quantize_int8(up_w)
            dq, ds = quantize_int8(down_w)
            module.register_buffer("gate_int8", gq)
            module.register_buffer("gate_scale", gs)
            module.register_buffer("up_int8", uq)
            module.register_buffer("up_scale", us)
            module.register_buffer("down_int8", dq)
            module.register_buffer("down_scale", ds)
            # Fused gate+up for single-matmul path
            module.register_buffer("gate_up_int8", torch.cat([gq, uq], dim=0))
            module.register_buffer("gate_up_scale", torch.cat([gs, us]))
            module.gate_up_inter = gq.shape[0]

        elif method == "int4":
            gp, gs, gz = quantize_int4(gate_w)
            up, us, uz = quantize_int4(up_w)
            dp, ds_, dz = quantize_int4(down_w)
            module.register_buffer("gate_int4", gp)
            module.register_buffer("gate_scale_int4", gs)
            module.register_buffer("gate_zero", gz)
            module.register_buffer("up_int4", up)
            module.register_buffer("up_scale_int4", us)
            module.register_buffer("up_zero", uz)
            module.register_buffer("down_int4", dp)
            module.register_buffer("down_scale_int4", ds_)
            module.register_buffer("down_zero", dz)


def _quantize_experts(model: nn.Module, method: str):
    """Apply post-load quantization to expert MLP weights."""
    from vllm_i64.core.quantization import quantize_int8, quantize_int4

    if method not in ("int8", "int4"):
        return

    for name, module in model.named_modules():
        if not (hasattr(module, 'gate_up_proj') and hasattr(module, 'down_proj')):
            continue

        gate_up = module.gate_up_proj
        down = module.down_proj

        # gate_up_proj / down_proj are nn.Parameter (not nn.Linear)
        gu_data = gate_up.data if isinstance(gate_up, nn.Parameter) else gate_up.weight.data
        dn_data = down.data if isinstance(down, nn.Parameter) else down.weight.data

        if method == "int8":
            num_experts = gu_data.shape[0]
            gu_q, gu_s, dn_q, dn_s = [], [], [], []
            for e in range(num_experts):
                q, s = quantize_int8(gu_data[e])
                gu_q.append(q)
                gu_s.append(s)
                q, s = quantize_int8(dn_data[e])
                dn_q.append(q)
                dn_s.append(s)
            module.register_buffer("gate_up_int8", torch.stack(gu_q))
            module.register_buffer("gate_up_scale", torch.stack(gu_s))
            module.register_buffer("down_int8", torch.stack(dn_q))
            module.register_buffer("down_scale", torch.stack(dn_s))

        elif method == "int4":
            num_experts = gu_data.shape[0]
            gu_p, gu_s, gu_z = [], [], []
            dn_p, dn_s, dn_z = [], [], []
            for e in range(num_experts):
                p, s, z = quantize_int4(gu_data[e])
                gu_p.append(p)
                gu_s.append(s)
                gu_z.append(z)
                p, s, z = quantize_int4(dn_data[e])
                dn_p.append(p)
                dn_s.append(s)
                dn_z.append(z)
            module.register_buffer("gate_up_int4", torch.stack(gu_p))
            module.register_buffer("gate_up_scale_int4", torch.stack(gu_s))
            module.register_buffer("gate_up_zero", torch.stack(gu_z))
            module.register_buffer("down_int4", torch.stack(dn_p))
            module.register_buffer("down_scale_int4", torch.stack(dn_s))
            module.register_buffer("down_zero", torch.stack(dn_z))


def _quantize_attention(model: nn.Module, method: str):
    """
    Apply post-load INT8 quantization to LlamaAttention layers.

    Fused QKV: cat([Q, K, V], dim=0) → single INT8 matmul, split output.
    O: separate INT8 matmul + all_reduce (RowParallel replacement).
    """
    from vllm_i64.models.llama.model import LlamaAttention
    from vllm_i64.core.quantization import quantize_int8

    if method != "int8":
        return  # Only INT8 for attention (INT4 attention has too much error)

    for name, module in model.named_modules():
        if not isinstance(module, LlamaAttention):
            continue

        # Fused QKV — single matmul for all three projections
        q_w = module.q_proj.linear.weight.data
        k_w = module.k_proj.linear.weight.data
        v_w = module.v_proj.linear.weight.data
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_q, qkv_s = quantize_int8(qkv_w)
        module.register_buffer("qkv_int8", qkv_q)
        module.register_buffer("qkv_scale", qkv_s)
        module.q_size = q_w.shape[0]
        module.kv_size = k_w.shape[0]

        # QKV bias (if present)
        if module.q_proj.linear.bias is not None:
            qkv_bias = torch.cat([
                module.q_proj.linear.bias.data,
                module.k_proj.linear.bias.data,
                module.v_proj.linear.bias.data,
            ])
            module.register_buffer("qkv_bias", qkv_bias)

        # O projection — INT8 matmul (replaces RowParallelLinear)
        o_w = module.o_proj.linear.weight.data
        o_q, o_s = quantize_int8(o_w)
        module.register_buffer("o_int8", o_q)
        module.register_buffer("o_scale", o_s)

        if module.o_proj.linear.bias is not None:
            module.register_buffer("o_bias", module.o_proj.linear.bias.data.clone())

        # Free float weights — INT8 buffers replace them
        module.q_proj.linear.weight = None
        module.k_proj.linear.weight = None
        module.v_proj.linear.weight = None
        module.o_proj.linear.weight = None


def _quantize_mu_attention(model: nn.Module, method: str):
    """
    Apply INT8 quantization to MuGuidedAttention layers.

    Fused QKV: cat([Q, K, V], dim=0) → single INT8 matmul, split output.
    Fused mu_QKV: cat([mu_to_q, mu_to_k, mu_to_v], dim=0) → single INT8 matmul.
    O: separate INT8 matmul.
    """
    from vllm_i64.models.complexity_deep.model import MuGuidedAttention
    from vllm_i64.core.quantization import quantize_int8

    if method != "int8":
        return

    for name, module in model.named_modules():
        if not isinstance(module, MuGuidedAttention):
            continue

        # Fused QKV
        q_w = module.q_proj.linear.weight.data
        k_w = module.k_proj.linear.weight.data
        v_w = module.v_proj.linear.weight.data
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_q, qkv_s = quantize_int8(qkv_w)
        module.register_buffer("qkv_int8", qkv_q)
        module.register_buffer("qkv_scale", qkv_s)
        module.q_size = q_w.shape[0]
        module.kv_size = k_w.shape[0]

        # QKV bias (if present)
        if module.q_proj.linear.bias is not None:
            qkv_bias = torch.cat([
                module.q_proj.linear.bias.data,
                module.k_proj.linear.bias.data,
                module.v_proj.linear.bias.data,
            ])
            module.register_buffer("qkv_bias", qkv_bias)

        # Fused mu projections: cat([mu_to_q, mu_to_k, mu_to_v])
        mu_q_w = module.mu_to_q.linear.weight.data
        mu_k_w = module.mu_to_k.linear.weight.data
        mu_v_w = module.mu_to_v.linear.weight.data
        mu_qkv_w = torch.cat([mu_q_w, mu_k_w, mu_v_w], dim=0)
        mu_qkv_q, mu_qkv_s = quantize_int8(mu_qkv_w)
        module.register_buffer("mu_qkv_int8", mu_qkv_q)
        module.register_buffer("mu_qkv_scale", mu_qkv_s)

        # O projection
        o_w = module.o_proj.linear.weight.data
        o_q, o_s = quantize_int8(o_w)
        module.register_buffer("o_int8", o_q)
        module.register_buffer("o_scale", o_s)
        if module.o_proj.linear.bias is not None:
            module.register_buffer("o_bias", module.o_proj.linear.bias.data.clone())

        # Free float weights
        module.q_proj.linear.weight = None
        module.k_proj.linear.weight = None
        module.v_proj.linear.weight = None
        module.o_proj.linear.weight = None
        module.mu_to_q.linear.weight = None
        module.mu_to_k.linear.weight = None
        module.mu_to_v.linear.weight = None


def _quantize_i64_dynamics(model: nn.Module, method: str):
    """
    Quantize INLDynamics controller weights to INT8.

    Enables integer dynamics: INT8 controller matmuls + LUT activations
    (sigmoid, softplus, silu). Zero-FLOP activations after quantization.
    """
    from vllm_i64.models.complexity_deep.model import INLDynamics
    from vllm_i64.core.quantization import quantize_int8

    if method != "int8":
        return

    for name, module in model.named_modules():
        if not isinstance(module, INLDynamics):
            continue

        # Controller in
        ciq, cis = quantize_int8(module.controller_in.weight.data)
        module.register_buffer("ctrl_in_int8", ciq)
        module.register_buffer("ctrl_in_scale", cis)
        module.register_buffer("ctrl_in_bias", module.controller_in.bias.data.clone())

        # Controller out
        coq, cos = quantize_int8(module.controller_out.weight.data)
        module.register_buffer("ctrl_out_int8", coq)
        module.register_buffer("ctrl_out_scale", cos)
        module.register_buffer("ctrl_out_bias", module.controller_out.bias.data.clone())

        # mu_proj
        mpq, mps = quantize_int8(module.mu_proj.weight.data)
        module.register_buffer("mu_proj_int8", mpq)
        module.register_buffer("mu_proj_scale", mps)

        # Free float weights
        module.controller_in.weight = None
        module.controller_out.weight = None
        module.mu_proj.weight = None


def _quantize_lm_head(model: nn.Module, method: str):
    """
    Quantize lm_head (or tied embed_tokens) to INT8.

    lm_head is a large matmul: (batch, hidden) × (vocab, hidden)^T.
    INT8 accelerates logit computation significantly for large vocabularies.
    """
    from vllm_i64.core.quantization import quantize_int8

    if method != "int8":
        return

    # Tied embeddings: quantize embed_tokens weight for use as lm_head
    if getattr(model, 'tie_word_embeddings', False) and model.embed_tokens is not None:
        w = model.embed_tokens.weight.data
        wq, ws = quantize_int8(w)
        model.register_buffer("embed_int8", wq)
        model.register_buffer("embed_scale", ws)
        return

    # Separate lm_head
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        w = model.lm_head.weight.data
        wq, ws = quantize_int8(w)
        model.register_buffer("lm_head_int8", wq)
        model.register_buffer("lm_head_scale", ws)
        model.lm_head.weight = None


def _quantize_rmsnorm(model: nn.Module, method: str):
    """
    Quantize all RMSNorm weights to Q12 INT16 for integer forward path.

    Enables integer RMSNorm: float rsqrt (irreducible) + INT32 weight multiply.
    Only applied for INT8 quantization (full integer pipeline).
    """
    from vllm_i64.layers.rmsnorm import RMSNorm, quantize_rmsnorm

    if method != "int8":
        return  # Only INT8 triggers full integer pipeline

    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            quantize_rmsnorm(module)


def _quantize_dense_mlp_fp8(model: nn.Module):
    """
    Quantize DenseMLP weights to FP8 E4M3 for H100/Ada tensor cores.

    ~2x throughput vs FP16 on Hopper GPUs. Falls back to float dequant on older GPUs.
    """
    from vllm_i64.layers.dense_mlp import DenseMLP
    from vllm_i64.core.fp8 import quantize_fp8

    for name, module in model.named_modules():
        if not isinstance(module, DenseMLP):
            continue

        # Gate projection
        w = module.gate_proj.linear.weight.data
        w_fp8, w_scale = quantize_fp8(w)
        module.register_buffer('gate_fp8', w_fp8)
        module.register_buffer('gate_fp8_scale', w_scale)

        # Up projection
        w = module.up_proj.linear.weight.data
        w_fp8, w_scale = quantize_fp8(w)
        module.register_buffer('up_fp8', w_fp8)
        module.register_buffer('up_fp8_scale', w_scale)

        # Down projection
        w = module.down_proj.linear.weight.data
        w_fp8, w_scale = quantize_fp8(w)
        module.register_buffer('down_fp8', w_fp8)
        module.register_buffer('down_fp8_scale', w_scale)
