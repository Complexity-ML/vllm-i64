"""
vllm-i64 :: AWQ/GPTQ Checkpoint Loading

Load pre-quantized AWQ and GPTQ checkpoints from HuggingFace format
and convert to our internal INT4 representation (packed uint8, scales, zeros)
compatible with int4_linear().

AWQ format (AutoAWQ):
  - qweight: uint32 packed (8 x int4 per uint32)
  - qzeros:  uint32 packed (8 x int4 per uint32)
  - scales:  float16 per group
  - Packing order: least-significant nibble first

GPTQ format (AutoGPTQ):
  - qweight: int32 packed (8 x int4 per int32)
  - qzeros:  int32 packed (8 x int4 per int32)
  - scales:  float16 per group
  - g_idx:   optional group assignment (desc_act=True)
  - Packing order: least-significant nibble first

Both are converted to our internal format:
  - packed: uint8 (2 values per byte, high nibble first)
  - scales: float per group, shape (out_features, num_groups)
  - zeros:  float per group, shape (out_features, num_groups)

INL - 2025
"""

import json as _json
import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

_logger = logging.getLogger("vllm_i64.awq_gptq")


# =========================================================================
# Quantization config detection
# =========================================================================

@dataclass
class AWQConfig:
    """AWQ quantization configuration from config.json."""
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"


@dataclass
class GPTQConfig:
    """GPTQ quantization configuration from config.json."""
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True


def detect_quant_config(checkpoint_path: str) -> Optional[Tuple[str, object]]:
    """
    Detect AWQ or GPTQ quantization from config.json.

    Reads the "quantization_config" field and returns:
      ("awq", AWQConfig) or ("gptq", GPTQConfig) or None.
    """
    config_file = Path(checkpoint_path) / "config.json"
    if not config_file.exists():
        return None

    try:
        with open(config_file, "r") as f:
            data = _json.load(f)
    except Exception:
        return None

    qconfig = data.get("quantization_config")
    if qconfig is None:
        return None

    quant_method = qconfig.get("quant_method", "").lower()

    if quant_method == "awq":
        cfg = AWQConfig(
            bits=qconfig.get("bits", 4),
            group_size=qconfig.get("group_size", 128),
            zero_point=qconfig.get("zero_point", True),
            version=qconfig.get("version", "GEMM"),
        )
        _logger.info(
            "Detected AWQ checkpoint: bits=%d, group_size=%d, zero_point=%s, version=%s",
            cfg.bits, cfg.group_size, cfg.zero_point, cfg.version,
        )
        return ("awq", cfg)

    elif quant_method == "gptq":
        cfg = GPTQConfig(
            bits=qconfig.get("bits", 4),
            group_size=qconfig.get("group_size", 128),
            desc_act=qconfig.get("desc_act", False),
            sym=qconfig.get("sym", True),
        )
        _logger.info(
            "Detected GPTQ checkpoint: bits=%d, group_size=%d, desc_act=%s, sym=%s",
            cfg.bits, cfg.group_size, cfg.desc_act, cfg.sym,
        )
        return ("gptq", cfg)

    return None


# =========================================================================
# AWQ unpacking: uint32 → int4 values → our packed uint8
# =========================================================================

def _unpack_awq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """
    Unpack AWQ qweight from uint32 to individual int4 values.

    AWQ packs 8 x int4 values into each uint32, LSB first.
    qweight shape: (in_features // 8, out_features)
    Output shape:  (in_features, out_features)

    Returns tensor of uint8 with values in [0, 15].
    """
    # Ensure int32 for bitwise ops (uint32 stored as int32 in PyTorch)
    qw = qweight.to(torch.int32)
    unpacked_parts = []
    for shift in range(0, 32, 4):
        vals = (qw >> shift) & 0xF
        unpacked_parts.append(vals)
    # Stack along a new dim then reshape
    # qweight is (in//8, out) → each position yields 8 values along in-dim
    unpacked = torch.stack(unpacked_parts, dim=-2)  # (in//8, 8, out)
    rows, _, cols = unpacked.shape
    unpacked = unpacked.reshape(rows * 8, cols)  # (in_features, out_features)
    return unpacked.to(torch.uint8)


def _unpack_awq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """
    Unpack AWQ qzeros from uint32 to individual int4 zero points.

    qzeros shape: (num_groups, out_features // 8)
    Output shape:  (num_groups, out_features)

    Returns tensor of uint8 with values in [0, 15].
    """
    qz = qzeros.to(torch.int32)
    unpacked_parts = []
    for shift in range(0, 32, 4):
        vals = (qz >> shift) & 0xF
        unpacked_parts.append(vals)
    unpacked = torch.stack(unpacked_parts, dim=-1)  # (num_groups, out//8, 8)
    groups, _, _ = unpacked.shape
    unpacked = unpacked.reshape(groups, -1)  # (num_groups, out_features)
    return unpacked.to(torch.uint8)


def _convert_to_internal_int4(
    unpacked_weights: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert unpacked int4 values to our internal INT4 format.

    Our format:
      - packed: (out_features, in_features // 2) uint8, high nibble first
      - scales: (out_features, num_groups) float
      - zeros:  (out_features, num_groups) float

    Args:
        unpacked_weights: (in_features, out_features) uint8 [0..15]
        scales: (num_groups, out_features) float16
        zeros:  (num_groups, out_features) uint8 [0..15]
        group_size: quantization group size

    Returns:
        (packed, scales, zeros) in our internal format
    """
    in_features, out_features = unpacked_weights.shape

    # Transpose to (out_features, in_features) — our format is row-major per output
    w = unpacked_weights.t().contiguous()  # (out, in)

    # Pack pairs into uint8: high nibble = even index, low nibble = odd index
    packed = (w[:, 0::2] << 4) | w[:, 1::2]
    packed = packed.to(torch.uint8)

    # Transpose scales and zeros to (out_features, num_groups)
    s = scales.t().contiguous().float()   # (out, groups)
    z = zeros.t().contiguous().float()    # (out, groups)

    return packed, s, z


# =========================================================================
# GPTQ unpacking: int32 → int4 values → our packed uint8
# =========================================================================

def _unpack_gptq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """
    Unpack GPTQ qweight from int32 to individual int4 values.

    GPTQ packs 8 x int4 values into each int32, LSB first.
    qweight shape: (in_features // 8, out_features)
    Output shape:  (in_features, out_features)

    Returns tensor of uint8 with values in [0, 15].
    """
    qw = qweight.to(torch.int32)
    unpacked_parts = []
    for shift in range(0, 32, 4):
        vals = (qw >> shift) & 0xF
        unpacked_parts.append(vals)
    unpacked = torch.stack(unpacked_parts, dim=-2)  # (in//8, 8, out)
    rows, _, cols = unpacked.shape
    unpacked = unpacked.reshape(rows * 8, cols)  # (in_features, out_features)
    return unpacked.to(torch.uint8)


def _unpack_gptq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """
    Unpack GPTQ qzeros from int32 to individual int4 zero points.

    qzeros shape: (num_groups, out_features // 8)
    Output shape:  (num_groups, out_features)

    Returns tensor of uint8 with values in [0, 15].
    """
    qz = qzeros.to(torch.int32)
    unpacked_parts = []
    for shift in range(0, 32, 4):
        vals = (qz >> shift) & 0xF
        unpacked_parts.append(vals)
    unpacked = torch.stack(unpacked_parts, dim=-1)  # (num_groups, out//8, 8)
    groups, _, _ = unpacked.shape
    unpacked = unpacked.reshape(groups, -1)  # (num_groups, out_features)
    return unpacked.to(torch.uint8)


def _reorder_by_g_idx(
    unpacked_weights: torch.Tensor,
    g_idx: torch.Tensor,
    group_size: int,
    num_groups: int,
) -> torch.Tensor:
    """
    Reorder GPTQ weights by g_idx (desc_act=True).

    When desc_act=True, GPTQ reorders input channels by activation magnitude
    (descending). g_idx maps each input channel to its group. We need to
    invert this permutation so weights align with sequential group ordering.

    Args:
        unpacked_weights: (in_features, out_features) uint8
        g_idx: (in_features,) int32 — group index per input channel
        group_size: group size
        num_groups: number of groups

    Returns:
        reordered weights: (in_features, out_features) uint8
    """
    in_features = unpacked_weights.shape[0]

    # Build inverse permutation: sort by g_idx to get sequential group order
    sorted_indices = torch.argsort(g_idx)
    reordered = unpacked_weights[sorted_indices]

    return reordered


# =========================================================================
# AWQ checkpoint loader
# =========================================================================

def _find_quantized_layers(
    state_dict: Dict[str, torch.Tensor],
    suffix: str = ".qweight",
) -> list:
    """
    Find all quantized layer prefixes in a state dict.

    Returns list of prefixes like "model.layers.0.self_attn.q_proj."
    """
    prefixes = []
    for key in state_dict:
        if key.endswith(suffix):
            prefix = key[: -len(suffix) + 1]  # keep trailing dot
            prefixes.append(prefix)
    return sorted(set(prefixes))


def load_awq_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> Dict:
    """
    Load AWQ pre-quantized checkpoint and convert to internal INT4 format.

    Loads qweight/qzeros/scales from the AWQ checkpoint, unpacks from
    uint32-packed int4 to our internal format (packed uint8, scales, zeros),
    and assigns to model parameters/buffers.

    Non-quantized weights (layernorm, embeddings) are loaded directly.

    Args:
        model: model to load into
        checkpoint_path: path to HuggingFace AWQ checkpoint directory
        device: target device
        dtype: dtype for non-quantized weights

    Returns:
        dict with loading statistics
    """
    from vllm_i64.core.loader import _load_state_dict

    path = Path(checkpoint_path)
    detection = detect_quant_config(checkpoint_path)
    if detection is None:
        raise ValueError(f"No AWQ quantization_config found in {checkpoint_path}/config.json")

    quant_method, qcfg = detection
    if quant_method != "awq":
        raise ValueError(f"Expected AWQ checkpoint, got: {quant_method}")

    if qcfg.bits != 4:
        raise ValueError(f"Only 4-bit AWQ supported, got {qcfg.bits}-bit")

    group_size = qcfg.group_size
    _logger.info("Loading AWQ checkpoint from %s (group_size=%d)", checkpoint_path, group_size)

    state_dict = _load_state_dict(checkpoint_path)

    # Separate quantized layers from non-quantized weights
    quant_prefixes = _find_quantized_layers(state_dict, ".qweight")
    loaded_keys = set()
    quant_layers = 0
    non_quant_loaded = 0

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Process quantized layers
    for prefix in quant_prefixes:
        qweight_key = prefix + "qweight"
        qzeros_key = prefix + "qzeros"
        scales_key = prefix + "scales"

        if qweight_key not in state_dict:
            _logger.warning("Missing qweight for %s, skipping", prefix)
            continue
        if scales_key not in state_dict:
            _logger.warning("Missing scales for %s, skipping", prefix)
            continue

        qweight = state_dict[qweight_key]
        scales = state_dict[scales_key]
        qzeros = state_dict.get(qzeros_key)

        # Unpack AWQ format
        unpacked_w = _unpack_awq_qweight(qweight)
        if qzeros is not None:
            unpacked_z = _unpack_awq_qzeros(qzeros)
            # AWQ convention: stored zeros need +1 offset correction
            # AutoAWQ subtracts 1 during packing, so we add it back
            unpacked_z = (unpacked_z + 1).clamp(0, 15).to(torch.uint8)
        else:
            # No zero points — use 8 (midpoint for unsigned 4-bit)
            num_groups = scales.shape[0]
            out_features = scales.shape[1]
            unpacked_z = torch.full((num_groups, out_features), 8, dtype=torch.uint8)

        # Convert to our internal format
        packed, s, z = _convert_to_internal_int4(unpacked_w, scales, unpacked_z, group_size)

        # Resolve the model parameter name
        # AWQ uses names like "model.layers.0.self_attn.q_proj.qweight"
        # We need to find the corresponding module and register INT4 buffers
        layer_name = prefix.rstrip(".")
        # Load bias if present
        bias_key = prefix + "bias"
        bias_tensor = state_dict.get(bias_key)
        _assign_int4_to_module(model, layer_name, packed, s, z, group_size, device, bias=bias_tensor)

        loaded_keys.update([qweight_key, scales_key])
        if qzeros_key in state_dict:
            loaded_keys.add(qzeros_key)
        if bias_key in state_dict:
            loaded_keys.add(bias_key)
        quant_layers += 1

    # Load non-quantized weights (layernorm, embeddings, etc.)
    for name, weight in state_dict.items():
        if name in loaded_keys:
            continue

        # Skip g_idx (GPTQ-only, but some AWQ exports include it)
        if name.endswith(".g_idx"):
            loaded_keys.add(name)
            continue

        # Skip rotary inv_freq
        if "rotary_emb.inv_freq" in name:
            loaded_keys.add(name)
            continue

        # Resolve parameter name
        resolved = name
        if resolved not in params:
            resolved = name.replace("model.", "", 1)

        if resolved in params:
            params[resolved].data.copy_(weight.to(dtype))
            loaded_keys.add(name)
            non_quant_loaded += 1
        elif resolved in buffers:
            buffers[resolved].data.copy_(weight)
            loaded_keys.add(name)
            non_quant_loaded += 1
        else:
            _logger.debug("Skipping unrecognized key: %s", name)

    unloaded = set(state_dict.keys()) - loaded_keys
    stats = {
        "quant_layers": quant_layers,
        "non_quant_loaded": non_quant_loaded,
        "unloaded_keys": len(unloaded),
        "format": "awq",
        "group_size": group_size,
    }

    _logger.info(
        "AWQ load complete: %d quantized layers, %d non-quantized weights, %d unloaded keys",
        quant_layers, non_quant_loaded, len(unloaded),
    )
    if unloaded:
        _logger.debug("Unloaded keys: %s", sorted(unloaded))

    return stats


# =========================================================================
# GPTQ checkpoint loader
# =========================================================================

def load_gptq_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
) -> Dict:
    """
    Load GPTQ pre-quantized checkpoint and convert to internal INT4 format.

    Loads qweight/qzeros/scales/g_idx from the GPTQ checkpoint, unpacks
    from int32-packed int4 to our internal format (packed uint8, scales, zeros),
    and assigns to model parameters/buffers.

    Supports desc_act=True (activation-order reordering via g_idx) and
    desc_act=False (sequential groups).

    Args:
        model: model to load into
        checkpoint_path: path to HuggingFace GPTQ checkpoint directory
        device: target device
        dtype: dtype for non-quantized weights

    Returns:
        dict with loading statistics
    """
    from vllm_i64.core.loader import _load_state_dict

    path = Path(checkpoint_path)
    detection = detect_quant_config(checkpoint_path)
    if detection is None:
        raise ValueError(f"No GPTQ quantization_config found in {checkpoint_path}/config.json")

    quant_method, qcfg = detection
    if quant_method != "gptq":
        raise ValueError(f"Expected GPTQ checkpoint, got: {quant_method}")

    if qcfg.bits != 4:
        raise ValueError(f"Only 4-bit GPTQ supported, got {qcfg.bits}-bit")

    group_size = qcfg.group_size
    desc_act = qcfg.desc_act
    _logger.info(
        "Loading GPTQ checkpoint from %s (group_size=%d, desc_act=%s)",
        checkpoint_path, group_size, desc_act,
    )

    state_dict = _load_state_dict(checkpoint_path)

    quant_prefixes = _find_quantized_layers(state_dict, ".qweight")
    loaded_keys = set()
    quant_layers = 0
    non_quant_loaded = 0

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Process quantized layers
    for prefix in quant_prefixes:
        qweight_key = prefix + "qweight"
        qzeros_key = prefix + "qzeros"
        scales_key = prefix + "scales"
        g_idx_key = prefix + "g_idx"

        if qweight_key not in state_dict:
            _logger.warning("Missing qweight for %s, skipping", prefix)
            continue
        if scales_key not in state_dict:
            _logger.warning("Missing scales for %s, skipping", prefix)
            continue

        qweight = state_dict[qweight_key]
        scales = state_dict[scales_key]
        qzeros = state_dict.get(qzeros_key)
        g_idx = state_dict.get(g_idx_key)

        # Unpack GPTQ format
        unpacked_w = _unpack_gptq_qweight(qweight)
        in_features = unpacked_w.shape[0]
        out_features = unpacked_w.shape[1]
        num_groups = in_features // group_size

        if qzeros is not None:
            unpacked_z = _unpack_gptq_qzeros(qzeros)
            # GPTQ convention: stored zeros are offset by 1 in most implementations
            # The actual zero point = stored_value + 1
            unpacked_z = (unpacked_z + 1).clamp(0, 15).to(torch.uint8)
        else:
            # Symmetric quantization — zero point at midpoint
            unpacked_z = torch.full(
                (num_groups, out_features), 8, dtype=torch.uint8,
            )

        # Handle desc_act reordering
        if desc_act and g_idx is not None:
            unpacked_w = _reorder_by_g_idx(unpacked_w, g_idx, group_size, num_groups)
            _logger.debug("Applied desc_act reorder for %s", prefix)

        # Convert to our internal format
        packed, s, z = _convert_to_internal_int4(unpacked_w, scales, unpacked_z, group_size)

        # Assign to model
        layer_name = prefix.rstrip(".")
        bias_key = prefix + "bias"
        bias_tensor = state_dict.get(bias_key)
        _assign_int4_to_module(model, layer_name, packed, s, z, group_size, device, bias=bias_tensor)

        loaded_keys.update([qweight_key, scales_key])
        if qzeros_key in state_dict:
            loaded_keys.add(qzeros_key)
        if g_idx_key in state_dict:
            loaded_keys.add(g_idx_key)
        if bias_key in state_dict:
            loaded_keys.add(bias_key)
        quant_layers += 1

    # Load non-quantized weights
    for name, weight in state_dict.items():
        if name in loaded_keys:
            continue

        if "rotary_emb.inv_freq" in name:
            loaded_keys.add(name)
            continue

        resolved = name
        if resolved not in params:
            resolved = name.replace("model.", "", 1)

        if resolved in params:
            params[resolved].data.copy_(weight.to(dtype))
            loaded_keys.add(name)
            non_quant_loaded += 1
        elif resolved in buffers:
            buffers[resolved].data.copy_(weight)
            loaded_keys.add(name)
            non_quant_loaded += 1
        else:
            _logger.debug("Skipping unrecognized key: %s", name)

    unloaded = set(state_dict.keys()) - loaded_keys
    stats = {
        "quant_layers": quant_layers,
        "non_quant_loaded": non_quant_loaded,
        "unloaded_keys": len(unloaded),
        "format": "gptq",
        "group_size": group_size,
        "desc_act": desc_act,
    }

    _logger.info(
        "GPTQ load complete: %d quantized layers, %d non-quantized weights, %d unloaded keys",
        quant_layers, non_quant_loaded, len(unloaded),
    )
    if unloaded:
        _logger.debug("Unloaded keys: %s", sorted(unloaded))

    return stats


# =========================================================================
# Internal: assign INT4 buffers to model modules
# =========================================================================

def _resolve_module_name(model: nn.Module, layer_name: str) -> Tuple[Optional[nn.Module], str]:
    """
    Resolve a checkpoint layer name to a model module.

    Handles:
      - "model.layers.0.self_attn.q_proj" → module at that path
      - Stripping "model." prefix if needed
      - TP-wrapped ".linear" suffix

    Returns (module, param_base_name) or (None, "") if not found.
    """
    # Try direct path first
    parts = layer_name.split(".")
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            # Try without "model." prefix
            stripped = layer_name.replace("model.", "", 1)
            module = model
            for p in stripped.split("."):
                if hasattr(module, p):
                    module = getattr(module, p)
                else:
                    return None, ""
            return module, stripped

    return module, layer_name


def _assign_int4_to_module(
    model: nn.Module,
    layer_name: str,
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
    device: str,
    bias: Optional[torch.Tensor] = None,
):
    """
    Register INT4 quantized buffers on the appropriate model module.

    Finds the nn.Module corresponding to layer_name and registers:
      - {leaf}_int4: packed uint8 weights
      - {leaf}_scale_int4: per-group scales
      - {leaf}_zero: per-group zero points
      - {leaf}_bias: bias tensor (if provided)

    Args:
        model: the model
        layer_name: full dotted name (e.g. "model.layers.0.self_attn.q_proj")
        packed: (out, in//2) uint8
        scales: (out, groups) float
        zeros:  (out, groups) float
        group_size: group size
        device: target device
        bias: optional bias tensor
    """
    module, resolved_name = _resolve_module_name(model, layer_name)

    if module is None:
        _logger.warning("Could not resolve module for %s, skipping", layer_name)
        return

    # Get the leaf name (e.g. "q_proj" from "model.layers.0.self_attn.q_proj")
    parts = resolved_name.split(".")
    leaf = parts[-1] if parts else layer_name.split(".")[-1]

    # Find the parent module to register buffers on
    parent = model
    parent_parts = parts[:-1] if len(parts) > 1 else []
    for p in parent_parts:
        if hasattr(parent, p):
            parent = getattr(parent, p)

    # Register INT4 buffers on the parent module
    parent.register_buffer(f"{leaf}_int4", packed.to(device))
    parent.register_buffer(f"{leaf}_scale_int4", scales.to(device))
    parent.register_buffer(f"{leaf}_zero", zeros.to(device))

    # Load bias if provided
    if bias is not None:
        parent.register_buffer(f"{leaf}_bias", bias.to(device).float())

    # Store group_size as attribute for int4_linear() calls
    if not hasattr(parent, "_int4_group_size"):
        parent._int4_group_size = group_size

    # Free the original float weight if it exists
    if hasattr(module, "weight") and module.weight is not None:
        module.weight = None
    # Also check TP-wrapped .linear.weight
    if hasattr(module, "linear") and hasattr(module.linear, "weight"):
        module.linear.weight = None

    _logger.debug("Assigned INT4 buffers for %s on parent %s", leaf, type(parent).__name__)
