"""
vllm-i64 :: Model Registry

Registry for i64 models (token-routed MoE + dense float).

Two ways to resolve a model:
  1. By name: register_model("pacific-prime", ...) → get_model_entry("pacific-prime")
  2. By HF architecture: config.json → architectures: ["LlamaForCausalLM"]
     → auto-detect via _ARCHITECTURE_MAP → LlamaForCausalLM

INL - 2025
"""

import json as _json
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger("vllm_i64.registry")


@dataclass
class ModelEntry:
    """A registered model."""
    name: str
    model_class: str           # e.g. "ComplexityDeepModel"
    config_loader: str         # e.g. "complexity_deep_config.ComplexityDeepConfig"
    config_path: Optional[str] # path to config.json
    checkpoint: Optional[str]  # path to .pt file
    parameters: str            # e.g. "~1.47B"
    description: str


# =========================================================================
# Global registry
# =========================================================================

_REGISTRY: Dict[str, ModelEntry] = {}


def register_model(
    name: str,
    model_class: str,
    config_loader: str,
    config_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    parameters: str = "",
    description: str = "",
):
    """
    Register a token-routed model.

    Args:
        name: unique model name (e.g. "pacific-prime-chat")
        model_class: dotted path to class (e.g. "vllm_i64.models.complexity_deep.model.ComplexityDeepModel")
        config_loader: dotted path to config class
        config_path: path to config.json (optional)
        checkpoint: path to checkpoint file (optional)
        parameters: human-readable param count
        description: model description
    """
    _REGISTRY[name] = ModelEntry(
        name=name,
        model_class=model_class,
        config_loader=config_loader,
        config_path=config_path,
        checkpoint=checkpoint,
        parameters=parameters,
        description=description,
    )


def get_model_entry(name: str) -> ModelEntry:
    """Get a registered model entry."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return _REGISTRY[name]


def get_checkpoint_path(name: str) -> Optional[str]:
    """Get checkpoint path for a model."""
    return get_model_entry(name).checkpoint


def list_models() -> list:
    """List all registered models."""
    return [
        {
            "name": e.name,
            "model_class": e.model_class,
            "parameters": e.parameters,
            "description": e.description,
        }
        for e in _REGISTRY.values()
    ]


# =========================================================================
# HuggingFace architecture → (model_class, config_loader) mapping
#
# When a user points --checkpoint at a HF model dir, we read config.json,
# look at "architectures", and resolve to our model class.
#
# dense = modulo 1 in i64 routing theory.
# =========================================================================

_LLAMA = (
    "vllm_i64.models.llama.model.LlamaForCausalLM",
    "vllm_i64.models.llama.config.LlamaConfig",
)
_MISTRAL = (
    "vllm_i64.models.mistral.model.MistralForCausalLM",
    "vllm_i64.models.mistral.config.MistralConfig",
)
_QWEN2 = (
    "vllm_i64.models.qwen2.model.Qwen2ForCausalLM",
    "vllm_i64.models.qwen2.config.Qwen2Config",
)
_MIXTRAL = (
    "vllm_i64.models.mixtral.model.MixtralForCausalLM",
    "vllm_i64.models.mixtral.config.MixtralConfig",
)
_COMPLEXITY_DEEP = (
    "vllm_i64.models.complexity_deep.model.ComplexityDeepModel",
    "vllm_i64.models.complexity_deep.config.ComplexityDeepConfig",
)

# Map: HF architectures string → (model_class, config_loader)
# Each model imported one by one.
_ARCHITECTURE_MAP: Dict[str, Tuple[str, str]] = {
    # --- Dense (modulo 1) ---
    "LlamaForCausalLM": _LLAMA,
    "LLaMAForCausalLM": _LLAMA,
    "MistralForCausalLM": _MISTRAL,
    "Qwen2ForCausalLM": _QWEN2,
    # Llama-compatible
    "GemmaForCausalLM": _LLAMA,
    "Gemma2ForCausalLM": _LLAMA,
    "Phi3ForCausalLM": _LLAMA,
    "InternLMForCausalLM": _LLAMA,
    "InternLM3ForCausalLM": _LLAMA,
    "YiForCausalLM": _LLAMA,
    "AquilaForCausalLM": _LLAMA,
    "AquilaModel": _LLAMA,
    "XverseForCausalLM": _LLAMA,
    # --- Sparse MoE ---
    "MixtralForCausalLM": _MIXTRAL,
    # --- i64 token-routed MoE ---
    "DeepForCausalLM": _COMPLEXITY_DEEP,
}


def resolve_architecture(checkpoint_path: str) -> Optional[Tuple[str, str, str]]:
    """
    Auto-detect model architecture from a HuggingFace checkpoint directory.

    Reads config.json, checks "architectures", and maps to our model class.

    Returns: (model_class, config_loader, config_path) or None if not found.
    """
    ckpt = Path(checkpoint_path)
    config_path = ckpt / "config.json" if ckpt.is_dir() else None

    if config_path is None or not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            data = _json.load(f)
    except Exception:
        return None

    architectures = data.get("architectures", [])
    for arch in architectures:
        if arch in _ARCHITECTURE_MAP:
            model_class, config_loader = _ARCHITECTURE_MAP[arch]
            _logger.info(f"Auto-detected architecture: {arch} → {model_class.rsplit('.', 1)[-1]}")
            return model_class, config_loader, str(config_path)

    _logger.warning(
        f"Unknown architecture(s): {architectures}. "
        f"Supported: {sorted(_ARCHITECTURE_MAP.keys())}"
    )
    return None


# =========================================================================
# Built-in registrations
# =========================================================================

# --- Dense models (modulo 1) — use --checkpoint to point at HF dir ---

register_model(
    name="llama",
    model_class=_LLAMA[0],
    config_loader=_LLAMA[1],
    description="Llama-family dense model — use --checkpoint for HF dir",
)

register_model(
    name="mistral",
    model_class=_MISTRAL[0],
    config_loader=_MISTRAL[1],
    description="Mistral dense model — use --checkpoint for HF dir",
)

register_model(
    name="qwen2",
    model_class=_QWEN2[0],
    config_loader=_QWEN2[1],
    description="Qwen2 dense model — use --checkpoint for HF dir",
)

# --- Sparse MoE ---

register_model(
    name="mixtral",
    model_class=_MIXTRAL[0],
    config_loader=_MIXTRAL[1],
    description="Mixtral sparse MoE — use --checkpoint for HF dir",
)

# --- Complexity Deep (i64 token-routed MoE) ---

register_model(
    name="pacific-i64",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    description="Complexity Deep i64 — generic entry, use --checkpoint to specify model",
)

register_model(
    name="pacific-prime",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    config_path="checkpoints/config.json",
    checkpoint="checkpoints/final.pt",
    parameters="~1.47B",
    description="Complexity Deep v0.13.0 base model",
)

register_model(
    name="pacific-prime-chat",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    config_path="checkpoints/pacific-prime-chat/converted/fp16/config.json",
    checkpoint="checkpoints/pacific-prime-chat/converted/fp16",
    parameters="~1.58B",
    description="Complexity Deep v0.13.0 chat fine-tune (FP16 safetensors)",
)

register_model(
    name="pacific-chat",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    description="Complexity Deep v0.13.0 chat — alias, use --checkpoint to specify model",
)

register_model(
    name="pacific-ros2",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    description="Complexity Deep v0.13.0 ROS2 specialist — use --checkpoint to specify model",
)

register_model(
    name="pacific-prime-python",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    config_path="checkpoints/pacific-prime-python/converted/fp16/config.json",
    checkpoint="checkpoints/pacific-prime-python/converted/fp16",
    parameters="~1.58B",
    description="Complexity Deep v0.13.0 python fine-tune (FP16 safetensors)",
)
