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
_COMPLEXITY_I64 = (
    "vllm_i64.models.complexity_deep.model.ComplexityDeepModel",
    "vllm_i64.models.complexity_i64.config.ComplexityI64Config",
)
_LLAVA = (
    "vllm_i64.models.llava.model.LlavaForConditionalGeneration",
    "vllm_i64.models.llava.config.LlavaConfig",
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
    # --- i64 integer-native ---
    "I64ForCausalLM": _COMPLEXITY_I64,
    # --- Vision-Language Models ---
    "LlavaForConditionalGeneration": _LLAVA,
    "LlavaNextForConditionalGeneration": _LLAVA,
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
        with open(config_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except (OSError, ValueError, _json.JSONDecodeError):
        return None

    architectures = data.get("architectures", [])
    for arch in architectures:
        if arch in _ARCHITECTURE_MAP:
            model_class, config_loader = _ARCHITECTURE_MAP[arch]
            _logger.info("Auto-detected architecture: %s → %s", arch, model_class.rsplit('.', 1)[-1])
            return model_class, config_loader, str(config_path)

    _logger.warning(
        "Unknown architecture(s): %s. Supported: %s",
        architectures, sorted(_ARCHITECTURE_MAP.keys()),
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

# --- Complexity-I64 (integer-native) ---

register_model(
    name="complexity-i64",
    model_class=_COMPLEXITY_I64[0],
    config_loader=_COMPLEXITY_I64[1],
    description="Complexity-I64 integer-native model — use --checkpoint to specify model",
)

# --- Ablation runs (TMLR paper) ---
# run1: dense baseline (no routing, no dynamics)
register_model(
    name="run1-dense",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    config_path="C:/INL/pacific-prime/run1-dense/final/config.json",
    checkpoint="C:/INL/pacific-prime/run1-dense/final",
    parameters="~150M",
    description="Ablation: dense SwigLU baseline (no token routing, no dynamics)",
)

# run2: full architecture (token-routed + mu + pid)
register_model(
    name="run2-full",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    config_path="C:/INL/pacific-prime/run2-full/final/config.json",
    checkpoint="C:/INL/pacific-prime/run2-full/final",
    parameters="~150M",
    description="Ablation: full Complexity Deep (token-routed + mu + pid)",
)

# run3: no mu guidance (token-routed + pid only)
register_model(
    name="run3-no-mu",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    config_path="C:/INL/pacific-prime/run3-no-mu/final/config.json",
    checkpoint="C:/INL/pacific-prime/run3-no-mu/final",
    parameters="~150M",
    description="Ablation: no mu guidance (token-routed + pid, mu bypassed)",
)

# run4: no PID scaler (token-routed + mu only)
register_model(
    name="run4-no-pid",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    config_path="C:/INL/pacific-prime/run4-no-pid/final/config.json",
    checkpoint="C:/INL/pacific-prime/run4-no-pid/final",
    parameters="~150M",
    description="Ablation: no PID scaler (token-routed + mu only)",
)

# --- Pacific Tiny Chat (space model) ---
register_model(
    name="pacific-tiny-chat",
    model_class=_COMPLEXITY_DEEP[0],
    config_loader=_COMPLEXITY_DEEP[1],
    config_path="C:/INL/pacific-prime/pacific-tiny-chat/config.json",
    checkpoint="C:/INL/pacific-prime/pacific-tiny-chat",
    parameters="~170M",
    description="Pacific Tiny Chat — token-routed i64 (space model)",
)

# --- Vision-Language Models ---

register_model(
    name="llava",
    model_class=_LLAVA[0],
    config_loader=_LLAVA[1],
    description="LLaVA vision-language model — use --checkpoint for HF dir",
)


# =========================================================================
# Plugin discovery via entry points
#
# Third-party packages can register models by declaring entry points:
#
#   [project.entry-points."vllm_i64.models"]
#   my-model = "my_package.model:register"
#
# The entry point must be a callable that receives no arguments
# and calls register_model() and/or updates _ARCHITECTURE_MAP.
# =========================================================================

def _discover_plugins():
    """Load model plugins from installed packages via entry points."""
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return

    try:
        eps = entry_points(group="vllm_i64.models")
    except TypeError:
        # Python 3.9 compat (shouldn't happen with >=3.10 but defensive)
        eps = entry_points().get("vllm_i64.models", [])

    for ep in eps:
        try:
            register_fn = ep.load()
            register_fn()
            _logger.info("Loaded model plugin: %s", ep.name)
        except Exception:
            _logger.warning("Failed to load model plugin: %s", ep.name)


_discover_plugins()
