"""
vllm-i64 :: Model Registry

Generic registry for ANY token-routed model.
Not tied to a specific architecture — register anything
that uses i64 token routing.

To add a new model:
    1. Create its config class (like ComplexityDeepConfig)
    2. Create its model class using layers/token_routed_mlp.py
    3. Register it: register_model("my-model", ...)

INL - 2025
"""

from typing import Dict, Optional
from dataclasses import dataclass


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
# Built-in registrations: Complexity Deep (pacific-prime)
# Other models can call register_model() to add themselves.
# =========================================================================

register_model(
    name="pacific-i64",
    model_class="vllm_i64.models.complexity_deep.model.ComplexityDeepModel",
    config_loader="vllm_i64.models.complexity_deep.config.ComplexityDeepConfig",
    config_path=None,
    checkpoint=None,
    parameters="~1.58B",
    description="Complexity Deep i64 — generic entry, use --checkpoint to specify model",
)

register_model(
    name="pacific-prime",
    model_class="vllm_i64.models.complexity_deep.model.ComplexityDeepModel",
    config_loader="vllm_i64.models.complexity_deep.config.ComplexityDeepConfig",
    config_path="checkpoints/config.json",
    checkpoint="checkpoints/final.pt",
    parameters="~1.47B",
    description="Complexity Deep v0.13.0 base model",
)

register_model(
    name="pacific-prime-chat",
    model_class="vllm_i64.models.complexity_deep.model.ComplexityDeepModel",
    config_loader="vllm_i64.models.complexity_deep.config.ComplexityDeepConfig",
    config_path="checkpoints/pacific-prime-chat/converted/fp16/config.json",
    checkpoint="checkpoints/pacific-prime-chat/converted/fp16",
    parameters="~1.58B",
    description="Complexity Deep v0.13.0 chat fine-tune (FP16 safetensors)",
)

register_model(
    name="pacific-chat",
    model_class="vllm_i64.models.complexity_deep.model.ComplexityDeepModel",
    config_loader="vllm_i64.models.complexity_deep.config.ComplexityDeepConfig",
    config_path=None,
    checkpoint=None,
    parameters="~1.58B",
    description="Complexity Deep v0.13.0 chat — alias, use --checkpoint to specify model",
)

register_model(
    name="pacific-prime-python",
    model_class="vllm_i64.models.complexity_deep.model.ComplexityDeepModel",
    config_loader="vllm_i64.models.complexity_deep.config.ComplexityDeepConfig",
    config_path="checkpoints/pacific-prime-python/converted/fp16/config.json",
    checkpoint="checkpoints/pacific-prime-python/converted/fp16",
    parameters="~1.58B",
    description="Complexity Deep v0.13.0 python fine-tune (FP16 safetensors)",
)
