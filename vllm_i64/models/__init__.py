"""
Model implementations for vllm-i64.
Token-routed models with pre-computed i64 expert routing.
"""

from vllm_i64.models.token_routed_model import TokenRoutedModel, TokenRoutedConfig
from vllm_i64.models.model_registry import get_model, list_models, MODEL_CONFIGS
