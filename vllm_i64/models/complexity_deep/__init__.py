"""
Complexity Deep (Pacific-Prime) model for vllm-i64.
"""

from vllm_i64.models.complexity_deep.config import ComplexityDeepConfig
from vllm_i64.models.complexity_deep.model import (
    ComplexityDeepModel,
    ComplexityDecoderLayer,
    MuGuidedAttention,
    MuGuidedTokenRoutedMLP,
    INLDynamics,
)
