"""
Complexity-I64 (Integer-native) model for vllm-i64.

Same architecture as Complexity Deep, same checkpoint format.
Architecture string: I64ForCausalLM
Model type: complexity-i64

Uses ComplexityDeepModel for inference — the INT8 quantization
(LUT activations, integer dynamics) is applied via the loader.
"""

from vllm_i64.models.complexity_i64.config import ComplexityI64Config
