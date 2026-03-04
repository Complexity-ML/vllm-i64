"""
vllm-i64 :: Qwen2ForCausalLM

Qwen2 = Llama-compatible architecture with different defaults.
Imports from llama — same as vllm does.

dense = modulo 1 in i64 routing theory.

INL - 2025
"""

from vllm_i64.models.llama.model import LlamaForCausalLM


class Qwen2ForCausalLM(LlamaForCausalLM):
    """Qwen2 model. Same architecture as Llama."""
    pass
