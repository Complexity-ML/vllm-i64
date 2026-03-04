"""
vllm-i64 :: MistralForCausalLM

Mistral = Llama architecture + sliding window + GQA (8 KV heads).
Imports from llama — same as vllm does.

dense = modulo 1 in i64 routing theory.

INL - 2025
"""

from vllm_i64.models.llama.model import LlamaForCausalLM


class MistralForCausalLM(LlamaForCausalLM):
    """Mistral model. Same architecture as Llama."""
    pass
