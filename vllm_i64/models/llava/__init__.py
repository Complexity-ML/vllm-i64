"""
vllm-i64 :: LLaVA Vision-Language Model

LLaVA-style VLM combining a vision encoder with a Llama language model.
Supports multimodal input with <image> token placeholders.

INL - 2025
"""

from vllm_i64.models.llava.model import LlavaForConditionalGeneration
from vllm_i64.models.llava.config import LlavaConfig

__all__ = ["LlavaForConditionalGeneration", "LlavaConfig"]
