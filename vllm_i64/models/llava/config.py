"""
vllm-i64 :: LLaVA Config

Extends LlamaConfig with vision-language fields for LLaVA-style models.

Fields:
  - vision_tower: HF model name for the vision encoder
  - mm_projector_type: "linear" or "mlp2x_gelu"
  - image_token_index: token ID for the <image> placeholder
  - mm_vision_select_layer: which vision layer to extract features from

INL - 2025
"""

import json
from typing import Optional

from vllm_i64.models.llama.config import LlamaConfig


class LlavaConfig(LlamaConfig):
    """
    LLaVA config — LlamaConfig + vision fields.

    Inherits all Llama fields (hidden_size, num_heads, etc.) and adds
    multimodal configuration for the vision tower and projector.
    """

    model_type = "llava"

    def __init__(self, **kwargs):
        # Vision-specific fields (pop before passing to LlamaConfig)
        self.vision_tower: Optional[str] = kwargs.pop(
            "vision_tower", "openai/clip-vit-large-patch14-336"
        )
        self.mm_projector_type: str = kwargs.pop("mm_projector_type", "mlp2x_gelu")
        self.image_token_index: int = kwargs.pop("image_token_index", 32000)
        self.mm_vision_select_layer: int = kwargs.pop("mm_vision_select_layer", -2)
        self.freeze_vision_tower: bool = kwargs.pop("freeze_vision_tower", True)

        # Handle mm_hidden_size (vision hidden size, for reference)
        self.mm_hidden_size: Optional[int] = kwargs.pop("mm_hidden_size", None)

        super().__init__(**kwargs)

    @classmethod
    def from_json(cls, path: str) -> "LlavaConfig":
        """Load config from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
