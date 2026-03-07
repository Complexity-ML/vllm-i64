"""
vllm-i64 :: Vision Encoder

Vision tower for VLM support — wraps pretrained CLIP/SigLIP models
and projects image features to LLM hidden dimension.

Supports:
  - CLIPVisionModel (OpenAI CLIP family)
  - SiglipVisionModel (Google SigLIP family)
  - Linear or MLP projector to match LLM hidden_size
  - Image preprocessing (resize, normalize)

INL - 2025
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("vllm_i64.layers.vision")


class VisionEncoder(nn.Module):
    """
    Vision encoder that wraps a pretrained CLIP or SigLIP vision tower
    and projects image features into the LLM embedding space.

    Args:
        vision_tower_name: HuggingFace model name/path for the vision tower
            (e.g. "openai/clip-vit-large-patch14-336").
        hidden_size: LLM hidden dimension to project into.
        mm_projector_type: Projector type — "linear" or "mlp2x_gelu".
        freeze_vision: Whether to freeze vision tower weights.
    """

    def __init__(
        self,
        vision_tower_name: str,
        hidden_size: int,
        mm_projector_type: str = "mlp2x_gelu",
        freeze_vision: bool = True,
    ):
        super().__init__()
        self.vision_tower_name = vision_tower_name
        self.hidden_size = hidden_size
        self.mm_projector_type = mm_projector_type

        # Load vision tower and processor
        self.vision_tower, self.image_processor, vision_hidden_size = (
            self._load_vision_tower(vision_tower_name)
        )

        if freeze_vision:
            self.vision_tower.requires_grad_(False)
            logger.info("Vision tower frozen: %s", vision_tower_name)

        # Build projector: vision_hidden_size → hidden_size
        self.mm_projector = self._build_projector(
            vision_hidden_size, hidden_size, mm_projector_type,
        )

        logger.info(
            "VisionEncoder: tower=%s, vision_dim=%d, llm_dim=%d, projector=%s",
            vision_tower_name, vision_hidden_size, hidden_size, mm_projector_type,
        )

    @staticmethod
    def _load_vision_tower(name: str) -> Tuple[nn.Module, object, int]:
        """
        Load a pretrained vision tower and its image processor.

        Returns: (vision_model, image_processor, hidden_size)
        """
        from transformers import AutoImageProcessor

        # Try SigLIP first, then CLIP
        try:
            from transformers import SiglipVisionModel
            tower = SiglipVisionModel.from_pretrained(name)
            processor = AutoImageProcessor.from_pretrained(name)
            vis_hidden = tower.config.hidden_size
            logger.info("Loaded SigLIP vision tower: %s (dim=%d)", name, vis_hidden)
            return tower, processor, vis_hidden
        except Exception:
            pass

        try:
            from transformers import CLIPVisionModel
            tower = CLIPVisionModel.from_pretrained(name)
            processor = AutoImageProcessor.from_pretrained(name)
            vis_hidden = tower.config.hidden_size
            logger.info("Loaded CLIP vision tower: %s (dim=%d)", name, vis_hidden)
            return tower, processor, vis_hidden
        except Exception as e:
            raise RuntimeError(
                f"Failed to load vision tower '{name}'. "
                f"Ensure it is a valid CLIP or SigLIP model. Error: {e}"
            ) from e

    @staticmethod
    def _build_projector(
        in_dim: int, out_dim: int, projector_type: str,
    ) -> nn.Module:
        """Build the multimodal projector."""
        if projector_type == "linear":
            return nn.Linear(in_dim, out_dim)
        elif projector_type == "mlp2x_gelu":
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
        else:
            raise ValueError(
                f"Unknown projector type: {projector_type}. "
                f"Supported: 'linear', 'mlp2x_gelu'"
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images and project to LLM hidden dimension.

        Args:
            pixel_values: Preprocessed image tensor
                shape (batch, channels, height, width).

        Returns:
            image_features: shape (batch, num_patches, hidden_size).
        """
        # Run vision tower — extract last hidden state (patch embeddings)
        with torch.no_grad() if not self.vision_tower.training else torch.enable_grad():
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values.to(
                    dtype=self.vision_tower.dtype,
                    device=self.vision_tower.device,
                ),
            )

        # Use last_hidden_state, skip CLS token (index 0) if present
        image_features = vision_outputs.last_hidden_state
        if hasattr(self.vision_tower.config, "num_image_tokens"):
            # SigLIP: all tokens are patch tokens
            pass
        else:
            # CLIP: first token is CLS, skip it
            image_features = image_features[:, 1:, :]

        # Project to LLM dimension
        image_features = self.mm_projector(image_features)
        return image_features

    def preprocess_image(self, image) -> torch.Tensor:
        """
        Preprocess a PIL Image for the vision tower.

        Args:
            image: PIL.Image.Image instance.

        Returns:
            pixel_values: tensor of shape (1, C, H, W).
        """
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs["pixel_values"]

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the vision tower."""
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Return the device of the vision tower."""
        return next(self.vision_tower.parameters()).device

    def num_parameters(self) -> int:
        """Total parameter count (vision tower + projector)."""
        return sum(p.numel() for p in self.parameters())
