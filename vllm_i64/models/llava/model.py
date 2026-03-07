"""
vllm-i64 :: LlavaForConditionalGeneration

LLaVA-style Vision-Language Model:
  vision_encoder(image) → image_features
  embed(text_tokens) → text_embeddings
  merge at <image> positions → combined_embeddings
  LlamaForCausalLM(combined_embeddings) → logits

Architecture:
  VisionEncoder (CLIP/SigLIP) + Linear/MLP projector
  + LlamaForCausalLM (dense decoder)

The <image> token in the input is replaced with projected
image patch embeddings from the vision tower.

INL - 2025
"""

import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_i64.layers.vision import VisionEncoder
from vllm_i64.models.llama.model import LlamaForCausalLM

logger = logging.getLogger("vllm_i64.models.llava")


class LlavaForConditionalGeneration(nn.Module):
    """
    LLaVA-style vision-language model.

    Combines a VisionEncoder with a LlamaForCausalLM backbone.
    When pixel_values are provided, image features are merged into
    the text embedding sequence at <image> token positions.

    Same interface as LlamaForCausalLM:
      - forward(token_ids, positions, kv_cache, seq_ids, tokens_per_seq, pixel_values)
      - decode_step(token_ids, positions, kv_cache, seq_ids_tensor)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Language model backbone
        self.language_model = LlamaForCausalLM(config)

        # Vision encoder (lazy — only created if vision_tower is set)
        self.vision_tower_loaded = False
        self.vision_encoder: Optional[VisionEncoder] = None
        self.image_token_index = getattr(config, "image_token_index", 32000)

        vision_tower_name = getattr(config, "vision_tower", None)
        if vision_tower_name:
            self._init_vision_tower(config)

        logger.info(
            "LlavaForConditionalGeneration: vision=%s, image_token=%d",
            getattr(config, "vision_tower", "none"),
            self.image_token_index,
        )

    def _init_vision_tower(self, config) -> None:
        """Initialize the vision encoder from config."""
        self.vision_encoder = VisionEncoder(
            vision_tower_name=config.vision_tower,
            hidden_size=config.hidden_size,
            mm_projector_type=getattr(config, "mm_projector_type", "mlp2x_gelu"),
            freeze_vision=getattr(config, "freeze_vision_tower", True),
        )
        self.vision_tower_loaded = True
        logger.info("Vision tower initialized: %s", config.vision_tower)

    def _merge_image_features(
        self,
        text_embeddings: torch.Tensor,
        image_features: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace <image> token embeddings with projected image features.

        Args:
            text_embeddings: (total_tokens, hidden_size) from embed_tokens.
            image_features: (num_images, num_patches, hidden_size) from vision encoder.
            token_ids: (total_tokens,) original input token IDs.

        Returns:
            merged: (total_tokens_expanded, hidden_size) with image tokens replaced.
        """
        # Find <image> token positions
        image_mask = token_ids == self.image_token_index
        image_positions = torch.where(image_mask)[0]

        if len(image_positions) == 0:
            return text_embeddings

        num_images = image_features.shape[0]
        num_patches = image_features.shape[1]

        if len(image_positions) != num_images:
            logger.warning(
                "Image token count (%d) != image count (%d). "
                "Using min(%d, %d) images.",
                len(image_positions), num_images,
                len(image_positions), num_images,
            )

        num_to_merge = min(len(image_positions), num_images)

        # Build merged sequence: for each <image> token, insert num_patches
        # embeddings in place of the single token embedding
        result_parts: List[torch.Tensor] = []
        prev_idx = 0

        for i in range(num_to_merge):
            img_pos = image_positions[i].item()

            # Text before this <image> token
            if img_pos > prev_idx:
                result_parts.append(text_embeddings[prev_idx:img_pos])

            # Insert image patch embeddings
            result_parts.append(image_features[i])

            prev_idx = img_pos + 1

        # Remaining text after last <image>
        if prev_idx < text_embeddings.shape[0]:
            result_parts.append(text_embeddings[prev_idx:])

        merged = torch.cat(result_parts, dim=0)
        return merged

    def forward(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache=None,
        seq_ids: Optional[List[int]] = None,
        tokens_per_seq: Optional[List[int]] = None,
        intermediate_tensors=None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional image input.

        When pixel_values is provided:
          1. Encode images through vision tower
          2. Get text embeddings from embed_tokens
          3. Merge image features at <image> token positions
          4. Run merged embeddings through the decoder layers

        When pixel_values is None, behaves identically to LlamaForCausalLM.
        """
        from vllm_i64.parallel.pipeline_parallel import is_first_pp_rank, is_last_pp_rank
        from vllm_i64.parallel.pp_utils import IntermediateTensors

        if is_first_pp_rank() and pixel_values is not None and self.vision_encoder is not None:
            # Encode images
            image_features = self.vision_encoder(pixel_values)

            # Get text embeddings
            text_embeddings = self.language_model.embed_tokens(token_ids.long())

            # Merge image features into text embeddings
            hidden = self._merge_image_features(
                text_embeddings, image_features, token_ids,
            )

            # Adjust positions and tokens_per_seq for expanded sequence
            orig_len = text_embeddings.shape[0]
            new_len = hidden.shape[0]

            if new_len != orig_len:
                # Rebuild positions for the expanded sequence
                positions = torch.arange(new_len, device=positions.device, dtype=positions.dtype)

                # Adjust tokens_per_seq if provided
                if tokens_per_seq is not None:
                    expansion = new_len - orig_len
                    # Distribute expansion to the first sequence (simplification)
                    tokens_per_seq = list(tokens_per_seq)
                    tokens_per_seq[0] += expansion

            # Run through decoder layers directly (skip embed_tokens)
            for layer_idx in range(self.language_model.start_layer, self.language_model.end_layer):
                hidden = self.language_model.layers[layer_idx](
                    hidden, positions,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx,
                    seq_ids=seq_ids,
                    tokens_per_seq=tokens_per_seq,
                )

            if not is_last_pp_rank():
                return IntermediateTensors({"hidden_states": hidden})

            hidden = self.language_model.norm(hidden)

            if self.language_model.tie_word_embeddings and self.language_model.embed_tokens is not None:
                logits = F.linear(hidden, self.language_model.embed_tokens.weight)
            elif hasattr(self.language_model, 'lm_head'):
                logits = self.language_model.lm_head(hidden)
            else:
                raise RuntimeError(
                    "No lm_head or tied embeddings found — cannot compute logits."
                )

            return logits

        # No images — delegate to language model
        return self.language_model.forward(
            token_ids, positions,
            kv_cache=kv_cache,
            seq_ids=seq_ids,
            tokens_per_seq=tokens_per_seq,
            intermediate_tensors=intermediate_tensors,
        )

    def decode_step(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache,
        seq_ids_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode step — no image input during autoregressive decoding.

        Images are only processed during the prefill (forward) pass.
        Decode steps are pure text, delegated to the language model.
        """
        return self.language_model.decode_step(
            token_ids, positions, kv_cache, seq_ids_tensor,
        )

    def num_parameters(self) -> int:
        """Total parameter count (language model + vision encoder)."""
        return sum(p.numel() for p in self.parameters())
