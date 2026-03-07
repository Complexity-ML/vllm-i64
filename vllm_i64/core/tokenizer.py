"""
vllm-i64 :: Tokenizer

Load tokenizer from checkpoint directory.
Wraps HuggingFace tokenizers for i64 ↔ text conversion.

INL - 2025
"""

import logging
import os
from typing import Optional, List

logger = logging.getLogger("vllm_i64.tokenizer")

from vllm_i64.core.registry import get_model_entry


class I64Tokenizer:
    """
    Tokenizer wrapper.

    Input:  text (str)
    Output: token IDs (List[int] — i64)

    Uses tokenizers library (HuggingFace fast tokenizer).
    """

    def __init__(self, tokenizer_path: str):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text: str) -> List[int]:
        """Text → i64 token IDs. Strips trailing EOS (not part of prompt)."""
        ids = self.tokenizer.encode(text).ids
        # Strip trailing EOS — the model should not see EOS in the input prompt
        eos = self.eos_token_id
        if ids and ids[-1] == eos:
            ids = ids[:-1]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """i64 token IDs → text."""
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def _find_token(self, candidates: list, fallback: int) -> int:
        """Try multiple token names, return first match or fallback."""
        for name in candidates:
            tid = self.tokenizer.token_to_id(name)
            if tid is not None:
                return tid
        return fallback

    @property
    def eos_token_id(self) -> int:
        return self._find_token(
            ["</s>", "<|endoftext|>", "<|end|>", "<eos>", "<|eot_id|>"], 0
        )

    @property
    def bos_token_id(self) -> int:
        return self._find_token(
            ["<s>", "<|startoftext|>", "<|begin|>", "<bos>", "<|begin_of_text|>"], 0
        )

    @property
    def pad_token_id(self) -> int:
        return self._find_token(
            ["<pad>", "<|pad|>", "<|padding|>"], self.eos_token_id
        )


def load_tokenizer(model_name: str) -> Optional[I64Tokenizer]:
    """
    Load tokenizer for a registered model.

    Looks for tokenizer.json in the checkpoint directory.
    """
    entry = get_model_entry(model_name)
    if not entry.config_path:
        return None

    # Look for tokenizer.json next to config.json
    config_dir = os.path.dirname(entry.config_path)
    tokenizer_path = os.path.join(config_dir, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        logger.info("Tokenizer: %s", tokenizer_path)
        return I64Tokenizer(tokenizer_path)

    # Try parent directory
    parent_tokenizer = os.path.join(os.path.dirname(config_dir), "tokenizer.json")
    if os.path.exists(parent_tokenizer):
        logger.info("Tokenizer: %s", parent_tokenizer)
        return I64Tokenizer(parent_tokenizer)

    logger.warning("Tokenizer not found (using byte fallback)")
    return None
