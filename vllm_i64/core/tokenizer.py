"""
vllm-i64 :: Tokenizer

Load tokenizer from checkpoint directory.
Wraps HuggingFace tokenizers for i64 ↔ text conversion.

INL - 2025
"""

import logging
import os
import re
from typing import Optional, List

import torch

logger = logging.getLogger("vllm_i64.tokenizer")

from vllm_i64.core.registry import get_model_entry


# Patterns for artifact tokens that should be heavily penalized
_ARTIFACT_RE = re.compile(r'^\[\[|\]\]$|^<\|.*\|>$')
# Pattern for byte fallback tokens like <0xNN>
_BYTE_FALLBACK_RE = re.compile(r'^<0x[0-9A-Fa-f]{2}>$')


def compute_token_quality_vector(tokenizer, alpha: float = 1.5) -> torch.Tensor:
    """
    Pre-compute a per-token quality score vector from the tokenizer vocabulary.

    Scores are based on heuristics (applied before alpha scaling):
      - Special tokens (EOS, BOS, PAD)          : -inf
      - <unk> token                              : -10.0
      - Byte fallback <0xNN>                     : -2.0
      - Known artifact ([[, ]], <|...|>)          : -5.0
      - Space-prefixed complete word (▁word)      : +0.3
      - Multi-char content token                  : +0.2

    The vector is multiplied by alpha and added to logits before sampling.

    Args:
        tokenizer: HuggingFace fast tokenizer instance
        alpha: scaling factor for the quality bias

    Returns:
        (vocab_size,) float32 tensor of logit biases
    """
    vocab_size = tokenizer.get_vocab_size()
    scores = torch.zeros(vocab_size, dtype=torch.float32)

    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    for tid in range(vocab_size):
        token_str = id_to_token.get(tid, "")

        # Special tokens: EOS, PAD, BOS
        if token_str in ("</s>", "<pad>", "<s>"):
            scores[tid] = float("-inf")
            continue

        # Unknown token
        if token_str == "<unk>":
            scores[tid] = -10.0
            continue

        # Byte fallback tokens (some tokenizers use <0xNN>)
        if _BYTE_FALLBACK_RE.match(token_str):
            scores[tid] = -2.0
            continue

        # Known artifacts
        if _ARTIFACT_RE.match(token_str):
            scores[tid] = -5.0
            continue

        # Space-prefixed = word boundary token (SentencePiece ▁)
        if token_str.startswith("\u2581") or token_str.startswith(" "):
            scores[tid] += 0.3

        # Multi-character content tokens (more informative than single chars)
        clean = token_str.lstrip("\u2581 ")
        if len(clean) > 1:
            scores[tid] += 0.2

    # Apply alpha scaling (only to finite scores)
    finite_mask = scores.isfinite()
    scores[finite_mask] = scores[finite_mask] * alpha

    non_zero = (scores != 0).sum().item()
    neg_inf = scores.isinf().sum().item()
    logger.info(
        "Token quality vector: %d/%d tokens biased (%d suppressed)",
        non_zero, vocab_size, neg_inf,
    )

    return scores


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
        self._token_quality_vector: Optional[torch.Tensor] = None

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

    @property
    def token_quality_vector(self) -> torch.Tensor:
        """Lazily compute and cache the token quality vector."""
        if self._token_quality_vector is None:
            self._token_quality_vector = compute_token_quality_vector(self.tokenizer)
        return self._token_quality_vector

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
