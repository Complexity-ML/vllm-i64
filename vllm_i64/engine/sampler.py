"""
vllm-i64 :: Sampler

Decoupled sampling logic — wraps the sampling module with engine-level
concerns (per-request params, logprobs, logit processors).

The engine delegates all sampling to this class instead of inlining
the logic in the step loop.

INL - 2025
"""

import numpy as np
import torch
from typing import Dict, List, Optional

from vllm_i64.core.sampling import (
    SamplingParams, sample_batch, sample_batch_with_logprobs, TokenLogprob,
)


class I64Sampler:
    """
    Integer-first sampler.

    Input:  float logits from model.forward()
    Output: integer token IDs (i64)

    Supports:
      - Global default params (greedy by default)
      - Per-request params override
      - Logprobs collection
      - Logit processors (per-request)
    """

    def __init__(self, default_params: Optional[SamplingParams] = None):
        self.default_params = default_params or SamplingParams(temperature=0.0)

    def sample(
        self,
        logits: torch.Tensor,
        params: Optional[SamplingParams] = None,
        past_tokens_list: Optional[List[List[int]]] = None,
    ) -> np.ndarray:
        """
        Sample next tokens from logits.

        Returns: (batch_size,) i64 numpy array of token IDs.
        """
        p = params or self.default_params
        token_ids = sample_batch(logits, p, past_tokens_list=past_tokens_list)
        return token_ids.cpu().numpy().astype(np.int64)

    def sample_with_logprobs(
        self,
        logits: torch.Tensor,
        params: Optional[SamplingParams] = None,
        past_tokens_list: Optional[List[List[int]]] = None,
    ):
        """
        Sample with log-probability tracking.

        Returns: SampleOutput with token_ids and logprobs.
        """
        p = params or self.default_params
        return sample_batch_with_logprobs(logits, p, past_tokens_list=past_tokens_list)
