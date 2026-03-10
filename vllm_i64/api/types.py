"""
vllm-i64 :: API Types

CompletionRequest and CompletionResponse dataclasses.
INL - 2025
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


def compute_partition(api_key: Optional[str], user_id: Optional[str], n: int = 64) -> int:
    """Compute deterministic partition index for cache affinity and load balancing.

    partition = sha256(f"{api_key}:{user_id}") % n
    """
    key = f"{api_key or ''}:{user_id or ''}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") % n

from vllm_i64.core.sampling import SamplingParams
from vllm_i64.core.logits_processor import OutputConstraints


@dataclass
class CompletionRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0
    typical_p: float = 1.0
    repetition_penalty: float = 1.1
    min_tokens: int = 0
    stream: bool = False
    response_format: Optional[Dict] = None
    stop: Optional[list] = None
    n: int = 1
    best_of: int = 1
    logprobs: Optional[int] = None
    seed: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    priority: int = 0
    suppress_first_tokens: Optional[List[int]] = None
    user: Optional[str] = None  # OpenAI-compat user ID for partition affinity

    def validate(self, max_seq_len: int = 2048) -> Optional[str]:
        if not self.prompt or not self.prompt.strip():
            return "prompt must not be empty"
        if self.max_tokens < 1:
            return "max_tokens must be >= 1"
        if self.max_tokens > max_seq_len:
            return f"max_tokens must be <= {max_seq_len}"
        if self.temperature < 0:
            return "temperature must be >= 0"
        if self.top_k < 0:
            return "top_k must be >= 0"
        if self.top_p < 0 or self.top_p > 1:
            return "top_p must be in [0, 1]"
        if self.min_p < 0 or self.min_p > 1:
            return "min_p must be in [0, 1]"
        if self.typical_p < 0 or self.typical_p > 1:
            return "typical_p must be in [0, 1]"
        if self.min_tokens < 0:
            return "min_tokens must be >= 0"
        if self.repetition_penalty <= 0:
            return "repetition_penalty must be > 0"
        if self.logprobs is not None and (self.logprobs < 0 or self.logprobs > 20):
            return "logprobs must be between 0 and 20"
        if self.frequency_penalty < -2.0 or self.frequency_penalty > 2.0:
            return "frequency_penalty must be in [-2.0, 2.0]"
        if self.presence_penalty < -2.0 or self.presence_penalty > 2.0:
            return "presence_penalty must be in [-2.0, 2.0]"
        if self.logit_bias:
            for k, v in self.logit_bias.items():
                if not k.lstrip('-').isdigit():
                    return f"logit_bias keys must be token ID strings, got '{k}'"
                if v < -100 or v > 100:
                    return f"logit_bias values must be in [-100, 100], got {v}"
        return None

    def to_sampling_params(self, tokenizer=None) -> SamplingParams:
        constraints = None
        if self.response_format or self.stop or self.suppress_first_tokens:
            stop_seqs = None
            if self.stop and tokenizer is not None:
                stop_seqs = [tokenizer.encode(s) for s in self.stop]
            elif self.stop:
                stop_seqs = [[int(b) for b in s.encode("utf-8")] for s in self.stop]
            constraints = OutputConstraints(
                json_mode=bool(self.response_format and self.response_format.get("type") == "json_object"),
                regex_pattern=(
                    self.response_format.get("pattern")
                    if self.response_format and self.response_format.get("type") == "regex"
                    else None
                ),
                stop_sequences=stop_seqs,
                suppress_first_tokens=self.suppress_first_tokens,
            )
        logit_bias = {int(k): v for k, v in self.logit_bias.items()} if self.logit_bias else None
        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            typical_p=self.typical_p,
            repetition_penalty=self.repetition_penalty,
            min_tokens=self.min_tokens,
            json_mode=bool(self.response_format and self.response_format.get("type") == "json_object"),
            num_beams=self.best_of if self.best_of > 1 else 1,
            logprobs=self.logprobs,
            output_constraints=constraints,
            seed=self.seed,
            logit_bias=logit_bias,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )


@dataclass
class CompletionResponse:
    id: str
    object: str = "text_completion"
    created: int = 0
    model: str = "inl-token-routed"
    choices: List[Dict] = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = []

    def to_dict(self) -> dict:
        d = asdict(self)
        if hasattr(self, '_usage'):
            d["usage"] = self._usage
        if hasattr(self, '_engine_metrics'):
            d["engine_metrics"] = self._engine_metrics
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
