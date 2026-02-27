"""
vllm-i64 :: Inference Engine

Main engine that orchestrates:
  1. Scheduler (i64) — decides what to process
  2. Router (i64) — assigns tokens to experts
  3. Model (fp16) — runs expert MLP + attention
  4. Sampler (i64 argmax) — picks next token

Integer-first: the engine loop is 100% integer control flow.
Float only exists inside model.forward().

INL - 2025
"""

import torch
import numpy as np
import time
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

from vllm_i64.engine.i64_scheduler import I64Scheduler, I64Batch, I64Request


@dataclass
class GenerationResult:
    """Result of a generation request — integer token IDs."""
    request_id: int
    prompt_tokens: List[int]
    output_tokens: List[int]
    num_steps: int
    elapsed_ms: float   # Only float for human-readable timing


class I64Engine:
    """
    Integer-first inference engine.

    Control flow:
        while has_work:
            batch = scheduler.schedule()       # i64: pick requests, pre-route
            logits = model.forward(batch)      # fp16: the ONLY float step
            tokens = sampler.sample(logits)    # i64: argmax
            scheduler.update(tokens)           # i64: update state

    The engine never touches float. It passes integer-indexed batches
    to the model and gets back integer token IDs.
    """

    def __init__(
        self,
        model: Optional[object] = None,
        num_experts: int = 4,
        hidden_dim: int = 768,
        vocab_size: int = 100_000,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        max_kv_blocks: int = 4096,
        device: str = "cuda",
    ):
        self.model = model
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.device = device

        # Integer-only scheduler
        self.scheduler = I64Scheduler(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_experts=num_experts,
            max_kv_blocks=max_kv_blocks,
        )

        # Expert mask for routing (i64)
        self.expert_mask = np.int64(num_experts - 1)

        # Step counter (integer)
        self.total_steps: int = 0
        self.total_tokens_generated: int = 0

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
    ) -> int:
        """Add request. Returns integer request_id."""
        ids = np.array(prompt_token_ids, dtype=np.int64)
        return self.scheduler.add_request(ids, max_new_tokens)

    def _i64_sample(self, logits: torch.Tensor) -> np.ndarray:
        """
        Sample next tokens from logits.

        Uses argmax (greedy) — returns integer token IDs.
        This is the boundary: logits come in as float,
        token IDs go out as int64.

        For more sophisticated sampling (top-k, top-p),
        the comparison operations are still fundamentally integer
        (sorting indices, comparing thresholds).
        """
        # Greedy: argmax returns integer indices
        token_ids = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
        return token_ids

    def _i64_top_k_sample(
        self,
        logits: torch.Tensor,
        k: int = 50,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Top-k sampling.

        Even here, the heavy lifting is integer:
        - topk returns integer indices
        - multinomial returns integer indices
        - gather uses integer indices

        The only float is the softmax on k values (tiny).
        """
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k: get integer indices of top k logits
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

        # Softmax on k values only (small float operation)
        probs = torch.softmax(top_k_logits, dim=-1)

        # Multinomial returns integer index
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Gather the actual token ID (integer indexing)
        token_ids = top_k_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)

        return token_ids.cpu().numpy().astype(np.int64)

    def step(self) -> Dict[int, int]:
        """
        Execute one engine step.

        Returns {request_id: generated_token_id} — all integers.
        """
        # 1. Schedule (i64)
        batch = self.scheduler.schedule()
        if batch is None:
            return {}

        # 2. Model forward (fp16) — the ONLY float step
        if self.model is not None:
            logits = self._model_forward(batch)
        else:
            # Dummy logits for testing without model
            logits = torch.randn(batch.num_requests, self.vocab_size)

        # 3. Sample (i64 argmax)
        new_token_ids = self._i64_sample(logits)

        # 4. Map back to requests (integer indexing)
        result = {}
        for i, req_id in enumerate(batch.request_ids):
            result[int(req_id)] = int(new_token_ids[i])

        # 5. Update scheduler (i64)
        self.scheduler.update_after_step(result)

        # Integer counters
        self.total_steps += 1
        self.total_tokens_generated += len(result)

        return result

    def _model_forward(self, batch: I64Batch) -> torch.Tensor:
        """
        Run model forward pass.

        The batch contains pre-computed integer routing (expert_ids).
        The model uses these integer assignments directly — no
        float-based routing computation needed.
        """
        # Convert integer arrays to tensors
        token_ids = torch.from_numpy(batch.token_ids).to(self.device)
        expert_ids = torch.from_numpy(batch.expert_ids).to(self.device)
        positions = torch.from_numpy(batch.positions).to(self.device)

        # Model forward with pre-computed integer routing
        logits = self.model(
            token_ids=token_ids,
            expert_ids=expert_ids,
            positions=positions,
        )

        return logits

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
    ) -> GenerationResult:
        """
        Synchronous generation for a single request.
        """
        request_id = self.add_request(prompt_token_ids, max_new_tokens)

        start = time.perf_counter()
        steps = 0

        while True:
            results = self.step()
            steps += 1

            # Check if our request is done (integer comparison)
            req = None
            for r in self.scheduler.finished:
                if r.request_id == request_id:
                    req = r
                    break

            if req is not None:
                elapsed = (time.perf_counter() - start) * 1000
                return GenerationResult(
                    request_id=request_id,
                    prompt_tokens=list(req.prompt_token_ids),
                    output_tokens=req.output_token_ids,
                    num_steps=steps,
                    elapsed_ms=elapsed,
                )

    def run_continuous(self, callback: Optional[Callable] = None):
        """
        Run continuous batching loop.

        The loop is 100% integer control flow:
          while running (bool → int):
              batch = schedule()      # i64
              tokens = step()         # i64 in, fp16 compute, i64 out
              update()                # i64
        """
        while self.scheduler.pending or self.scheduler.running:
            results = self.step()

            if callback and results:
                callback(results)

    def get_stats(self) -> Dict[str, int]:
        """Engine stats — all integers."""
        sched_stats = self.scheduler.get_stats()
        return {
            **sched_stats,
            "total_steps": self.total_steps,
            "total_tokens_generated": self.total_tokens_generated,
        }
