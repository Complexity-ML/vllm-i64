"""
vllm-i64 :: Inference Engine

Main engine that orchestrates:
  1. Scheduler (i64) — decides what to process
  2. Router (i64) — assigns tokens to experts
  3. Model (fp16) — runs expert MLP + attention
  4. Sampler (i64 argmax) — picks next token

Two modes:
  - I64Engine: synchronous, for single-request or testing
  - AsyncI64Engine: async continuous batching, for production

Integer-first: the engine loop is 100% integer control flow.
Float only exists inside model.forward().

INL - 2025
"""

import torch
import numpy as np
import time
import asyncio
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
        """
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

        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(top_k_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
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
        The model uses these integer assignments directly.
        """
        token_ids = torch.from_numpy(batch.token_ids).to(self.device)
        positions = torch.from_numpy(batch.positions).to(self.device)

        with torch.no_grad():
            logits = self.model(
                token_ids=token_ids,
                positions=positions,
            )

        return logits

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
    ) -> GenerationResult:
        """Synchronous generation for a single request."""
        request_id = self.add_request(prompt_token_ids, max_new_tokens)

        start = time.perf_counter()
        steps = 0

        while True:
            self.step()
            steps += 1

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
        """Run continuous batching loop (synchronous)."""
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


# =========================================================================
# Async Continuous Batching Engine
# =========================================================================

class AsyncI64Engine:
    """
    Async wrapper for parallel request handling and continuous batching.

    Multiple concurrent generate() calls feed into the scheduler.
    A background engine loop continuously:
      1. Admits new requests from queue → scheduler
      2. Runs one step (schedule → forward → sample → update)
      3. Resolves futures for finished requests

    This maximizes GPU utilization: multiple requests are batched
    together in each forward pass, achieving higher tok/s.

    Usage:
        engine = AsyncI64Engine(model, ...)
        await engine.start()

        # These run concurrently, batched together:
        r1 = asyncio.create_task(engine.generate([1, 2, 3]))
        r2 = asyncio.create_task(engine.generate([4, 5, 6]))
        result1 = await r1
        result2 = await r2

        await engine.stop()
    """

    def __init__(
        self,
        model: Optional[object] = None,
        num_experts: int = 4,
        vocab_size: int = 100_000,
        max_batch_size: int = 64,
        max_seq_len: int = 2048,
        max_kv_blocks: int = 8192,
        device: str = "cuda",
    ):
        self.engine = I64Engine(
            model=model,
            num_experts=num_experts,
            vocab_size=vocab_size,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_kv_blocks=max_kv_blocks,
            device=device,
        )

        # Async state
        self._pending_futures: Dict[int, asyncio.Future] = {}
        self._request_times: Dict[int, float] = {}
        self._engine_task: Optional[asyncio.Task] = None
        self._running = False

        # Stats
        self.active_requests: int = 0
        self.peak_batch_size: int = 0

    async def start(self):
        """Start the continuous batching background loop."""
        self._running = True
        self._engine_task = asyncio.create_task(self._engine_loop())

    async def stop(self):
        """Stop the engine loop."""
        self._running = False
        if self._engine_task:
            self._engine_task.cancel()
            try:
                await self._engine_task
            except asyncio.CancelledError:
                pass

    async def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
    ) -> GenerationResult:
        """
        Submit a request and wait for completion.

        Multiple concurrent calls are automatically batched together
        by the engine loop for maximum throughput.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        request_id = self.engine.add_request(prompt_token_ids, max_new_tokens)
        self._pending_futures[request_id] = future
        self._request_times[request_id] = time.perf_counter()
        self.active_requests += 1

        return await future

    async def generate_stream(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
    ):
        """
        Submit a request and yield tokens as they are generated.

        Yields (token_id: int, is_finished: bool) tuples.
        """
        loop = asyncio.get_running_loop()

        request_id = self.engine.add_request(prompt_token_ids, max_new_tokens)
        self._request_times[request_id] = time.perf_counter()
        self.active_requests += 1

        # Use a queue for streaming tokens
        token_queue: asyncio.Queue = asyncio.Queue()
        self._pending_futures[request_id] = token_queue

        try:
            while True:
                item = await token_queue.get()
                if item is None:
                    # Generation complete
                    break
                yield item
        finally:
            self.active_requests -= 1
            self._request_times.pop(request_id, None)

    async def _engine_loop(self):
        """
        Continuous batching loop.

        Runs forever, processing whatever requests are in the scheduler.
        Idle when no requests; immediately picks up new ones.
        """
        while self._running:
            has_work = (
                self.engine.scheduler.pending or self.engine.scheduler.running
            )

            if has_work:
                # Track batch utilization
                batch_size = len(self.engine.scheduler.running)
                self.peak_batch_size = max(self.peak_batch_size, batch_size)

                # Run one step: schedule → forward → sample → update
                step_results = self.engine.step()

                # Deliver tokens to streaming futures
                for req_id, token_id in step_results.items():
                    if req_id in self._pending_futures:
                        target = self._pending_futures[req_id]
                        if isinstance(target, asyncio.Queue):
                            await target.put(token_id)

                # Check for finished requests → resolve futures
                finished_ids = set()
                for req in self.engine.scheduler.finished:
                    rid = req.request_id
                    if rid in self._pending_futures:
                        target = self._pending_futures.pop(rid)
                        elapsed = (time.perf_counter() - self._request_times.pop(rid, 0)) * 1000

                        if isinstance(target, asyncio.Future) and not target.done():
                            result = GenerationResult(
                                request_id=rid,
                                prompt_tokens=list(req.prompt_token_ids),
                                output_tokens=req.output_token_ids,
                                num_steps=req.num_generated,
                                elapsed_ms=elapsed,
                            )
                            target.set_result(result)
                        elif isinstance(target, asyncio.Queue):
                            await target.put(None)  # Signal end of stream

                        self.active_requests -= 1
                        finished_ids.add(rid)

                # Clean up finished from scheduler
                self.engine.scheduler.finished = [
                    r for r in self.engine.scheduler.finished
                    if r.request_id not in finished_ids
                ]

                # Yield to event loop after each step
                await asyncio.sleep(0)
            else:
                # No work — short sleep to avoid busy-waiting
                await asyncio.sleep(0.001)

    def get_stats(self) -> Dict[str, int]:
        """Engine + async stats."""
        stats = self.engine.get_stats()
        stats["active_requests"] = self.active_requests
        stats["peak_batch_size"] = self.peak_batch_size
        return stats
