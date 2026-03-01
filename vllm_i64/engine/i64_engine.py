"""
vllm-i64 :: Inference Engine

Main engine that orchestrates:
  1. Scheduler (i64) — decides what to process
  2. Router (i64) — assigns tokens to experts
  3. Model (fp16) — runs expert MLP + attention
  4. Sampler (i64 argmax or configurable) — picks next token

Two modes:
  - I64Engine: synchronous, for single-request or testing
  - AsyncI64Engine: async continuous batching, for production

Integrations:
  - PagedKVCache: per-request KV caching in attention
  - SamplingParams: configurable sampling (greedy/top-k/top-p/temperature)
  - CUDAGraphRunner: captured decode step for reduced kernel launch overhead
  - SpeculativeDecoder: draft+verify for faster generation
  - I64Metrics: Prometheus monitoring

Integer-first: the engine loop is 100% integer control flow.
Float only exists inside model.forward().

INL - 2025
"""

import torch
import numpy as np
import time
import asyncio
import signal
from typing import List, Dict, Optional, Callable, Set
from dataclasses import dataclass, field

from vllm_i64.engine.i64_scheduler import I64Scheduler, I64Batch, I64Request
from vllm_i64.core.sampling import (
    SamplingParams, sample_batch, sample_batch_with_logprobs,
    apply_repetition_penalty_batch, TokenLogprob,
)
from vllm_i64.core.logging import get_logger

logger = get_logger("vllm_i64.engine")


class AdaptiveBatchSizer:
    """Dynamically adjust max_batch_size based on throughput."""

    def __init__(self, initial: int, min_size: int = 1, max_size: int = 128, window: int = 20):
        self.current = initial
        self.min_size = min_size
        self.max_size = max_size
        self._throughputs: List[float] = []
        self.window = window

    def record(self, tokens: int, elapsed_ms: float):
        """Record a step's throughput."""
        tps = tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        self._throughputs.append(tps)
        if len(self._throughputs) > self.window:
            self._throughputs.pop(0)

    def adjust(self) -> int:
        """Adjust batch size based on throughput trend."""
        if len(self._throughputs) < self.window:
            return self.current
        avg_tps = sum(self._throughputs) / len(self._throughputs)
        recent_tps = sum(self._throughputs[-5:]) / 5

        if recent_tps > avg_tps * 1.05:
            self.current = min(self.current + 1, self.max_size)
        elif recent_tps < avg_tps * 0.9:
            self.current = max(self.current - 1, self.min_size)
        return self.current


@dataclass
class GenerationResult:
    """Result of a generation request — integer token IDs."""
    request_id: int
    prompt_tokens: List[int]
    output_tokens: List[int]
    num_steps: int
    elapsed_ms: float   # Only float for human-readable timing
    finish_reason: str = "length"  # "length", "stop", "timeout", "cancelled"
    token_logprobs: Optional[List[TokenLogprob]] = None


class I64Engine:
    """
    Integer-first inference engine with KV cache and configurable sampling.

    Control flow:
        while has_work:
            batch = scheduler.schedule()       # i64: pick requests, pre-route
            logits = model.forward(batch)      # fp16: the ONLY float step
            tokens = sampler.sample(logits)    # i64: argmax or configured
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
        enable_prefix_caching: bool = False,
        kv_cache_dtype: Optional[str] = None,
    ):
        self.model = model
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.device = device
        self.enable_prefix_caching = enable_prefix_caching
        self.kv_cache_dtype = kv_cache_dtype

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

        # Sampling parameters (configurable, default greedy)
        self.sampling_params = SamplingParams(temperature=0.0)
        # Per-request sampling params for multi-user isolation
        self._request_sampling_params: Dict[int, SamplingParams] = {}

        # === KV Cache ===
        self.kv_cache = None
        self._slot_pool: List[int] = []
        self._request_to_slot: Dict[int, int] = {}

        if self.model is not None and hasattr(self.model, 'config'):
            self._init_kv_cache(max_batch_size)

        # === CUDA Graph Runner ===
        self.cuda_graph_runner = None
        if device != "cpu" and self.model is not None:
            self._init_cuda_graph(max_batch_size)

        # === Speculative Decoding ===
        self.speculative_decoder = None

        # === Request management ===
        self._cancelled_requests: Set[int] = set()
        self._request_deadlines: Dict[int, float] = {}  # request_id → deadline timestamp
        self.default_timeout_s: float = 300.0  # 5 min default

        # === Logprobs tracking ===
        self._request_logprobs: Dict[int, List[TokenLogprob]] = {}

        # === Logits processors (per-request) ===
        self._request_processors: Dict[int, list] = {}

        # === Metrics ===
        self.metrics = None

    def _init_kv_cache(self, max_seqs: int):
        """Initialize paged KV cache from model config."""
        from vllm_i64.core.kv_cache import PagedKVCache
        from vllm_i64.parallel.tensor_parallel import get_tp

        config = self.model.config
        tp = get_tp()
        num_kv_heads = config.num_key_value_heads // tp.tp_size
        dtype = next(self.model.parameters()).dtype

        self.kv_cache = PagedKVCache(
            num_layers=config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
            head_dim=config.head_dim,
            block_size=16,
            num_blocks=max(256, max_seqs * 8),
            max_seqs=max_seqs,
            dtype=dtype,
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        self._slot_pool = list(range(max_seqs))

        if self.enable_prefix_caching:
            self.kv_cache.enable_prefix_caching()
            logger.info("Prefix caching enabled")

    def _init_cuda_graph(self, max_batch_size: int):
        """Initialize CUDA graph runner for decode steps."""
        try:
            from vllm_i64.core.cuda_graph import CUDAGraphRunner

            def _graph_forward(token_ids, positions, expert_ids):
                return self.model(token_ids=token_ids, positions=positions)

            self.cuda_graph_runner = CUDAGraphRunner(
                forward_fn=_graph_forward,
                max_batch_size=max_batch_size,
                device=self.device,
            )
        except Exception:
            self.cuda_graph_runner = None

    def warmup_and_capture_graphs(self):
        """Warmup model and capture CUDA graphs for common decode batch sizes."""
        if self.cuda_graph_runner is None or self.model is None:
            return
        if self.device == "cpu":
            return
        try:
            self.cuda_graph_runner.capture_common_sizes()
            logger.info(f"CUDA graphs captured for sizes: {sorted(self.cuda_graph_runner._captured_sizes)}")
        except Exception as e:
            logger.warning(f"CUDA graph capture failed: {e}")
            self.cuda_graph_runner = None

    def enable_metrics(self, port: int = 9090, model_name: str = ""):
        """Enable Prometheus metrics collection."""
        from vllm_i64.core.metrics import I64Metrics
        self.metrics = I64Metrics(port=port, model_name=model_name)

    def enable_speculative(self, draft_model, num_speculative: int = 5):
        """Enable speculative decoding with a draft model."""
        from vllm_i64.core.speculative import SpeculativeDecoder
        self.speculative_decoder = SpeculativeDecoder(
            target_model=self.model,
            draft_model=draft_model,
            num_speculative=num_speculative,
        )

    def _allocate_slot(self, request_id: int) -> int:
        """Allocate a KV cache slot for a request."""
        if not self._slot_pool:
            return -1
        slot = self._slot_pool.pop(0)
        self._request_to_slot[request_id] = slot
        return slot

    def _free_slot(self, request_id: int):
        """Free a KV cache slot when a request finishes."""
        if request_id in self._request_to_slot:
            slot = self._request_to_slot.pop(request_id)
            if self.kv_cache is not None:
                self.kv_cache.free_sequence(slot)
            self._slot_pool.append(slot)

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        timeout_s: Optional[float] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> int:
        """Add request. Returns integer request_id."""
        ids = np.array(prompt_token_ids, dtype=np.int64)

        # Get EOS token ID from model config
        eos_token_id = 0
        if self.model is not None and hasattr(self.model, 'config'):
            eos_token_id = getattr(self.model.config, 'eos_token_id', 0)

        request_id = self.scheduler.add_request(ids, max_new_tokens, eos_token_id=eos_token_id)

        # Store per-request sampling params
        if sampling_params is not None:
            self._request_sampling_params[request_id] = sampling_params
            # Build logits processors if output constraints are set
            if sampling_params.output_constraints is not None:
                processors = sampling_params.output_constraints.build_processors()
                if processors:
                    self._request_processors[request_id] = processors

        # Set deadline
        t = timeout_s if timeout_s is not None else self.default_timeout_s
        if t > 0:
            self._request_deadlines[request_id] = time.perf_counter() + t

        # Allocate KV cache slot
        if self.kv_cache is not None:
            self._allocate_slot(request_id)

            # Try to reuse prefix from cache
            if self.kv_cache.prefix_cache_enabled:
                slot = self._request_to_slot.get(request_id)
                if slot is not None:
                    reused = self.kv_cache.try_reuse_prefix(slot, list(ids))
                    if reused > 0:
                        for req in self.scheduler.pending:
                            if req.request_id == request_id:
                                req.prefill_progress = reused
                                req.seq_pos = reused
                                break
                        logger.info(f"Prefix cache hit: reused {reused} tokens for request {request_id}")

        return request_id

    def cancel_request(self, request_id: int):
        """Cancel a running or pending request."""
        self._cancelled_requests.add(request_id)

    def _i64_sample(self, logits: torch.Tensor, past_tokens_list=None) -> np.ndarray:
        """
        Sample next tokens from logits using configured sampling params.
        """
        token_ids = sample_batch(logits, self.sampling_params, past_tokens_list=past_tokens_list)
        return token_ids.cpu().numpy().astype(np.int64)

    def _check_timeouts_and_cancellations(self):
        """Remove timed-out and cancelled requests from scheduler."""
        now = time.perf_counter()
        to_remove = set()

        # Check cancellations
        for req in self.scheduler.running + self.scheduler.pending:
            if req.request_id in self._cancelled_requests:
                to_remove.add(req.request_id)

        # Check timeouts
        for req in self.scheduler.running:
            deadline = self._request_deadlines.get(req.request_id)
            if deadline and now > deadline:
                to_remove.add(req.request_id)
                logger.warning(f"Request {req.request_id} timed out")

        # Move to finished with appropriate reason
        for rid in to_remove:
            for req in self.scheduler.running[:]:
                if req.request_id == rid:
                    req.status = 3  # FINISHED
                    reason = "cancelled" if rid in self._cancelled_requests else "timeout"
                    req._finish_reason = reason
                    self.scheduler.running.remove(req)
                    self._free_kv_blocks_for(req)
                    self.scheduler.finished.append(req)
                    self._free_slot(req.request_id)
                    break
            for req in self.scheduler.pending[:]:
                if req.request_id == rid:
                    self.scheduler.pending.remove(req)
                    req.status = 3
                    req._finish_reason = "cancelled"
                    self.scheduler.finished.append(req)
                    break

        self._cancelled_requests -= to_remove
        for rid in to_remove:
            self._request_deadlines.pop(rid, None)

    def _free_kv_blocks_for(self, req):
        """Free KV blocks owned by a request."""
        if req.kv_block_ids:
            self.scheduler._free_kv_blocks(req.kv_block_ids)
            req.kv_block_ids = []

    def _speculative_step(self, batch: I64Batch) -> Dict[int, int]:
        """
        Speculative decode step for small decode-only batches.

        Uses draft model to generate K tokens, verifies with target model.
        Multi-token acceptance → fewer forward passes → higher throughput.
        """
        result = {}
        for i, req_id in enumerate(batch.request_ids):
            rid = int(req_id)
            req = next((r for r in self.scheduler.running if r.request_id == rid), None)
            if req is None:
                continue

            # Build context for speculative decoder
            ctx = torch.tensor(
                list(req.prompt_token_ids) + req.output_token_ids,
                dtype=torch.int64, device=self.device,
            )
            pos = torch.arange(len(ctx), dtype=torch.int32, device=self.device)

            accepted, _ = self.speculative_decoder.generate_step(ctx, pos)

            # Feed all accepted tokens except the last into the request directly
            for token in accepted[:-1]:
                req.output_token_ids.append(token)
                req.seq_pos = req.total_tokens

            # Return the last accepted token for normal update_after_step
            result[rid] = accepted[-1] if accepted else 0

        return result

    def step(self) -> Dict[int, int]:
        """
        Execute one engine step.

        Returns {request_id: generated_token_id} — all integers.
        """
        # 0. Handle timeouts and cancellations
        self._check_timeouts_and_cancellations()

        # 1. Schedule (i64)
        batch = self.scheduler.schedule()

        # Free KV slots for newly finished requests (must run even when batch is None)
        for req in self.scheduler.finished:
            self._free_slot(req.request_id)
            self._request_sampling_params.pop(req.request_id, None)
            self._request_processors.pop(req.request_id, None)
            # logprobs are consumed when building GenerationResult, not freed here

        if batch is None:
            return {}

        # Update metrics
        if self.metrics:
            self.metrics.update_batch_size(batch.num_requests)
            self.metrics.update_pending(len(self.scheduler.pending))
            if self.kv_cache:
                kv_stats = self.kv_cache.get_stats()
                self.metrics.update_kv_usage(kv_stats["used_blocks"], kv_stats["num_blocks"])

        # 1.5. Speculative decoding for small decode-only batches
        if (self.speculative_decoder is not None
                and batch.is_prefill.sum() == 0
                and batch.num_requests <= 4):
            result = self._speculative_step(batch)
            self.scheduler.update_after_step(result)
            for req in self.scheduler.finished:
                self._free_slot(req.request_id)
            self.total_steps += 1
            self.total_tokens_generated += len(result)
            return result

        # 2. Model forward (fp16) — the ONLY float step
        if self.model is not None:
            logits = self._model_forward(batch)
        else:
            logits = torch.randn(batch.num_requests, self.vocab_size)

        # 3. Extract last-token logits per request (for prefill: many tokens → 1 logit)
        if batch.tokens_per_request is not None and logits.shape[0] != batch.num_requests:
            last_indices = []
            offset = 0
            for tpr in batch.tokens_per_request:
                last_indices.append(offset + int(tpr) - 1)
                offset += int(tpr)
            logits = logits[last_indices]  # (num_requests, vocab_size)

        # 3.5. Per-request sampling with isolated params
        # Check if any request has custom sampling params
        has_custom = any(
            int(rid) in self._request_sampling_params
            for rid in batch.request_ids
        )

        if has_custom:
            # Sample each request individually with its own params
            result = {}
            for i, req_id in enumerate(batch.request_ids):
                rid = int(req_id)
                params = self._request_sampling_params.get(rid, self.sampling_params)
                req_logits = logits[i:i+1]

                past_tokens = None
                if params.repetition_penalty != 1.0:
                    req = next((r for r in self.scheduler.running if r.request_id == rid), None)
                    if req is not None:
                        past_tokens = [list(req.prompt_token_ids) + req.output_token_ids]
                    else:
                        past_tokens = [[]]

                # Apply logits processors if configured
                if rid in self._request_processors:
                    from vllm_i64.core.logits_processor import apply_logits_processors
                    req = next((r for r in self.scheduler.running if r.request_id == rid), None)
                    generated = req.output_token_ids if req else []
                    req_logits = apply_logits_processors(
                        req_logits.squeeze(0), self._request_processors[rid], generated
                    ).unsqueeze(0)
                    # Check stop sequence
                    for proc in self._request_processors[rid]:
                        if hasattr(proc, 'should_stop') and proc.should_stop:
                            if req is not None:
                                req.status = 3  # FINISHED
                                req._finish_reason = "stop"
                            break

                if params.logprobs is not None:
                    sample_out = sample_batch_with_logprobs(req_logits, params, past_tokens_list=past_tokens)
                    token_id = int(sample_out.token_ids.cpu().numpy().astype(np.int64)[0])
                    if sample_out.logprobs:
                        self._request_logprobs.setdefault(rid, []).append(sample_out.logprobs[0])
                    result[rid] = token_id
                else:
                    token_ids = sample_batch(req_logits, params, past_tokens_list=past_tokens)
                    result[rid] = int(token_ids.cpu().numpy().astype(np.int64)[0])
        else:
            # Fast path: all requests share the same params
            past_tokens_list = None
            if self.sampling_params.repetition_penalty != 1.0:
                past_tokens_list = []
                for req_id in batch.request_ids:
                    req = next((r for r in self.scheduler.running if r.request_id == int(req_id)), None)
                    if req is not None:
                        past_tokens_list.append(list(req.prompt_token_ids) + req.output_token_ids)
                    else:
                        past_tokens_list.append([])

            # 4. Sample (configurable, includes repetition penalty)
            new_token_ids = self._i64_sample(logits, past_tokens_list)

            # 5. Map back to requests (integer indexing)
            result = {}
            for i, req_id in enumerate(batch.request_ids):
                result[int(req_id)] = int(new_token_ids[i])

        # 6. Update scheduler (i64)
        self.scheduler.update_after_step(result)

        # 6.5 Register prefix blocks for requests that just completed prefill
        if self.kv_cache is not None and self.kv_cache.prefix_cache_enabled:
            for req in self.scheduler.running:
                if req.request_id in result and req.prefill_complete and req.num_generated == 1:
                    slot = self._request_to_slot.get(req.request_id)
                    if slot is not None:
                        self.kv_cache.register_prefix_blocks(slot, list(req.prompt_token_ids))

        # Integer counters
        self.total_steps += 1
        self.total_tokens_generated += len(result)

        return result

    def _model_forward(self, batch: I64Batch) -> torch.Tensor:
        """
        Run model forward pass with KV cache support.
        """
        token_ids = torch.from_numpy(batch.token_ids).to(self.device)
        positions = torch.from_numpy(batch.positions).to(self.device)

        # Build KV cache metadata
        seq_ids = None
        tokens_per_seq = None

        if self.kv_cache is not None and batch.tokens_per_request is not None:
            seq_ids = [
                self._request_to_slot.get(int(rid), 0)
                for rid in batch.request_ids
            ]
            tokens_per_seq = batch.tokens_per_request.tolist()

        # Use CUDA graph for pure decode batches
        use_graph = (
            self.cuda_graph_runner is not None
            and self.cuda_graph_runner.is_captured
            and batch.is_prefill.sum() == 0
        )

        with torch.no_grad():
            if use_graph:
                expert_ids = torch.from_numpy(batch.expert_ids).to(self.device)
                logits = self.cuda_graph_runner.run(token_ids, positions, expert_ids)
            else:
                logits = self.model(
                    token_ids=token_ids,
                    positions=positions,
                    kv_cache=self.kv_cache,
                    seq_ids=seq_ids,
                    tokens_per_seq=tokens_per_seq,
                )

        return logits

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        sampling_params: Optional[SamplingParams] = None,
    ) -> GenerationResult:
        """Synchronous generation for a single request."""
        request_id = self.add_request(
            prompt_token_ids, max_new_tokens,
            sampling_params=sampling_params,
        )

        metrics_start = None
        if self.metrics:
            metrics_start = self.metrics.on_request_start()

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
                # Determine finish reason
                finish_reason = getattr(req, "_finish_reason", None)
                if finish_reason is None:
                    if req.output_token_ids and req.output_token_ids[-1] == req.eos_token_id:
                        finish_reason = "stop"
                    else:
                        finish_reason = "length"

                if self.metrics and metrics_start is not None:
                    self.metrics.on_request_end(
                        metrics_start, len(prompt_token_ids), len(req.output_token_ids),
                    )

                # Collect logprobs if tracked
                token_logprobs = self._request_logprobs.pop(request_id, None)

                return GenerationResult(
                    request_id=request_id,
                    prompt_tokens=list(req.prompt_token_ids),
                    output_tokens=req.output_token_ids,
                    num_steps=steps,
                    elapsed_ms=elapsed,
                    finish_reason=finish_reason,
                    token_logprobs=token_logprobs if token_logprobs else None,
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
        stats = {
            **sched_stats,
            "total_steps": self.total_steps,
            "total_tokens_generated": self.total_tokens_generated,
        }
        if self.kv_cache:
            stats.update(self.kv_cache.get_stats())
        return stats


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
        self._draining = False

        # Stats
        self.active_requests: int = 0
        self.peak_batch_size: int = 0

    async def start(self):
        """Start the continuous batching background loop."""
        self._running = True
        self._engine_task = asyncio.create_task(self._engine_loop())

    async def stop(self, drain_timeout: float = 30.0):
        """
        Graceful shutdown: drain active requests before stopping.

        Args:
            drain_timeout: max seconds to wait for in-flight requests
        """
        logger.info(f"Engine shutdown requested, draining {self.active_requests} requests...")
        self._draining = True

        # Wait for in-flight requests to finish
        deadline = time.perf_counter() + drain_timeout
        while self.active_requests > 0 and time.perf_counter() < deadline:
            await asyncio.sleep(0.05)

        if self.active_requests > 0:
            logger.warning(f"Drain timeout: {self.active_requests} requests still active, cancelling")
            # Cancel remaining futures
            for rid, target in list(self._pending_futures.items()):
                if isinstance(target, asyncio.Future) and not target.done():
                    target.cancel()
                elif isinstance(target, asyncio.Queue):
                    await target.put(None)

        self._running = False
        if self._engine_task:
            self._engine_task.cancel()
            try:
                await self._engine_task
            except asyncio.CancelledError:
                pass
        logger.info("Engine stopped")

    async def cancel_request(self, request_id: int):
        """Cancel a running request."""
        self.engine.cancel_request(request_id)

    async def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        sampling_params: Optional[SamplingParams] = None,
        timeout_s: Optional[float] = None,
    ) -> GenerationResult:
        """
        Submit a request and wait for completion.

        Multiple concurrent calls are automatically batched together
        by the engine loop for maximum throughput.
        """
        if self._draining:
            raise RuntimeError("Engine is shutting down, not accepting new requests")

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        request_id = self.engine.add_request(
            prompt_token_ids, max_new_tokens, timeout_s=timeout_s,
            sampling_params=sampling_params,
        )
        self._pending_futures[request_id] = future
        self._request_times[request_id] = time.perf_counter()
        self.active_requests += 1

        return await future

    async def generate_stream(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        sampling_params: Optional[SamplingParams] = None,
    ):
        """
        Submit a request and yield tokens as they are generated.
        """
        request_id = self.engine.add_request(
            prompt_token_ids, max_new_tokens,
            sampling_params=sampling_params,
        )
        self._request_times[request_id] = time.perf_counter()
        self.active_requests += 1

        # Use a queue for streaming tokens
        token_queue: asyncio.Queue = asyncio.Queue()
        self._pending_futures[request_id] = token_queue

        try:
            while True:
                item = await token_queue.get()
                if item is None:
                    break
                yield item
        finally:
            self.active_requests -= 1
            self._request_times.pop(request_id, None)

    async def _engine_loop(self):
        """Continuous batching loop."""
        while self._running:
            has_work = (
                self.engine.scheduler.pending or self.engine.scheduler.running
            )

            if has_work:
                batch_size = len(self.engine.scheduler.running)
                self.peak_batch_size = max(self.peak_batch_size, batch_size)

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
                            finish_reason = getattr(req, "_finish_reason", None)
                            if finish_reason is None:
                                if req.output_token_ids and req.output_token_ids[-1] == req.eos_token_id:
                                    finish_reason = "stop"
                                else:
                                    finish_reason = "length"
                            token_logprobs = self.engine._request_logprobs.pop(rid, None)
                            result = GenerationResult(
                                request_id=rid,
                                prompt_tokens=list(req.prompt_token_ids),
                                output_tokens=req.output_token_ids,
                                num_steps=req.num_generated,
                                elapsed_ms=elapsed,
                                finish_reason=finish_reason,
                                token_logprobs=token_logprobs if token_logprobs else None,
                            )
                            target.set_result(result)
                            self.active_requests -= 1
                        elif isinstance(target, asyncio.Queue):
                            # Signal end to generate_stream; its finally block handles active_requests
                            await target.put(None)
                        finished_ids.add(rid)

                self.engine.scheduler.finished = [
                    r for r in self.engine.scheduler.finished
                    if r.request_id not in finished_ids
                ]

                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0.001)

    def get_stats(self) -> Dict[str, int]:
        """Engine + async stats."""
        stats = self.engine.get_stats()
        stats["active_requests"] = self.active_requests
        stats["peak_batch_size"] = self.peak_batch_size
        return stats
