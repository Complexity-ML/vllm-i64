"""
vllm-i64 :: CPU Engine

Dedicated CPU inference engine — isolated from GPU code.

Full-featured (same capabilities as the GPU engine):
  - Paged KV cache
  - Continuous batching scheduler
  - Per-request sampling params (temperature, top-k, top-p, repetition penalty...)
  - Logprobs, logits processors, output constraints
  - Request timeouts and cancellations
  - Request merging (dedup identical prompts)
  - Streaming (SSE) and non-streaming modes
  - Event-driven async loop (no polling)

No GPU, no CUDA, no Triton, no pybind11.
All step() calls run in a thread executor so the asyncio event loop is
never blocked by the CPU-heavy forward pass.

Changes to i64_engine.py (GPU engine) cannot break this file.

INL - 2025
"""

import asyncio
import time
import torch
from typing import Optional

from vllm_i64.engine.i64_engine import (
    I64Engine,
    AsyncI64Engine,
    GenerationResult,
)
from vllm_i64.engine.i64_scheduler import I64Batch
from vllm_i64.core.logging import get_logger

logger = get_logger("vllm_i64.cpu")


# =============================================================================
# CPUEngine — synchronous, no CUDA graphs
# =============================================================================

class CPUEngine(I64Engine):
    """
    CPU-only synchronous inference engine.

    Inherits all features from I64Engine (scheduler, KV cache, sampling,
    timeouts, request merging, logprobs...) but overrides the GPU-specific
    parts with clean CPU alternatives:

      - _init_cuda_graph     → disabled (no-op)
      - warmup_and_capture_graphs → disabled (no-op)
      - _model_forward       → direct model call, no CUDA graph dispatch
    """

    def __init__(
        self,
        model=None,
        num_experts: int = 4,
        vocab_size: int = 100_000,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        max_kv_blocks: int = 0,
        enable_prefix_caching: bool = False,
        kv_cache_dtype: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            num_experts=num_experts,
            vocab_size=vocab_size,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            max_kv_blocks=max_kv_blocks,
            device="cpu",
            enable_prefix_caching=enable_prefix_caching,
            kv_cache_dtype=kv_cache_dtype,
        )

    # -------------------------------------------------------------------------
    # GPU overrides
    # -------------------------------------------------------------------------

    def _init_cuda_graph(self, max_batch_size: int) -> None:
        """No CUDA graphs on CPU."""
        self.cuda_graph_runner = None

    def warmup_and_capture_graphs(self) -> None:
        """No-op on CPU."""
        pass

    def _model_forward(self, batch: I64Batch) -> torch.Tensor:
        """
        CPU model forward — no CUDA graph dispatch.

        Converts numpy batch arrays directly to CPU tensors and calls
        model.forward(). No static buffers, no graph runner, no GPU sync.
        """
        token_ids = torch.from_numpy(batch.token_ids)
        positions = torch.from_numpy(batch.positions)

        seq_ids = None
        tokens_per_seq = None

        if self.kv_cache is not None and batch.tokens_per_request is not None:
            seq_ids = []
            valid_indices = []
            for i, rid in enumerate(batch.request_ids):
                slot = self._request_to_slot.get(int(rid))
                if slot is None:
                    logger.error(
                        "Request %d has no KV slot — skipping from batch", int(rid)
                    )
                    continue
                seq_ids.append(slot)
                valid_indices.append(i)
            tokens_per_seq = [batch.tokens_per_request[i] for i in valid_indices]

        with torch.no_grad():
            logits = self.model(
                token_ids=token_ids,
                positions=positions,
                kv_cache=self.kv_cache,
                seq_ids=seq_ids,
                tokens_per_seq=tokens_per_seq,
            )

        return logits


# =============================================================================
# AsyncCPUEngine — event-driven async wrapper, always uses thread executor
# =============================================================================

class AsyncCPUEngine(AsyncI64Engine):
    """
    Async continuous-batching wrapper for CPUEngine.

    Key difference vs AsyncI64Engine:
      - _engine_loop() ALWAYS runs step() in a thread executor.
      - No `if self.device == "cpu":` check — this class is CPU-only by design.
      - Adding GPU features to AsyncI64Engine cannot accidentally break this loop.

    All other features (streaming, request queuing, backpressure, drain
    on shutdown, merged secondaries...) are inherited from AsyncI64Engine.
    """

    @classmethod
    def from_cpu_engine(cls, engine: CPUEngine) -> "AsyncCPUEngine":
        """Wrap an existing CPUEngine in an async continuous-batching loop."""
        instance = cls.__new__(cls)
        instance.engine = engine
        instance.device = "cpu"
        instance._pending_futures = {}
        instance._request_times = {}
        instance._engine_task = None
        instance._running = False
        instance._draining = False
        instance._new_request_event = None
        instance.active_requests = 0
        instance.peak_batch_size = 0
        instance.max_queue_depth = engine.scheduler.max_batch_size * 8
        return instance

    async def _engine_loop(self) -> None:
        """
        CPU engine loop — always uses thread executor for step().

        Identical logic to AsyncI64Engine._engine_loop but with the
        `if self.device == "cpu":` branch removed: we always run in a
        thread executor, so the asyncio event loop is never blocked.
        """
        _consecutive_errors = 0
        _MAX_CONSECUTIVE_ERRORS = 10

        while self._running:
            has_work = (
                self.engine.scheduler.pending or self.engine.scheduler.running
            )

            if has_work:
                batch_size = len(self.engine.scheduler.running)
                self.peak_batch_size = max(self.peak_batch_size, batch_size)
                try:
                    # Always run in thread executor on CPU — forward pass can
                    # take several seconds and must not block the event loop.
                    loop = asyncio.get_running_loop()
                    step_results = await loop.run_in_executor(
                        None, self.engine.step
                    )
                    _consecutive_errors = 0

                except Exception as e:
                    _consecutive_errors += 1
                    logger.error(
                        "Engine step failed (%d/%d): %s",
                        _consecutive_errors, _MAX_CONSECUTIVE_ERRORS, e,
                    )

                    # Fail all stuck requests so clients get an error response
                    for req in list(self.engine.scheduler.running):
                        rid = req.request_id
                        if rid in self._pending_futures:
                            target = self._pending_futures.pop(rid)
                            if isinstance(target, asyncio.Future) and not target.done():
                                target.set_exception(RuntimeError("Engine step failed — check server logs"))
                                self.active_requests -= 1
                            elif isinstance(target, asyncio.Queue):
                                await target.put(None)
                        self.engine._free_kv_blocks_for(req)
                        self.engine._free_slot(rid)
                        self.engine._request_deadlines.pop(rid, None)
                        self.engine._request_sampling_params.pop(rid, None)
                        self.engine._request_processors.pop(rid, None)
                        self.engine._request_logprobs.pop(rid, None)
                    self.engine.scheduler.running.clear()

                    if _consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        logger.error("Too many consecutive errors, CPU engine loop stopping")
                        break

                    await asyncio.sleep(0.1)
                    continue

                # Deliver tokens to streaming futures
                for req_id, token_id in step_results.items():
                    if req_id in self._pending_futures:
                        target = self._pending_futures[req_id]
                        if isinstance(target, asyncio.Queue):
                            await target.put(token_id)

                # Resolve finished requests → futures / queues
                finished_ids = set()
                for req in self.engine.scheduler.finished:
                    rid = req.request_id
                    if rid in self._pending_futures:
                        target = self._pending_futures.pop(rid)
                        elapsed = (
                            time.perf_counter() - self._request_times.pop(rid, 0)
                        ) * 1000

                        if isinstance(target, asyncio.Future) and not target.done():
                            finish_reason = getattr(req, "_finish_reason", None)
                            if finish_reason is None:
                                if (
                                    req.output_token_ids
                                    and req.output_token_ids[-1] == req.eos_token_id
                                ):
                                    finish_reason = "stop"
                                else:
                                    finish_reason = "length"

                            token_logprobs = self.engine._request_logprobs.pop(rid, None)
                            # Truncate stop sequence tokens from output
                            output_tokens = req.output_token_ids
                            if rid in self.engine._request_processors:
                                for proc in self.engine._request_processors[rid]:
                                    if hasattr(proc, 'should_stop') and proc.should_stop:
                                        idx = proc.stop_index
                                        if 0 <= idx < len(output_tokens):
                                            output_tokens = output_tokens[:idx]
                                            if token_logprobs:
                                                token_logprobs = token_logprobs[:idx]
                                        break
                            result = GenerationResult(
                                request_id=rid,
                                prompt_tokens=req.prompt_list,
                                output_tokens=output_tokens,
                                num_steps=req.num_generated,
                                elapsed_ms=elapsed,
                                finish_reason=finish_reason,
                                token_logprobs=token_logprobs or None,
                            )
                            target.set_result(result)
                            self.active_requests -= 1

                        elif isinstance(target, asyncio.Queue):
                            await target.put(None)

                        finished_ids.add(rid)
                    else:
                        finished_ids.add(rid)
                        self._request_times.pop(rid, None)

                # Clean up all engine state for finished requests
                for rid in finished_ids:
                    self.engine._free_slot(rid)
                    self.engine._request_deadlines.pop(rid, None)
                    self.engine._request_sampling_params.pop(rid, None)
                    self.engine._request_processors.pop(rid, None)
                    self.engine._request_logprobs.pop(rid, None)
                    self.engine.scheduler._tokens_generated_per_req.pop(rid, None)

                self.engine.scheduler.finished = [
                    r for r in self.engine.scheduler.finished
                    if r.request_id not in finished_ids
                ]

                # Resolve finished merged secondaries
                for mr in list(self.engine._merged_finished_results):
                    rid = mr.request_id
                    if rid in self._pending_futures:
                        target = self._pending_futures.pop(rid)
                        mr.elapsed_ms = (
                            time.perf_counter() - self._request_times.pop(rid, 0)
                        ) * 1000
                        if isinstance(target, asyncio.Future) and not target.done():
                            target.set_result(mr)
                            self.active_requests -= 1
                        elif isinstance(target, asyncio.Queue):
                            await target.put(None)
                self.engine._merged_finished_results.clear()

                # Yield to event loop between steps
                await asyncio.sleep(0)

            else:
                # No work — wait for a new request event (event-driven, not polling)
                if self._new_request_event:
                    self._new_request_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._new_request_event.wait(), timeout=0.01
                        )
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(0.001)
