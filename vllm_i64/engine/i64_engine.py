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
from typing import List, Dict, Optional, Callable, Set, Tuple
from dataclasses import dataclass
from collections import deque

from vllm_i64.engine.i64_scheduler import I64Scheduler, I64Batch, I64Request, RequestStatus
from vllm_i64.core.sampling import (
    SamplingParams, sample_batch, sample_batch_with_logprobs,
    apply_repetition_penalty_batch, TokenLogprob,
)
from vllm_i64.engine.sampler import I64Sampler
from vllm_i64.core.logging import get_logger

logger = get_logger("vllm_i64.engine")


class AdaptiveBatchSizer:
    """Dynamically adjust max_batch_size based on throughput."""

    def __init__(self, initial: int, min_size: int = 1, max_size: int = 128, window: int = 20):
        self.current = initial
        self.min_size = min_size
        self.max_size = max_size
        self._throughputs: deque = deque(maxlen=window)
        self.window = window

    def record(self, tokens: int, elapsed_ms: float):
        """Record a step's throughput."""
        tps = tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        self._throughputs.append(tps)

    def adjust(self) -> int:
        """Adjust batch size based on throughput trend."""
        if len(self._throughputs) < self.window:
            return self.current
        avg_tps = sum(self._throughputs) / len(self._throughputs)
        recent_tps = sum(list(self._throughputs)[-5:]) / 5

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
        max_prefill_tokens: int = 512,
        device: str = "cuda",
        enable_prefix_caching: bool = False,
        kv_cache_dtype: Optional[str] = None,
    ):
        self.model = model
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        if max_kv_blocks <= 0:  # 0 = auto
            max_kv_blocks = max(256, max_batch_size * 8)
        self.max_kv_blocks = max_kv_blocks
        self.device = device
        self.enable_prefix_caching = enable_prefix_caching
        self.kv_cache_dtype = kv_cache_dtype

        # Integer-only scheduler
        self.scheduler = I64Scheduler(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_experts=num_experts,
            max_kv_blocks=max_kv_blocks,
            max_prefill_tokens=max_prefill_tokens,
        )

        # Expert mask for routing (i64)
        self.expert_mask = np.int64(num_experts - 1)

        # Persistent input buffers (avoid per-step allocation)
        _buf_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        _max_tokens = max_batch_size * max_seq_len
        self._buf_token_ids = torch.zeros(_max_tokens, dtype=torch.long, device=_buf_device)
        self._buf_positions = torch.zeros(_max_tokens, dtype=torch.long, device=_buf_device)
        # Static buffer for CUDA graph seq_ids (avoid per-step allocation)
        self._buf_seq_ids = torch.zeros(max_batch_size, dtype=torch.long, device=_buf_device)

        # Step counter (integer)
        self.total_steps: int = 0
        self.total_tokens_generated: int = 0

        # Performance breakdown (cumulative ms)
        self._perf_total_ms: float = 0.0
        self._perf_forward_ms: float = 0.0
        self._perf_schedule_ms: float = 0.0
        self._perf_sample_ms: float = 0.0

        # Sampling parameters (configurable, default greedy)
        self.sampling_params = SamplingParams(temperature=0.0)
        self.sampler = I64Sampler(default_params=self.sampling_params)
        # Per-request sampling params for multi-user isolation
        self._request_sampling_params: Dict[int, SamplingParams] = {}

        # === KV Cache ===
        self.kv_cache = None
        self._slot_pool: deque = deque()
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

        # === Pipelined prefill/decode ===
        self._pipeline_enabled = False
        self._prefill_result_cache: Optional[Dict[int, torch.Tensor]] = None

        # === Request merging (dedup identical prompts) ===
        self._merge_enabled = False
        # prompt_hash → (primary_rid, prompt_tokens_tuple, [secondary_rids])
        self._merge_primaries: Dict[int, Tuple[int, tuple, List[int]]] = {}
        self._request_to_merge_group: Dict[int, int] = {}  # rid → prompt_hash
        # Secondary requests (NOT in scheduler): sec_rid → info dict
        self._merged_secondaries: Dict[int, dict] = {}
        # Finished secondary results, consumed by generate() / _engine_loop
        self._merged_finished_results: List['GenerationResult'] = []

        # === Metrics ===
        self.metrics = None

        # === Adaptive batch sizing ===
        self._batch_sizer = AdaptiveBatchSizer(initial=max_batch_size)

        # === LoRA Manager ===
        self._lora_manager = None

        # === VLM pixel_values (per-request, consumed on prefill) ===
        self._request_pixel_values: Dict[int, object] = {}

    def _init_kv_cache(self, max_seqs: int):
        """Initialize paged KV cache from model config."""
        from vllm_i64.core.kv_cache import PagedKVCache
        from vllm_i64.parallel.tensor_parallel import get_tp

        config = self.model.config
        tp = get_tp()
        # GQA: replicate KV heads when fewer than TP ranks
        if config.num_key_value_heads >= tp.tp_size:
            num_kv_heads = config.num_key_value_heads // tp.tp_size
        else:
            num_kv_heads = config.num_key_value_heads
        dtype = next(self.model.parameters()).dtype

        block_size = 16
        num_blocks = self.max_kv_blocks if self.max_kv_blocks > 0 else max(256, max_seqs * 8)
        # Cap max_blocks_per_seq to model's max context window (avoids huge static
        # tensor allocations in _tensor_paged_decode_attention during CUDA graph capture)
        max_pos = getattr(config, 'max_position_embeddings', 2048)
        max_blocks_per_seq = min(num_blocks, (max_pos + block_size - 1) // block_size)

        self.kv_cache = PagedKVCache(
            num_layers=config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
            head_dim=config.head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            max_seqs=max_seqs,
            max_blocks_per_seq=max_blocks_per_seq,
            dtype=dtype,
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        self._slot_pool = deque(range(max_seqs))

        if self.enable_prefix_caching:
            self.kv_cache.enable_prefix_caching()
            logger.info("Prefix caching enabled")

    def _init_cuda_graph(self, max_batch_size: int):
        """Initialize CUDA graph runner for decode steps."""
        try:
            from vllm_i64.core.cuda_graph import CUDAGraphRunner

            # Init graph buffers in KV cache for static block_table/seqlens
            if self.kv_cache is not None:
                self.kv_cache.init_graph_buffers(max_batch_size)

            # Pre-allocate static seq_ids tensor for graph mode
            self._graph_seq_ids = torch.arange(
                max_batch_size, dtype=torch.long, device=self.device
            )

            has_decode_step = hasattr(self.model, 'decode_step')

            def _graph_forward(token_ids, positions, _expert_ids):
                if has_decode_step and self.kv_cache is not None:
                    # Graph-compatible path: tensor-only KV writes
                    return self.model.decode_step(
                        token_ids=token_ids,
                        positions=positions,
                        kv_cache=self.kv_cache,
                        seq_ids_tensor=self._graph_seq_ids[:token_ids.shape[0]],
                    )
                else:
                    return self.model(
                        token_ids=token_ids,
                        positions=positions,
                        kv_cache=self.kv_cache,
                        seq_ids=list(range(token_ids.shape[0])),
                        tokens_per_seq=[1] * token_ids.shape[0],
                    )

            self.cuda_graph_runner = CUDAGraphRunner(
                forward_fn=_graph_forward,
                max_batch_size=max_batch_size,
                device=self.device,
            )
        except Exception as e:
            logger.warning("CUDA graph init failed: %s — falling back to eager mode", e)
            self.cuda_graph_runner = None

    def warmup_and_capture_graphs(self):
        """Warmup model and capture CUDA graphs for common decode batch sizes."""
        if self.cuda_graph_runner is None or self.model is None:
            return
        if self.device == "cpu":
            return
        try:
            if self.kv_cache is not None:
                self.kv_cache._graph_mode = True
                # Non-zero seqlens ensures attention kernels are captured
                if self.kv_cache._graph_cache_seqlens is not None:
                    self.kv_cache._graph_cache_seqlens.fill_(1)
            self.cuda_graph_runner.capture_common_sizes()
            if self.kv_cache is not None:
                self.kv_cache._graph_mode = False
                # Capture warmup wrote garbage to block 0 — mark dirty
                self.kv_cache._dirty_blocks.add(0)
            logger.info("CUDA graphs captured for sizes: %s", sorted(self.cuda_graph_runner._captured_sizes))
        except Exception as e:
            logger.warning("CUDA graph capture failed: %s", e)
            # Free all partially-captured graphs and release private pool memory
            if self.cuda_graph_runner is not None:
                self.cuda_graph_runner.graphs.clear()
                self.cuda_graph_runner.static_inputs.clear()
                self.cuda_graph_runner.static_outputs.clear()
            self.cuda_graph_runner = None
            if self.kv_cache is not None:
                self.kv_cache._graph_mode = False
            torch.cuda.empty_cache()

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

    def enable_pipeline(self):
        """Enable pipelined prefill/decode overlap."""
        self._pipeline_enabled = True
        self._prefill_result_cache = {}

    def enable_request_merging(self):
        """Enable request merging for identical prompts."""
        self._merge_enabled = True

    def _hash_prompt(self, token_ids: List[int]) -> int:
        """Fast hash for prompt dedup — single call, no per-token loop."""
        import hashlib
        h = hashlib.md5()
        h.update(np.array(token_ids, dtype=np.int64).tobytes())
        return int.from_bytes(h.digest()[:8], "little")

    def _allocate_slot(self, request_id: int) -> int:
        """Allocate a KV cache slot for a request."""
        if not self._slot_pool:
            return -1
        slot = self._slot_pool.popleft()
        self._request_to_slot[request_id] = slot
        return slot

    def _free_slot(self, request_id: int):
        """Free a KV cache slot when a request finishes."""
        if request_id in self._request_to_slot:
            slot = self._request_to_slot.pop(request_id)
            if self.kv_cache is not None:
                self.kv_cache.free_sequence(slot)
            if slot not in self._slot_pool:
                self._slot_pool.append(slot)

    def enable_lora(self, target_names: Optional[List[str]] = None):
        """Enable LoRA adapter support by wrapping model layers."""
        if self.model is None:
            raise RuntimeError("No model loaded")
        from vllm_i64.layers.lora import LoRAManager
        self._lora_manager = LoRAManager(self.model)
        self._lora_manager.auto_wrap(target_names)
        logger.info("LoRA enabled: %d modules wrapped", len(self._lora_manager._lora_modules))

    def load_lora_adapter(
        self,
        adapter_id: int,
        adapter_name: str,
        adapter_path: str,
        scaling: float = 1.0,
    ) -> bool:
        """Load a LoRA adapter from safetensors or PyTorch file."""
        if self._lora_manager is None:
            return False
        import os
        weights = {}
        if adapter_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            weights = load_file(adapter_path)
        elif adapter_path.endswith((".pt", ".pth", ".bin")):
            weights = torch.load(adapter_path, map_location="cpu", weights_only=True)
        elif os.path.isdir(adapter_path):
            # Try safetensors in directory
            for fname in os.listdir(adapter_path):
                if fname.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    weights.update(load_file(os.path.join(adapter_path, fname)))
        if not weights:
            return False
        self._lora_manager.load_adapter(adapter_id, adapter_name, weights, scaling=scaling)
        logger.info("LoRA adapter loaded: %s (id=%d)", adapter_name, adapter_id)
        return True

    def unload_lora_adapter(self, adapter_id: int):
        """Unload a LoRA adapter."""
        if self._lora_manager is not None:
            self._lora_manager.unload_adapter(adapter_id)

    def set_active_lora(self, adapter_id: Optional[int]):
        """Set the active LoRA adapter (None = base model)."""
        if self._lora_manager is not None:
            self._lora_manager.set_active_adapter(adapter_id)

    def list_lora_adapters(self) -> Dict[int, str]:
        """List loaded LoRA adapters."""
        if self._lora_manager is None:
            return {}
        return self._lora_manager.list_adapters()

    def embed(self, token_ids: List[int]) -> List[float]:
        """
        Compute embeddings for token IDs.
        Uses model's embedding layer + final norm, then mean-pools.
        Returns a flat list of floats.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            if hasattr(self.model, 'embed_tokens') and self.model.embed_tokens is not None:
                hidden = self.model.embed_tokens(ids)
            else:
                raise RuntimeError("Model has no embedding layer")

            # Apply final norm if available
            if hasattr(self.model, 'norm') and self.model.norm is not None:
                hidden = self.model.norm(hidden)

            # Mean pool over token dimension
            embedding = hidden.mean(dim=0)

        return embedding.cpu().float().tolist()

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        timeout_s: Optional[float] = None,
        sampling_params: Optional[SamplingParams] = None,
        pixel_values: Optional[object] = None,
    ) -> int:
        """Add request. Returns integer request_id."""
        ids = np.array(prompt_token_ids, dtype=np.int64)

        # Get EOS token ID from model config (-1 = no EOS when no model)
        eos_token_id = -1
        if self.model is not None and hasattr(self.model, 'config'):
            eos_token_id = getattr(self.model.config, 'eos_token_id', 0)

        # Request merging: if identical prompt already running, piggyback on it
        if self._merge_enabled:
            prompt_hash = self._hash_prompt(prompt_token_ids)
            if prompt_hash in self._merge_primaries:
                primary_rid, primary_prompt, sec_list = self._merge_primaries[prompt_hash]
                sp = sampling_params or self.sampling_params
                # Only merge if: exact same prompt (not just hash), greedy, primary still active
                if (sp.temperature == 0.0
                        and tuple(prompt_token_ids) == primary_prompt
                        and primary_rid in self._request_to_merge_group):
                    # Allocate unique ID without scheduling (no compute, no KV slot)
                    sec_rid = self.scheduler.next_request_id
                    self.scheduler.next_request_id += 1
                    self._merged_secondaries[sec_rid] = {
                        "prompt_tokens": list(prompt_token_ids),
                        "output_tokens": [],
                        "max_new_tokens": max_new_tokens,
                    }
                    sec_list.append(sec_rid)
                    self._request_to_merge_group[sec_rid] = prompt_hash
                    t = timeout_s if timeout_s is not None else self.default_timeout_s
                    if t > 0:
                        self._request_deadlines[sec_rid] = time.perf_counter() + t
                    return sec_rid

        request_id = self.scheduler.add_request(ids, max_new_tokens, eos_token_id=eos_token_id)

        # Register as merge primary if merging is enabled
        if self._merge_enabled:
            prompt_hash = self._hash_prompt(prompt_token_ids)
            self._merge_primaries[prompt_hash] = (request_id, tuple(prompt_token_ids), [])
            self._request_to_merge_group[request_id] = prompt_hash

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
            slot = self._allocate_slot(request_id)
            if slot < 0:
                logger.error("No KV cache slots available for request %d", request_id)
                # Remove the request from the scheduler so it doesn't loop forever
                self.scheduler.running = [
                    r for r in self.scheduler.running if r.request_id != request_id
                ]
                raise RuntimeError(f"No KV cache slots available for request {request_id}")

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
                        logger.info("Prefix cache hit: reused %d tokens for request %d", reused, request_id)

        # Store pixel_values for VLM prefill (consumed and freed after first forward)
        if pixel_values is not None:
            self._request_pixel_values[request_id] = pixel_values

        return request_id

    def cancel_request(self, request_id: int):
        """Cancel a running or pending request."""
        self._cancelled_requests.add(request_id)

    def _i64_sample(self, logits: torch.Tensor, past_tokens_list=None) -> np.ndarray:
        """
        Sample next tokens from logits using configured sampling params.
        """
        return self.sampler.sample(logits, self.sampling_params, past_tokens_list=past_tokens_list)

    def _check_timeouts_and_cancellations(self):
        """Remove timed-out and cancelled requests from scheduler. Single-pass O(n)."""
        now = time.perf_counter()
        removed = set()

        # Single pass over running: check both cancellations and timeouts
        still_running = []
        for req in self.scheduler.running:
            rid = req.request_id
            is_cancelled = rid in self._cancelled_requests
            is_timed_out = False
            if not is_cancelled:
                deadline = self._request_deadlines.get(rid)
                is_timed_out = deadline is not None and now > deadline

            if is_cancelled or is_timed_out:
                req.status = RequestStatus.FINISHED
                req._finish_reason = "cancelled" if is_cancelled else "timeout"
                if is_timed_out:
                    logger.warning("Request %d timed out", rid)
                self._free_kv_blocks_for(req)
                self.scheduler.finished.append(req)
                self._free_slot(rid)
                removed.add(rid)
                # Clean up merge group if this was a primary
                if self._merge_enabled:
                    phash = self._request_to_merge_group.pop(rid, None)
                    if phash and phash in self._merge_primaries:
                        primary_rid, _, sec_rids = self._merge_primaries[phash]
                        if rid == primary_rid:
                            fr = "cancelled" if is_cancelled else "timeout"
                            for sec_rid in sec_rids:
                                sec = self._merged_secondaries.pop(sec_rid, None)
                                if sec is not None:
                                    self._merged_finished_results.append(GenerationResult(
                                        request_id=sec_rid,
                                        prompt_tokens=sec["prompt_tokens"],
                                        output_tokens=sec["output_tokens"],
                                        num_steps=len(sec["output_tokens"]),
                                        elapsed_ms=0,
                                        finish_reason=fr,
                                    ))
                                self._request_to_merge_group.pop(sec_rid, None)
                                self._request_deadlines.pop(sec_rid, None)
                            del self._merge_primaries[phash]
            else:
                still_running.append(req)
        self.scheduler.running = still_running

        # Single pass over pending: only cancellations apply
        if self._cancelled_requests:
            still_pending = []
            for req in self.scheduler.pending:
                if req.request_id in self._cancelled_requests:
                    req.status = RequestStatus.FINISHED
                    req._finish_reason = "cancelled"
                    self.scheduler.finished.append(req)
                    removed.add(req.request_id)
                else:
                    still_pending.append(req)
            self.scheduler.pending = still_pending

        self._cancelled_requests -= removed
        for rid in removed:
            self._request_deadlines.pop(rid, None)

    def _free_kv_blocks_for(self, req):
        """Free KV blocks owned by a request."""
        if req.kv_block_ids:
            self.scheduler._free_kv_blocks(req.kv_block_ids)
            req.kv_block_ids = []

    def _speculative_step(self, batch: I64Batch, _running_index: Optional[Dict[int, I64Request]] = None) -> Dict[int, int]:
        """
        Speculative decode step for small decode-only batches.

        Uses draft model to generate K tokens, verifies with target model.
        Multi-token acceptance → fewer forward passes → higher throughput.
        """
        if _running_index is None:
            _running_index = {req.request_id: req for req in self.scheduler.running}
        result = {}
        for i, req_id in enumerate(batch.request_ids):
            rid = int(req_id)
            req = _running_index.get(rid)
            if req is None:
                continue

            # Build context for speculative decoder
            ctx = torch.tensor(
                req.prompt_list + req.output_token_ids,
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
        _step_start = time.perf_counter()

        # 0. Clean up finished from PREVIOUS step (free slots, clear per-request state)
        #    Must happen before schedule() so that newly finished requests remain
        #    visible to callers (generate(), _engine_loop) after this step returns.
        for req in self.scheduler.finished:
            rid = req.request_id
            self._free_slot(rid)
            self._request_sampling_params.pop(rid, None)
            self._request_processors.pop(rid, None)
            self._request_deadlines.pop(rid, None)
            self._request_logprobs.pop(rid, None)
            self._request_pixel_values.pop(rid, None)
            self.scheduler._tokens_generated_per_req.pop(rid, None)
            # Clean up merge group
            if self._merge_enabled:
                phash = self._request_to_merge_group.pop(rid, None)
                if phash is not None and phash in self._merge_primaries:
                    primary_rid, _, sec_rids = self._merge_primaries[phash]
                    if rid == primary_rid:
                        # Primary finished — finish all remaining secondaries
                        eos_id = 0
                        if self.model is not None and hasattr(self.model, 'config'):
                            eos_id = getattr(self.model.config, 'eos_token_id', 0)
                        for sec_rid in sec_rids:
                            sec = self._merged_secondaries.pop(sec_rid, None)
                            if sec is not None:
                                fr = "length"
                                if sec["output_tokens"] and sec["output_tokens"][-1] == eos_id:
                                    fr = "stop"
                                self._merged_finished_results.append(GenerationResult(
                                    request_id=sec_rid,
                                    prompt_tokens=sec["prompt_tokens"],
                                    output_tokens=sec["output_tokens"],
                                    num_steps=len(sec["output_tokens"]),
                                    elapsed_ms=0,
                                    finish_reason=fr,
                                ))
                            self._request_to_merge_group.pop(sec_rid, None)
                            self._request_deadlines.pop(sec_rid, None)
                        del self._merge_primaries[phash]
        self.scheduler.finished.clear()

        # 1. Handle timeouts and cancellations
        self._check_timeouts_and_cancellations()

        # 1.0.1 Check secondary merge requests for timeouts/cancellations/orphans
        if self._merge_enabled and self._merged_secondaries:
            now = time.perf_counter()
            # Build set of active primary rids for orphan detection
            active_primaries = {
                prid for _, (prid, _, _) in self._merge_primaries.items()
            }
            for sec_rid in list(self._merged_secondaries.keys()):
                is_cancelled = sec_rid in self._cancelled_requests
                deadline = self._request_deadlines.get(sec_rid)
                is_timed_out = deadline is not None and now > deadline
                # Orphan: secondary's primary no longer exists
                phash = self._request_to_merge_group.get(sec_rid)
                is_orphan = (phash is None
                             or phash not in self._merge_primaries
                             or self._merge_primaries[phash][0] not in active_primaries)
                if is_cancelled or is_timed_out or is_orphan:
                    sec = self._merged_secondaries.pop(sec_rid)
                    fr = "cancelled" if is_cancelled else ("timeout" if is_timed_out else "length")
                    self._merged_finished_results.append(GenerationResult(
                        request_id=sec_rid,
                        prompt_tokens=sec["prompt_tokens"],
                        output_tokens=sec["output_tokens"],
                        num_steps=len(sec["output_tokens"]),
                        elapsed_ms=0,
                        finish_reason=fr,
                    ))
                    phash = self._request_to_merge_group.pop(sec_rid, None)
                    if phash and phash in self._merge_primaries:
                        _, _, sec_rids = self._merge_primaries[phash]
                        if sec_rid in sec_rids:
                            sec_rids.remove(sec_rid)
                    self._cancelled_requests.discard(sec_rid)
                    self._request_deadlines.pop(sec_rid, None)

        # 1.1 Auto-enable FP8 KV cache under memory pressure
        if self.kv_cache is not None:
            self.kv_cache.maybe_enable_fp8()

        # 2. Schedule (i64) — moves is_finished requests to scheduler.finished
        _t_sched = time.perf_counter()
        batch = self.scheduler.schedule()
        _t_sched_end = time.perf_counter()
        self._perf_schedule_ms += (_t_sched_end - _t_sched) * 1000

        if batch is None:
            return {}

        # Update metrics
        if self.metrics:
            self.metrics.update_batch_size(batch.num_requests)
            self.metrics.update_pending(len(self.scheduler.pending))
            if self.kv_cache:
                kv_stats = self.kv_cache.get_stats()
                self.metrics.update_kv_usage(kv_stats["used_blocks"], kv_stats["num_blocks"])

        # Build request index for O(1) lookups
        # Only rebuild if running list changed (skip if same object ids)
        _running_index: Dict[int, I64Request] = {
            req.request_id: req for req in self.scheduler.running
        }

        # 1.5. Speculative decoding for decode-only batches (threshold batch<=8)
        if (self.speculative_decoder is not None
                and not batch.is_prefill.any()
                and batch.num_requests <= 8):
            result = self._speculative_step(batch, _running_index)
            self.scheduler.update_after_step(result)
            # NOTE: finished request cleanup (slot freeing) happens at top of next step()
            self.total_steps += 1
            self.total_tokens_generated += len(result)
            return result

        # Pin ALL running sequences (not just this batch) so LRU eviction
        # during block allocation can't corrupt any active request's KV data.
        if self.kv_cache is not None:
            pinned = set()
            for req in self.scheduler.running:
                slot = self._request_to_slot.get(req.request_id)
                if slot is not None:
                    pinned.add(slot)
            self.kv_cache._pinned_seq_ids = pinned

        # 2. Model forward (fp16) — the ONLY float step
        _t_fwd = time.perf_counter()
        try:
            if self.model is not None:
                logits = self._model_forward(batch)
            else:
                logits = torch.randn(batch.num_requests, self.vocab_size)
        except Exception as e:
            self._perf_forward_ms += (time.perf_counter() - _t_fwd) * 1000
            if self.kv_cache is not None:
                self.kv_cache._pinned_seq_ids.clear()
            raise  # propagate to _engine_loop error handler

        self._perf_forward_ms += (time.perf_counter() - _t_fwd) * 1000

        # Unpin after forward
        if self.kv_cache is not None:
            self.kv_cache._pinned_seq_ids.clear()

        # 3. Extract last-token logits per request (for prefill: many tokens → 1 logit)
        if batch.tokens_per_request is not None and logits.shape[0] != batch.num_requests:
            tpr_list = batch.tokens_per_request.tolist()  # One GPU→CPU transfer
            last_indices = []
            offset = 0
            for tpr in tpr_list:
                last_indices.append(offset + tpr - 1)
                offset += tpr
            logits = logits[last_indices]  # (num_requests, vocab_size)

        # 3.5. Per-request sampling with isolated params
        _t_sample = time.perf_counter()
        # Convert request_ids to Python int list once (avoid repeated int() casts)
        request_id_list = batch.request_ids.tolist()

        # Check if any request has custom sampling params
        has_custom = any(
            rid in self._request_sampling_params
            for rid in request_id_list
        )

        if has_custom:
            # Partition requests: "complex" (need individual handling) vs "batchable"
            # Complex = has logits processors, logprobs, or min_tokens
            result = {}
            complex_indices = []  # (batch_idx, rid)
            # Group batchable by params id → list of (batch_idx, rid)
            batchable_groups: Dict[int, list] = {}

            for i, rid in enumerate(request_id_list):
                params = self._request_sampling_params.get(rid, self.sampling_params)
                needs_individual = (
                    rid in self._request_processors
                    or params.logprobs is not None
                    or params.min_tokens > 0
                )
                if needs_individual:
                    complex_indices.append((i, rid))
                else:
                    batchable_groups.setdefault(id(params), []).append((i, rid, params))

            # Batch-sample each group of identical params together
            for _, group in batchable_groups.items():
                indices = [g[0] for g in group]
                rids = [g[1] for g in group]
                params = group[0][2]  # All share same params object
                group_logits = logits[indices]

                past_tokens_list = None
                if params.repetition_penalty != 1.0:
                    past_tokens_list = []
                    for _, rid, _ in group:
                        req = _running_index.get(rid)
                        if req is not None:
                            past_tokens_list.append(req.prompt_list + req.output_token_ids)
                        else:
                            past_tokens_list.append([])

                token_ids = sample_batch(group_logits, params, past_tokens_list=past_tokens_list)
                token_list = token_ids.tolist()
                for rid, tid in zip(rids, token_list):
                    result[rid] = tid

            # Handle complex requests individually
            for i, rid in complex_indices:
                params = self._request_sampling_params.get(rid, self.sampling_params)
                req_logits = logits[i:i+1]
                req = _running_index.get(rid)

                # Apply min_tokens: suppress EOS until minimum tokens generated
                if params.min_tokens > 0 and req is not None:
                    eos_id = getattr(req, 'eos_token_id', None)
                    if eos_id is not None:
                        from vllm_i64.core.sampling import apply_min_tokens
                        req_logits = apply_min_tokens(
                            req_logits, req.num_generated, params.min_tokens, eos_id
                        )

                past_tokens = None
                if params.repetition_penalty != 1.0:
                    if req is not None:
                        past_tokens = [req.prompt_list + req.output_token_ids]
                    else:
                        past_tokens = [[]]

                # Apply logits processors if configured
                if rid in self._request_processors:
                    from vllm_i64.core.logits_processor import apply_logits_processors
                    generated = req.output_token_ids if req else []
                    req_logits = apply_logits_processors(
                        req_logits.squeeze(0), self._request_processors[rid], generated
                    ).unsqueeze(0)
                    for proc in self._request_processors[rid]:
                        if hasattr(proc, 'should_stop') and proc.should_stop:
                            if req is not None:
                                req.status = RequestStatus.FINISHED
                                req._finish_reason = "stop"
                            break

                if params.logprobs is not None:
                    sample_out = sample_batch_with_logprobs(req_logits, params, past_tokens_list=past_tokens)
                    token_id = sample_out.token_ids[0].item()
                    if sample_out.logprobs:
                        self._request_logprobs.setdefault(rid, []).append(sample_out.logprobs[0])
                    result[rid] = token_id
                else:
                    token_ids = sample_batch(req_logits, params, past_tokens_list=past_tokens)
                    result[rid] = token_ids[0].item()
        else:
            # Fast path: all requests share the same params
            past_tokens_list = None
            if self.sampling_params.repetition_penalty != 1.0:
                past_tokens_list = []
                for rid in request_id_list:
                    req = _running_index.get(rid)
                    if req is not None:
                        past_tokens_list.append(req.prompt_list + req.output_token_ids)
                    else:
                        past_tokens_list.append([])

            # 4. Sample (configurable, includes repetition penalty)
            new_token_ids = self._i64_sample(logits, past_tokens_list)

            # 5. Map back to requests (integer indexing)
            new_tokens_list = new_token_ids.tolist()
            result = dict(zip(request_id_list, new_tokens_list))

        # 5.5 Propagate primary tokens to merged secondaries (no compute — just copy)
        if self._merge_enabled and self._merge_primaries:
            eos_id = 0
            if self.model is not None and hasattr(self.model, 'config'):
                eos_id = getattr(self.model.config, 'eos_token_id', 0)
            for phash, (primary_rid, _, sec_rids) in list(self._merge_primaries.items()):
                if primary_rid not in result:
                    continue
                token_id = result[primary_rid]
                done_secs = []
                for sec_rid in sec_rids:
                    sec = self._merged_secondaries.get(sec_rid)
                    if sec is None:
                        done_secs.append(sec_rid)
                        continue
                    sec["output_tokens"].append(token_id)
                    # Add to step result so streaming works in _engine_loop
                    result[sec_rid] = token_id
                    # Check if secondary is done
                    if (len(sec["output_tokens"]) >= sec["max_new_tokens"]
                            or token_id == eos_id):
                        fr = "stop" if token_id == eos_id else "length"
                        self._merged_finished_results.append(GenerationResult(
                            request_id=sec_rid,
                            prompt_tokens=sec["prompt_tokens"],
                            output_tokens=sec["output_tokens"],
                            num_steps=len(sec["output_tokens"]),
                            elapsed_ms=0,
                            finish_reason=fr,
                        ))
                        del self._merged_secondaries[sec_rid]
                        self._request_to_merge_group.pop(sec_rid, None)
                        self._request_deadlines.pop(sec_rid, None)
                        done_secs.append(sec_rid)
                for s in done_secs:
                    if s in sec_rids:
                        sec_rids.remove(s)

        self._perf_sample_ms += (time.perf_counter() - _t_sample) * 1000

        # 6. Update scheduler (i64)
        self.scheduler.update_after_step(result)

        # 6.5 Register prefix blocks for requests that just completed prefill
        if self.kv_cache is not None and self.kv_cache.prefix_cache_enabled:
            for req in self.scheduler.running:
                if req.request_id in result and req.prefill_complete and req.num_generated == 1:
                    slot = self._request_to_slot.get(req.request_id)
                    if slot is not None:
                        self.kv_cache.register_prefix_blocks(slot, req.prompt_list)

        # Integer counters
        self.total_steps += 1
        self.total_tokens_generated += len(result)

        # Adaptive batch sizing: record throughput and adjust
        step_elapsed = (time.perf_counter() - _step_start) * 1000
        self._perf_total_ms += step_elapsed
        if result:
            self._batch_sizer.record(len(result), step_elapsed)
            new_max = self._batch_sizer.adjust()
            if new_max != self.scheduler.max_batch_size:
                self.scheduler.max_batch_size = new_max

        return result

    def _model_forward(self, batch: I64Batch) -> torch.Tensor:
        """
        Run model forward pass with KV cache support.
        """
        n = batch.token_ids.shape[0]
        if n <= self._buf_token_ids.shape[0]:
            self._buf_token_ids[:n].copy_(torch.from_numpy(batch.token_ids))
            self._buf_positions[:n].copy_(torch.from_numpy(batch.positions))
            token_ids = self._buf_token_ids[:n]
            positions = self._buf_positions[:n]
        else:
            token_ids = torch.from_numpy(batch.token_ids).to(self.device)
            positions = torch.from_numpy(batch.positions).to(self.device)

        # Build KV cache metadata
        seq_ids = None
        tokens_per_seq = None

        if self.kv_cache is not None and batch.tokens_per_request is not None:
            seq_ids = []
            valid_indices = []
            for i, rid in enumerate(batch.request_ids):
                slot = self._request_to_slot.get(int(rid))
                if slot is None:
                    logger.error("Request %d has no KV slot — skipping from batch", int(rid))
                    continue
                seq_ids.append(slot)
                valid_indices.append(i)
            tokens_per_seq = [batch.tokens_per_request[i] for i in valid_indices]

        # CUDA graph for pure decode batches. Now supports KV cache via
        # model.decode_step() which uses tensor-only KV writes (graph-safe).
        use_graph = (
            self.cuda_graph_runner is not None
            and self.cuda_graph_runner.is_captured
            and not batch.is_prefill.any()
            and seq_ids is not None
            and len(seq_ids) > 0
            and hasattr(self.model, 'decode_step')
        )

        with torch.no_grad():
            if use_graph:
                # Enter graph mode: copy block_table + seqlens to static buffers
                self.kv_cache.enter_graph_mode(seq_ids)

                expert_ids = torch.from_numpy(batch.expert_ids).to(self.device)
                logits = self.cuda_graph_runner.run(token_ids, positions, expert_ids)

                self.kv_cache.exit_graph_mode()

                # Sync real seq_lens from graph (decode adds 1 token per seq)
                n = len(seq_ids)
                self._buf_seq_ids[:n] = torch.from_numpy(np.array(seq_ids, dtype=np.int64))
                seq_ids_tensor = self._buf_seq_ids[:n]
                self.kv_cache.seq_lens[seq_ids_tensor] += 1
                # Touch LRU tracking
                for sid in seq_ids:
                    self.kv_cache._touch(sid)
            else:
                # Collect pixel_values for VLM prefill requests
                pixel_values = None
                if batch.is_prefill.any() and self._request_pixel_values:
                    import torch as _torch
                    pv_list = []
                    for rid in batch.request_ids:
                        rid_int = int(rid)
                        if rid_int in self._request_pixel_values:
                            pv = self._request_pixel_values.pop(rid_int)
                            pv_list.append(pv)
                    if pv_list:
                        pixel_values = _torch.cat(pv_list, dim=0).to(self.device)

                # Pass pixel_values if model supports it (VLM)
                fwd_kwargs = dict(
                    token_ids=token_ids,
                    positions=positions,
                    kv_cache=self.kv_cache,
                    seq_ids=seq_ids,
                    tokens_per_seq=tokens_per_seq,
                )
                if pixel_values is not None and hasattr(self.model, 'vision_encoder'):
                    fwd_kwargs["pixel_values"] = pixel_values

                logits = self.model(**fwd_kwargs)

        return logits

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        sampling_params: Optional[SamplingParams] = None,
        pixel_values: Optional[object] = None,
    ) -> GenerationResult:
        """Synchronous generation for a single request."""
        request_id = self.add_request(
            prompt_token_ids, max_new_tokens,
            sampling_params=sampling_params,
            pixel_values=pixel_values,
        )

        metrics_start = None
        if self.metrics:
            metrics_start = self.metrics.on_request_start()

        start = time.perf_counter()
        steps = 0

        while True:
            self.step()
            steps += 1

            # Check merged secondary finished results first
            for i, mr in enumerate(self._merged_finished_results):
                if mr.request_id == request_id:
                    self._merged_finished_results.pop(i)
                    mr.elapsed_ms = (time.perf_counter() - start) * 1000
                    mr.num_steps = steps
                    return mr

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

                # Truncate stop sequence tokens from output
                output_tokens = req.output_token_ids
                if request_id in self._request_processors:
                    for proc in self._request_processors[request_id]:
                        if hasattr(proc, 'should_stop') and proc.should_stop:
                            idx = proc.stop_index
                            if 0 <= idx < len(output_tokens):
                                output_tokens = output_tokens[:idx]
                                if token_logprobs:
                                    token_logprobs = token_logprobs[:idx]
                            break

                return GenerationResult(
                    request_id=request_id,
                    prompt_tokens=req.prompt_list,
                    output_tokens=output_tokens,
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

    def get_stats(self) -> dict:
        """Engine stats — integers + perf breakdown."""
        sched_stats = self.scheduler.get_stats()
        stats = {
            **sched_stats,
            "total_steps": self.total_steps,
            "total_tokens_generated": self.total_tokens_generated,
        }
        if self.kv_cache:
            stats.update(self.kv_cache.get_stats())
        # Performance breakdown
        if self.total_steps > 0 and self._perf_total_ms > 0:
            overhead_ms = self._perf_total_ms - self._perf_forward_ms - self._perf_schedule_ms - self._perf_sample_ms
            stats["perf"] = {
                "total_ms": round(self._perf_total_ms, 1),
                "forward_ms": round(self._perf_forward_ms, 1),
                "schedule_ms": round(self._perf_schedule_ms, 1),
                "sample_ms": round(self._perf_sample_ms, 1),
                "overhead_ms": round(max(overhead_ms, 0), 1),
                "forward_pct": round(self._perf_forward_ms / self._perf_total_ms * 100, 1),
                "avg_step_ms": round(self._perf_total_ms / self.total_steps, 2),
                "tok_per_s": round(self.total_tokens_generated / (self._perf_total_ms / 1000), 1) if self._perf_total_ms > 0 else 0,
            }
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
        self.device = device
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
        self._request_first_token_sent: Set[int] = set()  # Track TTFT
        self._engine_task: Optional[asyncio.Task] = None
        self._running = False
        self._draining = False
        self._new_request_event: Optional[asyncio.Event] = None  # Set lazily in start()

        # Stats
        self.active_requests: int = 0
        self.peak_batch_size: int = 0
        self.max_queue_depth: int = max_batch_size * 8  # Backpressure limit

    @classmethod
    def from_sync_engine(cls, engine: 'I64Engine') -> 'AsyncI64Engine':
        """Create an AsyncI64Engine wrapping an existing sync engine."""
        instance = cls.__new__(cls)
        instance.engine = engine
        instance._pending_futures = {}
        instance._request_times = {}
        instance._request_first_token_sent = set()
        instance._engine_task = None
        instance._running = False
        instance._draining = False
        instance._new_request_event = None
        instance.device = engine.device
        instance.active_requests = 0
        instance.peak_batch_size = 0
        instance.max_queue_depth = engine.scheduler.max_batch_size * 8
        return instance

    async def start(self):
        """Start the continuous batching background loop."""
        self._running = True
        self._new_request_event = asyncio.Event()
        self._engine_task = asyncio.create_task(self._engine_loop())

    async def stop(self, drain_timeout: float = 30.0):
        """
        Graceful shutdown: drain active requests before stopping.

        Args:
            drain_timeout: max seconds to wait for in-flight requests
        """
        logger.info("Engine shutdown requested, draining %d requests...", self.active_requests)
        self._draining = True

        # Wait for in-flight requests to finish
        deadline = time.perf_counter() + drain_timeout
        while self.active_requests > 0 and time.perf_counter() < deadline:
            await asyncio.sleep(0.05)

        if self.active_requests > 0:
            logger.warning("Drain timeout: %d requests still active, cancelling", self.active_requests)
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
        pixel_values: Optional[object] = None,
    ) -> GenerationResult:
        """
        Submit a request and wait for completion.

        Multiple concurrent calls are automatically batched together
        by the engine loop for maximum throughput.
        """
        if self._draining:
            raise RuntimeError("Engine is shutting down, not accepting new requests")
        if self.active_requests >= self.max_queue_depth:
            raise RuntimeError(f"Queue full ({self.active_requests}/{self.max_queue_depth}), try again later")

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        request_id = self.engine.add_request(
            prompt_token_ids, max_new_tokens, timeout_s=timeout_s,
            sampling_params=sampling_params,
            pixel_values=pixel_values,
        )
        self._pending_futures[request_id] = future
        self._request_times[request_id] = time.perf_counter()
        self.active_requests += 1
        if self._new_request_event:
            self._new_request_event.set()  # Wake engine loop immediately

        return await future

    async def generate_stream(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        sampling_params: Optional[SamplingParams] = None,
        pixel_values=None,
    ):
        """
        Submit a request and yield tokens as they are generated.
        """
        request_id = self.engine.add_request(
            prompt_token_ids, max_new_tokens,
            sampling_params=sampling_params,
            pixel_values=pixel_values,
        )
        self._request_times[request_id] = time.perf_counter()
        self.active_requests += 1
        if self._new_request_event:
            self._new_request_event.set()  # Wake engine loop immediately

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
            # Cancel engine-side request if still running (e.g. client disconnect)
            # Only cancel if not already finished (avoids double-free)
            if request_id in self._pending_futures:
                self._pending_futures.pop(request_id, None)
                self.engine.cancel_request(request_id)
            self.active_requests -= 1
            self._request_times.pop(request_id, None)
            self._request_first_token_sent.discard(request_id)

    async def _engine_loop(self):
        """Continuous batching loop with crash recovery and event-driven wake-up."""
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
                    # Run step() in a thread executor on CPU to avoid blocking
                    # the event loop during the forward pass (which can take
                    # several seconds on CPU, causing SSE connection timeouts).
                    _step_start = time.perf_counter()
                    if self.device == "cpu":
                        loop = asyncio.get_event_loop()
                        step_results = await loop.run_in_executor(None, self.engine.step)
                    else:
                        step_results = self.engine.step()
                    _step_elapsed = time.perf_counter() - _step_start
                    _consecutive_errors = 0  # Reset on success

                    # ITL: observe per-step decode latency
                    if self.engine.metrics and step_results:
                        self.engine.metrics.observe_itl(_step_elapsed)
                except Exception as e:
                    _consecutive_errors += 1
                    logger.error("Engine step failed (%d/%d): %s", _consecutive_errors, _MAX_CONSECUTIVE_ERRORS, e)

                    # Fail stuck requests so clients get an error response
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
                        logger.error("Too many consecutive errors, engine loop stopping")
                        break

                    await asyncio.sleep(0.1)
                    continue

                # Deliver tokens to streaming futures
                for req_id, token_id in step_results.items():
                    # TTFT: record time to first token
                    if req_id not in self._request_first_token_sent:
                        self._request_first_token_sent.add(req_id)
                        if self.engine.metrics and req_id in self._request_times:
                            ttft = time.perf_counter() - self._request_times[req_id]
                            self.engine.metrics.observe_ttft(ttft)

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
                                token_logprobs=token_logprobs if token_logprobs else None,
                            )
                            target.set_result(result)
                            self.active_requests -= 1
                        elif isinstance(target, asyncio.Queue):
                            # Signal end to generate_stream; its finally block handles active_requests
                            await target.put(None)
                        finished_ids.add(rid)
                    else:
                        # Orphan finished request (no future) — clean it up
                        finished_ids.add(rid)
                        self._request_times.pop(rid, None)

                # Clean up finished + all associated engine state
                for rid in finished_ids:
                    self._request_first_token_sent.discard(rid)
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
                        mr.elapsed_ms = (time.perf_counter() - self._request_times.pop(rid, 0)) * 1000
                        if isinstance(target, asyncio.Future) and not target.done():
                            target.set_result(mr)
                            self.active_requests -= 1
                        elif isinstance(target, asyncio.Queue):
                            await target.put(None)
                self.engine._merged_finished_results.clear()

                await asyncio.sleep(0)
            else:
                # Event-driven: wait for new request signal instead of polling
                if self._new_request_event:
                    self._new_request_event.clear()
                    try:
                        await asyncio.wait_for(self._new_request_event.wait(), timeout=0.01)
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(0.001)

    def get_stats(self) -> Dict[str, int]:
        """Engine + async stats."""
        stats = self.engine.get_stats()
        stats["active_requests"] = self.active_requests
        stats["peak_batch_size"] = self.peak_batch_size
        return stats
