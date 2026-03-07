"""
vllm-i64 :: Disaggregated Prefill/Decode

Splits inference across two GPU groups:
  - Prefill GPU(s): run compute-bound prompt encoding, produce KV cache
  - Decode GPU(s): run memory-bound autoregressive token generation

Workflow:
  1. New request arrives at DisaggregatedCoordinator
  2. Coordinator routes prompt to PrefillWorker
  3. PrefillWorker runs model.forward() for full prompt, produces KV blocks
  4. KVTransfer sends KV blocks from prefill GPU to decode GPU (NCCL/TCP)
  5. DecodeWorker receives KV + first token, runs autoregressive decode
  6. Completed tokens stream back through Coordinator

Fallback: single GPU runs both prefill and decode (no disaggregation).

Key design:
  - Prefill GPU: high throughput for long prompts (compute-bound)
  - Decode GPU: low latency for token-by-token generation (memory-bound)
  - KV transfer uses torch.distributed (NCCL preferred, TCP fallback)
  - PagedKVCache format is shared — block-level serialization

INL - 2025
"""

import os
import time
import queue
import threading
import numpy as np
import torch
import torch.distributed as dist
from typing import Callable, List, Dict, Optional
from dataclasses import dataclass, field
from enum import IntEnum

from vllm_i64.core.logging import get_logger

logger = get_logger("vllm_i64.parallel.disaggregated")


# =========================================================================
# Data structures
# =========================================================================

class DisaggRole(IntEnum):
    """Role assignment for a GPU in disaggregated mode."""
    PREFILL = 0
    DECODE = 1


@dataclass
class KVTransferMetadata:
    """
    Metadata sent alongside KV blocks during prefill->decode transfer.
    All fields are integers — zero float in the control plane.
    """
    request_id: int
    prompt_token_ids: List[int]
    first_token_id: int           # First generated token (from prefill sampling)
    seq_len: int                  # Number of KV positions filled
    num_kv_blocks: int            # Number of physical blocks to transfer
    block_ids: List[int]          # Physical block IDs on prefill side
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    max_new_tokens: int           # Remaining tokens to generate
    eos_token_id: int = 0
    sampling_temperature: float = 0.0  # Only float: passed through for decode


class DisaggStatus(IntEnum):
    """Disaggregated request lifecycle."""
    PENDING = 0
    PREFILLING = 1
    TRANSFERRING = 2
    DECODING = 3
    FINISHED = 4


@dataclass
class DisaggRequest:
    """
    A request tracked by the coordinator.
    Integer lifecycle: PENDING -> PREFILLING -> TRANSFERRING -> DECODING -> FINISHED
    """
    request_id: int
    prompt_token_ids: List[int]
    max_new_tokens: int
    status: int = DisaggStatus.PENDING
    output_tokens: List[int] = field(default_factory=list)
    first_token_id: int = 0
    sampling_temperature: float = 0.0
    eos_token_id: int = 0
    arrival_time: float = 0.0
    finish_time: float = 0.0


# =========================================================================
# KVTransfer — serialize/deserialize KV cache blocks between GPUs
# =========================================================================

class KVTransfer:
    """
    Transfers KV cache blocks between prefill and decode GPUs.

    Uses torch.distributed for inter-GPU communication:
      - NCCL backend: fastest, GPU-direct RDMA when available
      - Gloo/TCP fallback: works across any network

    Transfer protocol (per request):
      1. Send metadata tensor (fixed size, integer)
      2. Send KV blocks layer by layer: k_block, v_block per layer
      3. Decode side receives and writes into its own PagedKVCache

    All metadata is integer. Only KV tensor data is float (inherent to model).
    """

    def __init__(
        self,
        prefill_rank: int,
        decode_rank: int,
        process_group: Optional[dist.ProcessGroup] = None,
        device: str = "cuda",
    ):
        self.prefill_rank = prefill_rank
        self.decode_rank = decode_rank
        self.group = process_group
        self.device = device

        # Pre-allocated metadata buffer (avoid per-transfer allocation)
        # Format: [request_id, seq_len, num_blocks, num_layers, num_kv_heads,
        #          head_dim, block_size, first_token_id, max_new_tokens, eos_token_id]
        self._meta_size = 10
        self._meta_buf = torch.zeros(self._meta_size, dtype=torch.int64, device=device)
        import threading
        self._send_lock = threading.Lock()

        # Transfer stats (integer counters)
        self.transfers_completed: int = 0
        self.total_blocks_transferred: int = 0
        self._total_transfer_ms: float = 0.0

    def send_kv(
        self,
        metadata: KVTransferMetadata,
        kv_cache,  # PagedKVCache on prefill GPU
    ):
        """
        Send KV blocks from prefill GPU to decode GPU.

        Sends metadata header, then layer-by-layer KV block data.
        Uses non-blocking sends where possible for pipelining.
        """
        with self._send_lock:
            self._send_kv_locked(metadata, kv_cache)

    def _send_kv_locked(self, metadata, kv_cache):
        t_start = time.perf_counter()

        # 1. Pack and send metadata
        self._meta_buf[0] = metadata.request_id
        self._meta_buf[1] = metadata.seq_len
        self._meta_buf[2] = metadata.num_kv_blocks
        self._meta_buf[3] = metadata.num_layers
        self._meta_buf[4] = metadata.num_kv_heads
        self._meta_buf[5] = metadata.head_dim
        self._meta_buf[6] = metadata.block_size
        self._meta_buf[7] = metadata.first_token_id
        self._meta_buf[8] = metadata.max_new_tokens
        self._meta_buf[9] = metadata.eos_token_id

        dist.send(self._meta_buf, dst=self.decode_rank, group=self.group)

        # 2. Send block IDs (integer tensor)
        block_ids_tensor = torch.tensor(
            metadata.block_ids, dtype=torch.int32, device=self.device,
        )
        dist.send(block_ids_tensor, dst=self.decode_rank, group=self.group)

        # 3. Send prompt token IDs (for decode-side scheduler state)
        prompt_tensor = torch.tensor(
            metadata.prompt_token_ids, dtype=torch.int64, device=self.device,
        )
        # First send length, then data
        len_tensor = torch.tensor([len(metadata.prompt_token_ids)], dtype=torch.int64, device=self.device)
        dist.send(len_tensor, dst=self.decode_rank, group=self.group)
        dist.send(prompt_tensor, dst=self.decode_rank, group=self.group)

        # 4. Send KV data layer by layer, block by block
        #    Each block is (block_size, num_kv_heads, head_dim) — contiguous
        for layer_idx in range(metadata.num_layers):
            k_cache, v_cache = kv_cache.get_cache_tensors(layer_idx)
            for block_id in metadata.block_ids:
                # k_cache[block_id] shape: (block_size, num_kv_heads, head_dim)
                dist.send(k_cache[block_id].contiguous(), dst=self.decode_rank, group=self.group)
                dist.send(v_cache[block_id].contiguous(), dst=self.decode_rank, group=self.group)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.transfers_completed += 1
        self.total_blocks_transferred += metadata.num_kv_blocks
        self._total_transfer_ms += elapsed_ms

        logger.debug(
            "KV transfer sent: request=%d, blocks=%d, layers=%d, %.1f ms",
            metadata.request_id, metadata.num_kv_blocks, metadata.num_layers, elapsed_ms,
        )

    def recv_kv(
        self,
        kv_cache,  # PagedKVCache on decode GPU
    ) -> Optional[KVTransferMetadata]:
        """
        Receive KV blocks from prefill GPU into decode GPU's KV cache.

        Allocates blocks in decode-side cache, then receives data layer by layer.
        Returns metadata describing the transferred request, or None on error.
        """
        t_start = time.perf_counter()

        # 1. Receive metadata
        meta_buf = torch.zeros(self._meta_size, dtype=torch.int64, device=self.device)
        dist.recv(meta_buf, src=self.prefill_rank, group=self.group)

        request_id = int(meta_buf[0].item())
        seq_len = int(meta_buf[1].item())
        num_blocks = int(meta_buf[2].item())
        num_layers = int(meta_buf[3].item())
        num_kv_heads = int(meta_buf[4].item())
        head_dim = int(meta_buf[5].item())
        block_size = int(meta_buf[6].item())
        first_token_id = int(meta_buf[7].item())
        max_new_tokens = int(meta_buf[8].item())
        eos_token_id = int(meta_buf[9].item())

        # Sentinel: request_id == -1 means shutdown signal
        if request_id == -1:
            return None

        # 2. Receive block IDs (from prefill side — for reference only)
        src_block_ids = torch.zeros(num_blocks, dtype=torch.int32, device=self.device)
        dist.recv(src_block_ids, src=self.prefill_rank, group=self.group)

        # 3. Receive prompt token IDs
        prompt_len_tensor = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(prompt_len_tensor, src=self.prefill_rank, group=self.group)
        prompt_len = int(prompt_len_tensor[0].item())
        prompt_tensor = torch.zeros(prompt_len, dtype=torch.int64, device=self.device)
        dist.recv(prompt_tensor, src=self.prefill_rank, group=self.group)
        prompt_token_ids = prompt_tensor.cpu().tolist()

        # 4. Allocate blocks in decode-side KV cache
        #    Use incrementing counter to avoid modulo collisions
        if not hasattr(self, '_next_seq_id'):
            self._next_seq_id = 0
        seq_id = self._next_seq_id % kv_cache.max_seqs
        self._next_seq_id += 1
        decode_block_ids = kv_cache.allocate_blocks(seq_id, num_blocks)

        # 5. Receive and write KV data
        block_shape = (block_size, num_kv_heads, head_dim)
        recv_buf_k = torch.zeros(block_shape, dtype=kv_cache.kv_dtype, device=self.device)
        recv_buf_v = torch.zeros(block_shape, dtype=kv_cache.kv_dtype, device=self.device)

        for layer_idx in range(num_layers):
            k_cache, v_cache = kv_cache.get_cache_tensors(layer_idx)
            for i in range(num_blocks):
                dist.recv(recv_buf_k, src=self.prefill_rank, group=self.group)
                dist.recv(recv_buf_v, src=self.prefill_rank, group=self.group)
                dst_block = decode_block_ids[i]
                k_cache[dst_block].copy_(recv_buf_k)
                v_cache[dst_block].copy_(recv_buf_v)

        # Update decode-side seq_lens
        kv_cache.seq_lens[seq_id] = seq_len

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.transfers_completed += 1
        self.total_blocks_transferred += num_blocks
        self._total_transfer_ms += elapsed_ms

        logger.debug(
            "KV transfer recv: request=%d, blocks=%d, %.1f ms",
            request_id, num_blocks, elapsed_ms,
        )

        return KVTransferMetadata(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            first_token_id=first_token_id,
            seq_len=seq_len,
            num_kv_blocks=num_blocks,
            block_ids=decode_block_ids,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    def send_shutdown(self):
        """Send shutdown signal to decode worker (request_id = -1)."""
        self._meta_buf.zero_()
        self._meta_buf[0] = -1
        dist.send(self._meta_buf, dst=self.decode_rank, group=self.group)

    def get_stats(self) -> Dict[str, float]:
        """Transfer statistics."""
        avg_ms = (
            self._total_transfer_ms / max(self.transfers_completed, 1)
        )
        return {
            "transfers_completed": self.transfers_completed,
            "total_blocks_transferred": self.total_blocks_transferred,
            "total_transfer_ms": round(self._total_transfer_ms, 1),
            "avg_transfer_ms": round(avg_ms, 2),
        }


# =========================================================================
# PrefillWorker — runs on prefill GPU(s)
# =========================================================================

class PrefillWorker:
    """
    Runs prefill (prompt encoding) on dedicated GPU(s).

    Workflow per request:
      1. Receive prompt token IDs
      2. Run model.forward() for full prompt (compute-bound, high throughput)
      3. Sample first token from final logits
      4. Transfer KV cache blocks to decode worker via KVTransfer
      5. Free local KV blocks (prefill GPU recycles memory)

    Uses the existing I64Engine infrastructure for model.forward() and
    PagedKVCache for KV storage during prefill.
    """

    def __init__(
        self,
        model,
        kv_cache,              # PagedKVCache on prefill GPU
        kv_transfer: KVTransfer,
        device: str = "cuda:0",
        vocab_size: int = 100_000,
    ):
        self.model = model
        self.kv_cache = kv_cache
        self.kv_transfer = kv_transfer
        self.device = device
        self.vocab_size = vocab_size

        # Integer counters
        self.prefills_completed: int = 0
        self.total_prefill_tokens: int = 0
        self._total_prefill_ms: float = 0.0

        # Slot management for prefill (recycled after transfer)
        self._next_slot: int = 0

        logger.info("PrefillWorker initialized on %s", device)

    def _allocate_slot(self) -> int:
        """Allocate a KV cache slot for prefill. Wraps around max_seqs."""
        slot = self._next_slot % self.kv_cache.max_seqs
        self._next_slot += 1
        return slot

    def run_prefill(
        self,
        request_id: int,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        eos_token_id: int = 0,
        sampling_temperature: float = 0.0,
    ) -> int:
        """
        Run prefill for a single request, transfer KV to decode, return first token.

        Returns the first generated token ID (integer).
        """
        t_start = time.perf_counter()
        prompt_len = len(prompt_token_ids)

        # 1. Allocate KV slot
        seq_id = self._allocate_slot()
        num_blocks = (prompt_len + self.kv_cache.block_size - 1) // self.kv_cache.block_size
        block_ids = self.kv_cache.allocate_blocks(seq_id, num_blocks)

        # 2. Prepare input tensors
        token_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(prompt_len, dtype=torch.long, device=self.device)

        # 3. Run model forward (prefill — the compute-bound step)
        with torch.no_grad():
            logits = self.model(
                token_ids=token_ids,
                positions=positions,
                kv_cache=self.kv_cache,
                seq_ids=[seq_id],
                tokens_per_seq=[prompt_len],
            )

        # 4. Sample first token (greedy or temperature-based)
        #    logits shape: (prompt_len, vocab) or (1, vocab) — take last position
        if logits.dim() == 2 and logits.shape[0] > 1:
            last_logits = logits[-1:, :]  # (1, vocab)
        else:
            last_logits = logits

        if sampling_temperature <= 0.0:
            first_token = int(last_logits.argmax(dim=-1).item())
        else:
            probs = torch.softmax(last_logits / sampling_temperature, dim=-1)
            first_token = int(torch.multinomial(probs, 1).item())

        # 5. Get block IDs from block table (single GPU->CPU transfer)
        actual_blocks = self.kv_cache.block_table[seq_id, :num_blocks].tolist()

        # 6. Transfer KV to decode GPU
        config = self.model.config
        meta = KVTransferMetadata(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            first_token_id=first_token,
            seq_len=prompt_len,
            num_kv_blocks=num_blocks,
            block_ids=actual_blocks,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            block_size=self.kv_cache.block_size,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            sampling_temperature=sampling_temperature,
        )
        self.kv_transfer.send_kv(meta, self.kv_cache)

        # 7. Free prefill-side KV blocks (decode now has the data)
        self.kv_cache.free_sequence(seq_id)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.prefills_completed += 1
        self.total_prefill_tokens += prompt_len
        self._total_prefill_ms += elapsed_ms

        logger.debug(
            "Prefill complete: request=%d, prompt_len=%d, first_token=%d, %.1f ms",
            request_id, prompt_len, first_token, elapsed_ms,
        )

        return first_token

    def shutdown(self):
        """Send shutdown signal to decode worker."""
        self.kv_transfer.send_shutdown()
        logger.info(
            "PrefillWorker shutdown: %d prefills, %d tokens, %.1f ms avg",
            self.prefills_completed,
            self.total_prefill_tokens,
            self._total_prefill_ms / max(self.prefills_completed, 1),
        )

    def get_stats(self) -> dict:
        """Prefill worker statistics."""
        return {
            "prefills_completed": self.prefills_completed,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_prefill_ms": round(self._total_prefill_ms, 1),
            "avg_prefill_ms": round(
                self._total_prefill_ms / max(self.prefills_completed, 1), 2
            ),
            "avg_tokens_per_prefill": round(
                self.total_prefill_tokens / max(self.prefills_completed, 1), 1
            ),
            "kv_transfer": self.kv_transfer.get_stats(),
        }


# =========================================================================
# DecodeWorker — runs on decode GPU(s)
# =========================================================================

class DecodeWorker:
    """
    Runs autoregressive decode on dedicated GPU(s).

    Workflow:
      1. Receive KV cache blocks + first token from PrefillWorker
      2. Add request to local scheduler for continuous batching
      3. Run model.decode_step() or model.forward() token by token
      4. Return completed sequences

    Uses I64Engine internally for scheduling and batched decode.
    Memory-bound workload: optimized for low latency per token.
    """

    def __init__(
        self,
        model,
        kv_cache,              # PagedKVCache on decode GPU
        kv_transfer: KVTransfer,
        device: str = "cuda:1",
        vocab_size: int = 100_000,
        max_batch_size: int = 32,
    ):
        from vllm_i64.engine.i64_scheduler import I64Scheduler, I64Request, RequestStatus
        from vllm_i64.core.sampling import SamplingParams, sample_batch

        self.model = model
        self.kv_cache = kv_cache
        self.kv_transfer = kv_transfer
        self.device = device
        self.vocab_size = vocab_size

        # Scheduler for batching decode requests
        config = model.config
        max_kv_blocks = kv_cache.num_blocks
        self.scheduler = I64Scheduler(
            max_batch_size=max_batch_size,
            max_seq_len=getattr(config, 'max_position_embeddings', 2048),
            num_experts=getattr(config, 'num_experts', 1),
            max_kv_blocks=max_kv_blocks,
        )

        # Default sampling (greedy)
        self.sampling_params = SamplingParams(temperature=0.0)

        # Track request -> KV slot mapping
        self._request_to_slot: Dict[int, int] = {}
        self._slot_pool = list(range(kv_cache.max_seqs))
        self._request_meta: Dict[int, KVTransferMetadata] = {}

        # Completed results: request_id -> output_tokens
        self._completed: Dict[int, List[int]] = {}

        # Running flag
        self._running = False

        # Integer counters
        self.decode_steps: int = 0
        self.total_decode_tokens: int = 0

        # Persistent buffers
        _buf_device = device
        _max_tokens = max_batch_size
        self._buf_token_ids = torch.zeros(_max_tokens, dtype=torch.long, device=_buf_device)
        self._buf_positions = torch.zeros(_max_tokens, dtype=torch.long, device=_buf_device)

        logger.info("DecodeWorker initialized on %s", device)

    def _allocate_slot(self, request_id: int) -> int:
        """Allocate a KV slot for a decode request."""
        if not self._slot_pool:
            logger.error("No free KV slots for decode request %d", request_id)
            return -1
        slot = self._slot_pool.pop(0)
        self._request_to_slot[request_id] = slot
        return slot

    def _free_slot(self, request_id: int):
        """Free a KV slot when a request completes."""
        slot = self._request_to_slot.pop(request_id, None)
        if slot is not None:
            self.kv_cache.free_sequence(slot)
            self._slot_pool.append(slot)

    def receive_and_add_request(self) -> Optional[int]:
        """
        Receive a single KV transfer from prefill worker.
        Returns request_id, or None on shutdown signal.
        """
        meta = self.kv_transfer.recv_kv(self.kv_cache)
        if meta is None:
            return None  # Shutdown

        request_id = meta.request_id
        self._request_meta[request_id] = meta

        # The KV cache already has blocks allocated by recv_kv.
        # Allocate a proper KV slot (avoid modulo collisions)
        seq_id = self._allocate_slot(request_id)
        if seq_id < 0:
            logger.error("Cannot accept transferred request %d: no free KV slots", request_id)
            return None

        # Add to scheduler as running (prefill already done)
        from vllm_i64.engine.i64_scheduler import I64Request, RequestStatus
        req = I64Request(
            request_id=request_id,
            prompt_token_ids=np.array(meta.prompt_token_ids, dtype=np.int64),
            max_new_tokens=meta.max_new_tokens,
            status=RequestStatus.RUNNING,
            eos_token_id=meta.eos_token_id,
        )
        # Mark prefill as complete
        req.prefill_progress = len(meta.prompt_token_ids)
        req.seq_pos = meta.seq_len
        # Add the first token from prefill sampling
        req.output_token_ids = [meta.first_token_id]
        req.seq_pos = meta.seq_len + 1

        # Assign KV blocks from the transfer
        req.kv_block_ids = list(meta.block_ids)

        # Add directly to running list (skip pending queue — already prefilled)
        self.scheduler.running.append(req)

        logger.debug(
            "Decode request added: id=%d, prompt_len=%d, first_token=%d",
            request_id, len(meta.prompt_token_ids), meta.first_token_id,
        )

        return request_id

    def decode_step(self) -> Dict[int, int]:
        """
        Run one decode step for all active requests.

        Returns {request_id: new_token_id} for requests that generated a token.
        Finished requests are moved to self._completed.
        """
        from vllm_i64.engine.i64_scheduler import RequestStatus
        from vllm_i64.core.sampling import SamplingParams, sample_batch

        batch = self.scheduler.schedule()
        if batch is None:
            return {}

        # Prepare tensors
        n = batch.token_ids.shape[0]
        token_ids = torch.from_numpy(batch.token_ids).to(self.device)
        positions = torch.from_numpy(batch.positions).to(self.device)

        # Build seq_ids mapping
        seq_ids = []
        tokens_per_seq = []
        for i, rid in enumerate(batch.request_ids):
            slot = self._request_to_slot.get(int(rid))
            if slot is not None:
                seq_ids.append(slot)
                tokens_per_seq.append(int(batch.tokens_per_request[i]))

        # Model forward (decode — memory-bound, low latency)
        with torch.no_grad():
            has_decode_step = hasattr(self.model, 'decode_step')
            if has_decode_step and not batch.is_prefill.any():
                # Use decode_step for single-token decode (faster path)
                seq_ids_tensor = torch.tensor(seq_ids, dtype=torch.long, device=self.device)
                logits = self.model.decode_step(
                    token_ids=token_ids,
                    positions=positions,
                    kv_cache=self.kv_cache,
                    seq_ids_tensor=seq_ids_tensor,
                )
            else:
                logits = self.model(
                    token_ids=token_ids,
                    positions=positions,
                    kv_cache=self.kv_cache,
                    seq_ids=seq_ids,
                    tokens_per_seq=tokens_per_seq,
                )

        # Extract last-token logits per request
        if batch.tokens_per_request is not None and logits.shape[0] != batch.num_requests:
            tpr_list = batch.tokens_per_request.tolist()
            last_indices = []
            offset = 0
            for tpr in tpr_list:
                idx = offset + tpr - 1
                if idx >= logits.shape[0]:
                    logger.error("Batch logits index %d >= shape %d, clamping", idx, logits.shape[0])
                    idx = logits.shape[0] - 1
                last_indices.append(idx)
                offset += tpr
            logits = logits[last_indices]

        # Sample (greedy)
        new_token_ids = sample_batch(logits, self.sampling_params)

        # Update scheduler
        result = {}
        for i, rid in enumerate(batch.request_ids.tolist()):
            result[rid] = new_token_ids[i]

        self.scheduler.update_after_step(result)

        # Collect finished requests
        still_running = []
        for req in self.scheduler.running:
            if req.is_finished:
                req.status = RequestStatus.FINISHED
                self._completed[req.request_id] = list(req.output_token_ids)
                self._free_slot(req.request_id)
                self._request_meta.pop(req.request_id, None)
                self.scheduler.finished.append(req)
            else:
                still_running.append(req)
        self.scheduler.running = still_running

        self.decode_steps += 1
        self.total_decode_tokens += len(result)

        return result

    def get_completed(self) -> Dict[int, List[int]]:
        """Pop completed request results. Returns {request_id: output_tokens}."""
        completed = dict(self._completed)
        self._completed.clear()
        return completed

    def has_active_requests(self) -> bool:
        """Check if there are any active decode requests."""
        return len(self.scheduler.running) > 0

    def get_stats(self) -> dict:
        """Decode worker statistics."""
        return {
            "decode_steps": self.decode_steps,
            "total_decode_tokens": self.total_decode_tokens,
            "active_requests": len(self.scheduler.running),
            "completed_pending": len(self._completed),
            "kv_transfer": self.kv_transfer.get_stats(),
        }


# =========================================================================
# DisaggregatedCoordinator — routes requests between prefill and decode
# =========================================================================

class DisaggregatedCoordinator:
    """
    Routes inference requests between prefill and decode workers.

    Design:
      - New requests go to PrefillWorker (compute-bound GPU)
      - After prefill + KV transfer, decode continues on DecodeWorker (memory-bound GPU)
      - Single-GPU fallback: runs both on same device (no disaggregation)

    Thread model:
      - Prefill runs on a background thread (blocks on model.forward)
      - Decode runs on the main engine loop (continuous batching)
      - KV transfer is synchronous between the two

    For the 2-GPU case:
      - GPU 0: prefill worker (rank 0)
      - GPU 1: decode worker (rank 1)
    """

    def __init__(
        self,
        prefill_worker: Optional[PrefillWorker],
        decode_worker: Optional[DecodeWorker],
        single_gpu_engine=None,
    ):
        self.prefill_worker = prefill_worker
        self.decode_worker = decode_worker
        self.single_gpu_engine = single_gpu_engine

        # Disaggregated mode if both workers present
        self.disaggregated = (prefill_worker is not None and decode_worker is not None)

        # Request tracking
        self._requests: Dict[int, DisaggRequest] = {}
        self._next_request_id: int = 0
        self._lock = threading.Lock()

        # Prefill queue (thread-safe)
        self._prefill_queue: queue.Queue = queue.Queue()

        # Prefill thread
        self._prefill_thread: Optional[threading.Thread] = None
        self._shutdown = False

        # Result callbacks
        self._result_callbacks: Dict[int, Callable] = {}

        if self.disaggregated:
            self._start_prefill_thread()
            logger.info("DisaggregatedCoordinator: disaggregated mode (prefill + decode)")
        else:
            logger.info("DisaggregatedCoordinator: single-GPU fallback (no disaggregation)")

    def _start_prefill_thread(self):
        """Start background thread for prefill processing."""
        self._prefill_thread = threading.Thread(
            target=self._prefill_loop,
            daemon=True,
            name="disagg-prefill",
        )
        self._prefill_thread.start()

    def _prefill_loop(self):
        """
        Background thread: process prefill requests from queue.
        Each prefill completes and triggers KV transfer to decode worker.
        """
        logger.info("Prefill thread started")
        while not self._shutdown:
            try:
                req = self._prefill_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if req is None:
                break  # Shutdown sentinel

            request_id = req.request_id
            try:
                with self._lock:
                    req.status = DisaggStatus.PREFILLING

                first_token = self.prefill_worker.run_prefill(
                    request_id=request_id,
                    prompt_token_ids=req.prompt_token_ids,
                    max_new_tokens=req.max_new_tokens,
                    eos_token_id=req.eos_token_id,
                    sampling_temperature=req.sampling_temperature,
                )

                with self._lock:
                    req.first_token_id = first_token
                    req.status = DisaggStatus.TRANSFERRING
                    # Decode worker receives on its own recv thread

            except Exception as e:
                logger.error("Prefill failed for request %d: %s", request_id, e)
                with self._lock:
                    req.status = DisaggStatus.FINISHED
                    req.output_tokens = []

        logger.info("Prefill thread exiting")

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 256,
        eos_token_id: int = 0,
        sampling_temperature: float = 0.0,
        callback: Optional[Callable] = None,
    ) -> int:
        """
        Add a new inference request.

        In disaggregated mode: routes to prefill worker.
        In single-GPU mode: passes through to underlying engine.

        Returns integer request_id.
        """
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1

        if not self.disaggregated:
            # Single-GPU fallback
            if self.single_gpu_engine is not None:
                from vllm_i64.core.sampling import SamplingParams
                sp = SamplingParams(temperature=sampling_temperature)
                return self.single_gpu_engine.add_request(
                    prompt_token_ids,
                    max_new_tokens=max_new_tokens,
                    sampling_params=sp,
                )
            return request_id

        # Create tracked request
        dreq = DisaggRequest(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            sampling_temperature=sampling_temperature,
            arrival_time=time.perf_counter(),
        )

        with self._lock:
            self._requests[request_id] = dreq
            if callback is not None:
                self._result_callbacks[request_id] = callback

        # Queue for prefill
        self._prefill_queue.put(dreq)

        return request_id

    def step(self) -> Dict[int, int]:
        """
        Run one coordinator step.

        In disaggregated mode:
          - Prefill runs on background thread
          - This step runs decode on all active decode requests

        In single-GPU mode: delegates to engine.step()
        """
        if not self.disaggregated:
            if self.single_gpu_engine is not None:
                return self.single_gpu_engine.step()
            return {}

        # Run one decode step
        result = self.decode_worker.decode_step()

        # Collect completed requests
        completed = self.decode_worker.get_completed()
        for rid, output_tokens in completed.items():
            with self._lock:
                dreq = self._requests.get(rid)
                if dreq is not None:
                    dreq.output_tokens = output_tokens
                    dreq.status = DisaggStatus.FINISHED
                    dreq.finish_time = time.perf_counter()

                cb = self._result_callbacks.pop(rid, None)
            if cb is not None:
                try:
                    cb(rid, output_tokens)
                except Exception as e:
                    logger.error("Result callback error for request %d: %s", rid, e)

        return result

    def has_work(self) -> bool:
        """Check if there's pending or active work."""
        if not self.disaggregated:
            if self.single_gpu_engine is not None:
                return (
                    len(self.single_gpu_engine.scheduler.running) > 0
                    or len(self.single_gpu_engine.scheduler.pending) > 0
                )
            return False

        return (
            not self._prefill_queue.empty()
            or self.decode_worker.has_active_requests()
            or any(r.status < 4 for r in self._requests.values())
        )

    def get_completed(self) -> Dict[int, List[int]]:
        """Get all completed request results."""
        with self._lock:
            completed = {}
            for rid, dreq in list(self._requests.items()):
                if dreq.status == 4:
                    completed[rid] = dreq.output_tokens
            # Don't remove — caller can clear via pop_completed
            return completed

    def pop_completed(self, request_id: int) -> Optional[List[int]]:
        """Pop a completed request's output tokens. Returns None if not done."""
        with self._lock:
            dreq = self._requests.get(request_id)
            if dreq is not None and dreq.status == 4:
                del self._requests[request_id]
                return dreq.output_tokens
        return None

    def shutdown(self):
        """Shutdown coordinator and workers."""
        self._shutdown = True

        # Signal prefill thread to stop
        self._prefill_queue.put(None)

        if self._prefill_thread is not None:
            self._prefill_thread.join(timeout=5.0)

        if self.prefill_worker is not None:
            self.prefill_worker.shutdown()

        logger.info("DisaggregatedCoordinator shutdown complete")

    def get_stats(self) -> dict:
        """Coordinator statistics."""
        stats = {
            "disaggregated": self.disaggregated,
            "total_requests": self._next_request_id,
            "active_requests": sum(1 for r in self._requests.values() if r.status < 4),
            "completed_requests": sum(1 for r in self._requests.values() if r.status == 4),
        }
        if self.prefill_worker is not None:
            stats["prefill"] = self.prefill_worker.get_stats()
        if self.decode_worker is not None:
            stats["decode"] = self.decode_worker.get_stats()
        return stats


# =========================================================================
# Factory — set up disaggregated inference from CLI/engine config
# =========================================================================

def setup_disaggregated(
    model,
    tp_size: int,
    device_base: str = "cuda",
    vocab_size: int = 100_000,
    max_batch_size: int = 32,
    kv_cache_dtype: Optional[str] = None,
) -> DisaggregatedCoordinator:
    """
    Set up disaggregated prefill/decode from a loaded model.

    For tp_size >= 2:
      - GPU 0: prefill
      - GPU 1: decode
      - Uses torch.distributed process group for KV transfer

    For tp_size == 1 (single GPU):
      - No disaggregation, returns coordinator wrapping a standard I64Engine

    Args:
        model: loaded model (already on device)
        tp_size: total number of GPUs available
        device_base: base device string (e.g. "cuda")
        vocab_size: model vocabulary size
        max_batch_size: max concurrent decode requests
        kv_cache_dtype: optional KV cache quantization ("fp8", "fp8_e5m2")

    Returns:
        DisaggregatedCoordinator ready for inference
    """
    if tp_size < 2 or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        # Single-GPU fallback
        logger.info("Disaggregated: single-GPU fallback (tp_size=%d, GPUs=%d)",
                     tp_size, torch.cuda.device_count() if torch.cuda.is_available() else 0)
        from vllm_i64.engine.i64_engine import I64Engine
        engine = I64Engine(
            model=model,
            num_experts=getattr(model.config, 'num_experts', 1),
            vocab_size=vocab_size,
            max_batch_size=max_batch_size,
            device=device_base if torch.cuda.is_available() else "cpu",
            kv_cache_dtype=kv_cache_dtype,
        )
        return DisaggregatedCoordinator(
            prefill_worker=None,
            decode_worker=None,
            single_gpu_engine=engine,
        )

    # Multi-GPU disaggregated setup
    prefill_device = f"{device_base}:0"
    decode_device = f"{device_base}:1"

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    block_size = 16
    dtype = next(model.parameters()).dtype

    # Determine KV cache block count per GPU
    # Prefill needs fewer blocks (recycled after transfer)
    # Decode needs more blocks (holds all active sequences)
    prefill_kv_blocks = max(64, max_batch_size * 4)
    decode_kv_blocks = max(256, max_batch_size * 8)

    # Initialize process group for KV transfer if not already initialized
    if not dist.is_initialized():
        logger.info("Initializing torch.distributed for disaggregated KV transfer")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    prefill_rank = 0
    decode_rank = 1

    # Create KV caches on respective GPUs
    from vllm_i64.core.kv_cache import PagedKVCache

    prefill_kv_cache = PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=prefill_kv_blocks,
        max_seqs=max_batch_size,
        dtype=dtype,
        device=prefill_device,
        kv_cache_dtype=kv_cache_dtype,
    )

    decode_kv_cache = PagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=decode_kv_blocks,
        max_seqs=max_batch_size,
        dtype=dtype,
        device=decode_device,
        kv_cache_dtype=kv_cache_dtype,
    )

    # Create KV transfer channels
    prefill_transfer = KVTransfer(
        prefill_rank=prefill_rank,
        decode_rank=decode_rank,
        device=prefill_device,
    )
    decode_transfer = KVTransfer(
        prefill_rank=prefill_rank,
        decode_rank=decode_rank,
        device=decode_device,
    )

    # Create workers
    # NOTE: in a real multi-process setup, each worker runs in its own process.
    # For the 2-GPU case with a single process, we place model copies on each GPU.
    # The model must support being moved/copied to a different device.
    prefill_worker = PrefillWorker(
        model=model,
        kv_cache=prefill_kv_cache,
        kv_transfer=prefill_transfer,
        device=prefill_device,
        vocab_size=vocab_size,
    )

    # For decode, we need the model on decode device
    # Use a separate model copy if model is on a different device
    decode_model = model
    if str(next(model.parameters()).device) != decode_device:
        import copy
        logger.info("Copying model to decode device %s", decode_device)
        decode_model = copy.deepcopy(model).to(decode_device)

    decode_worker = DecodeWorker(
        model=decode_model,
        kv_cache=decode_kv_cache,
        kv_transfer=decode_transfer,
        device=decode_device,
        vocab_size=vocab_size,
        max_batch_size=max_batch_size,
    )

    return DisaggregatedCoordinator(
        prefill_worker=prefill_worker,
        decode_worker=decode_worker,
    )


def launch_disaggregated(tp_size: int, args: list) -> int:
    """
    Launch disaggregated prefill/decode via torchrun.

    Splits tp_size GPUs:
      - Rank 0: prefill worker
      - Rank 1: decode worker
      - Remaining ranks (if any): additional TP within each group

    Args:
        tp_size: total number of GPUs
        args: CLI arguments forwarded to workers

    Returns:
        process return code
    """
    import subprocess
    import sys
    from vllm_i64.parallel.launcher import _find_free_port

    nproc = tp_size
    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_port", _find_free_port(),
        "-m", "vllm_i64.parallel.worker",
    ] + args

    env = os.environ.copy()
    env["VLLM_I64_TP_SIZE"] = str(tp_size)
    env["VLLM_I64_DISAGGREGATED"] = "1"

    logger.info(
        "Launching disaggregated: %d GPUs (1 prefill + 1 decode)",
        tp_size,
    )
    logger.info("  cmd: %s", " ".join(cmd))

    proc = subprocess.run(cmd, env=env)
    return proc.returncode
