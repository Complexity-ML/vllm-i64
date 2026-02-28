"""
vllm-i64 :: CUDA Graph Capture

Capture the decode loop as a CUDA graph to eliminate kernel launch overhead.

Token-routed models are ideal for CUDA graphs:
  - Routing is deterministic (no dynamic branching per token)
  - Expert dispatch is fixed topology
  - Decode step has fixed shape (1 token per sequence)

Multi-batch: captures graphs for common batch sizes (1, 2, 4, 8, ...)
and selects the smallest captured graph >= actual batch size.

INL - 2025
"""

import torch
from typing import Optional, Callable, Dict, Set


class CUDAGraphRunner:
    """
    Captures and replays decode steps as CUDA graphs.

    Supports multiple batch sizes: captures graphs for common sizes
    (1, 2, 4, 8, 16, 32, 64) and selects the best fit at runtime.

    Usage:
        runner = CUDAGraphRunner(model_forward, max_batch=64)
        runner.capture_common_sizes()
        output = runner.run(real_input)
    """

    def __init__(
        self,
        forward_fn: Callable,
        max_batch_size: int = 64,
        device: str = "cuda",
    ):
        self.forward_fn = forward_fn
        self.max_batch_size = max_batch_size
        self.device = device

        # Multiple graphs keyed by batch size
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[int, dict] = {}
        self.static_outputs: Dict[int, torch.Tensor] = {}
        self._captured_sizes: Set[int] = set()

    def capture(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        expert_ids: torch.Tensor,
    ):
        """
        Capture the forward pass as a CUDA graph for the given batch size.

        Must be called with representative inputs (correct shapes).
        """
        assert token_ids.device.type == "cuda", "CUDA graph requires CUDA tensors"
        bs = token_ids.shape[0]

        # Static input buffers (graph replays with these)
        static_in = {
            "token_ids": token_ids.clone(),
            "positions": positions.clone(),
            "expert_ids": expert_ids.clone(),
        }

        # Warmup (CUDA graphs require warm streams)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = self.forward_fn(
                    static_in["token_ids"],
                    static_in["positions"],
                    static_in["expert_ids"],
                )
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.forward_fn(
                static_in["token_ids"],
                static_in["positions"],
                static_in["expert_ids"],
            )

        self.graphs[bs] = graph
        self.static_inputs[bs] = static_in
        self.static_outputs[bs] = out
        self._captured_sizes.add(bs)

    def capture_common_sizes(self):
        """Capture graphs for common batch sizes up to max_batch_size."""
        sizes = [bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= self.max_batch_size]
        for bs in sizes:
            token_ids = torch.zeros(bs, dtype=torch.int64, device=self.device)
            positions = torch.zeros(bs, dtype=torch.int32, device=self.device)
            expert_ids = torch.zeros(bs, dtype=torch.int32, device=self.device)
            self.capture(token_ids, positions, expert_ids)

    def _find_best_size(self, batch_size: int) -> Optional[int]:
        """Find the smallest captured size >= batch_size."""
        candidates = [s for s in self._captured_sizes if s >= batch_size]
        return min(candidates) if candidates else None

    def run(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run with the best matching captured graph.

        Pads to the nearest captured size, replays graph, slices output.
        Falls back to direct forward if no suitable graph exists.
        """
        batch_size = token_ids.shape[0]
        graph_size = self._find_best_size(batch_size)

        if graph_size is None:
            # No graph for this size — run directly
            return self.forward_fn(token_ids, positions, expert_ids)

        static_in = self.static_inputs[graph_size]

        # Zero-fill then copy actual data
        static_in["token_ids"].zero_()
        static_in["positions"].zero_()
        static_in["expert_ids"].zero_()

        static_in["token_ids"][:batch_size].copy_(token_ids)
        static_in["positions"][:batch_size].copy_(positions)
        static_in["expert_ids"][:batch_size].copy_(expert_ids)

        # Replay captured graph
        self.graphs[graph_size].replay()

        return self.static_outputs[graph_size][:batch_size]

    @property
    def is_captured(self) -> bool:
        return len(self._captured_sizes) > 0
