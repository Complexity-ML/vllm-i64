"""
vllm-i64 :: CUDA Graph Capture

Capture the decode loop as a CUDA graph to eliminate kernel launch overhead.

Token-routed models are ideal for CUDA graphs:
  - Routing is deterministic (no dynamic branching per token)
  - Expert dispatch is fixed topology
  - Decode step has fixed shape (1 token per sequence)

INL - 2025
"""

import torch
from typing import Optional, Callable


class CUDAGraphRunner:
    """
    Captures and replays a decode step as a CUDA graph.

    Usage:
        runner = CUDAGraphRunner(model_forward, max_batch=64)
        runner.capture(sample_input)
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
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_inputs = {}
        self.static_output = None
        self._captured = False

    def capture(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        expert_ids: torch.Tensor,
    ):
        """
        Capture the forward pass as a CUDA graph.

        Must be called with representative inputs (correct shapes).
        """
        assert token_ids.device.type == "cuda", "CUDA graph requires CUDA tensors"

        # Static input buffers (graph replays with these)
        self.static_inputs = {
            "token_ids": token_ids.clone(),
            "positions": positions.clone(),
            "expert_ids": expert_ids.clone(),
        }

        # Warmup (CUDA graphs require warm streams)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.static_output = self.forward_fn(
                    self.static_inputs["token_ids"],
                    self.static_inputs["positions"],
                    self.static_inputs["expert_ids"],
                )
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self.forward_fn(
                self.static_inputs["token_ids"],
                self.static_inputs["positions"],
                self.static_inputs["expert_ids"],
            )

        self._captured = True

    def run(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run captured graph with new inputs.

        Copies new data into static buffers, replays graph, returns output.
        """
        if not self._captured:
            # Fallback: run without graph
            return self.forward_fn(token_ids, positions, expert_ids)

        # Copy new data into static buffers
        batch_size = token_ids.shape[0]
        self.static_inputs["token_ids"][:batch_size].copy_(token_ids)
        self.static_inputs["positions"][:batch_size].copy_(positions)
        self.static_inputs["expert_ids"][:batch_size].copy_(expert_ids)

        # Replay
        self.graph.replay()

        return self.static_output[:batch_size]

    @property
    def is_captured(self) -> bool:
        return self._captured
