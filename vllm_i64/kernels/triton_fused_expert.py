"""
vllm-i64 :: Triton Fused Expert Kernel

Fused Gate+Up → SwiGLU → Down kernel for token-routed MLP.
Eliminates intermediate memory reads/writes vs. the 3-op PyTorch path.

Two entry points:
  - triton_expert_forward():     FP16/BF16/FP32
  - triton_expert_forward_int8(): INT8 dequant inline

Fallback: if Triton not available, returns None and caller uses PyTorch.

INL - 2025
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """Check if Triton is available and we're on CUDA."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_swiglu_kernel(
        # Input
        x_ptr,
        # Expert weights
        gate_up_ptr,
        down_ptr,
        # Output
        output_ptr,
        # Dimensions
        N,      # number of tokens for this expert
        H,      # hidden_size
        I,      # intermediate_per_tp
        # Strides for x (N, H)
        stride_x_n, stride_x_h,
        # Strides for gate_up (H, 2*I)
        stride_gu_h, stride_gu_i,
        # Strides for down (I, H)
        stride_d_i, stride_d_h,
        # Strides for output (N, H)
        stride_o_n, stride_o_h,
        # Block sizes
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        """
        Fused SwiGLU expert kernel.

        For a chunk of tokens assigned to one expert:
          1. gate_up = x @ gate_up_weight        → (N, 2*I)
          2. gate, up = split(gate_up)
          3. inter = silu(gate) * up              → (N, I)
          4. output = inter @ down_weight         → (N, H)

        Fuses steps 1-4 into a single kernel, avoiding materializing
        the (N, 2*I) and (N, I) intermediate tensors in global memory.
        """
        # Program ID: each program handles one tile of (tokens, output_hidden)
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)

        # Token indices for this tile
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # Output hidden indices for this tile
        h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        # Accumulator for output (BLOCK_N, BLOCK_H) — float32 for precision
        acc = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)

        # Loop over intermediate dimension in tiles
        for i_start in range(0, I, BLOCK_I):
            i_offsets = i_start + tl.arange(0, BLOCK_I)
            i_mask = i_offsets < I

            # ---- Step 1: Compute gate and up values ----
            # gate = x @ gate_up[:, :I]  → need (BLOCK_N, BLOCK_I)
            # up   = x @ gate_up[:, I:]  → need (BLOCK_N, BLOCK_I)
            gate_acc = tl.zeros((BLOCK_N, BLOCK_I), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_N, BLOCK_I), dtype=tl.float32)

            for h_start in range(0, H, BLOCK_H):
                hk_offsets = h_start + tl.arange(0, BLOCK_H)
                hk_mask = hk_offsets < H

                # Load x tile: (BLOCK_N, BLOCK_H)
                x_ptrs = x_ptr + n_offsets[:, None] * stride_x_n + hk_offsets[None, :] * stride_x_h
                x_tile = tl.load(x_ptrs, mask=n_mask[:, None] & hk_mask[None, :], other=0.0)

                # Load gate weights: gate_up[h, i] for i in [0, I)
                gu_gate_ptrs = gate_up_ptr + hk_offsets[:, None] * stride_gu_h + i_offsets[None, :] * stride_gu_i
                gu_gate_tile = tl.load(gu_gate_ptrs, mask=hk_mask[:, None] & i_mask[None, :], other=0.0)

                # Load up weights: gate_up[h, i+I] for i in [0, I)
                gu_up_ptrs = gate_up_ptr + hk_offsets[:, None] * stride_gu_h + (i_offsets[None, :] + I) * stride_gu_i
                gu_up_tile = tl.load(gu_up_ptrs, mask=hk_mask[:, None] & i_mask[None, :], other=0.0)

                # Accumulate: (BLOCK_N, BLOCK_H) @ (BLOCK_H, BLOCK_I) → (BLOCK_N, BLOCK_I)
                gate_acc += tl.dot(x_tile, gu_gate_tile)
                up_acc += tl.dot(x_tile, gu_up_tile)

            # ---- Step 2: SwiGLU activation ----
            # silu(gate) * up
            gate_val = gate_acc.to(tl.float32)
            sigmoid_gate = tl.sigmoid(gate_val)
            silu_gate = gate_val * sigmoid_gate
            inter = silu_gate * up_acc  # (BLOCK_N, BLOCK_I)

            # ---- Step 3: down projection accumulation ----
            # acc += inter @ down[i_offsets, h_offsets]
            # down is (I, H) → load (BLOCK_I, BLOCK_H) tile
            d_ptrs = down_ptr + i_offsets[:, None] * stride_d_i + h_offsets[None, :] * stride_d_h
            d_tile = tl.load(d_ptrs, mask=i_mask[:, None] & h_mask[None, :], other=0.0)

            # (BLOCK_N, BLOCK_I) @ (BLOCK_I, BLOCK_H) → (BLOCK_N, BLOCK_H)
            acc += tl.dot(inter.to(d_tile.dtype), d_tile)

        # ---- Store output ----
        o_ptrs = output_ptr + n_offsets[:, None] * stride_o_n + h_offsets[None, :] * stride_o_h
        tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=n_mask[:, None] & h_mask[None, :])


def triton_expert_forward(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """
    Fused SwiGLU expert forward via Triton.

    Args:
        x: (N, H) input tokens for this expert
        gate_up: (H, 2*I) gate+up weight matrix
        down: (I, H) down projection weight matrix
        num_tokens: actual number of tokens (N)

    Returns:
        output: (N, H)
    """
    if not is_triton_available():
        return None

    H = x.shape[1]
    I = down.shape[0]  # intermediate_per_tp
    N = num_tokens

    output = torch.empty(N, H, device=x.device, dtype=x.dtype)

    # Block sizes — tuned for typical MoE dimensions
    BLOCK_N = min(32, triton.next_power_of_2(N))
    BLOCK_H = min(64, triton.next_power_of_2(H))
    BLOCK_I = min(64, triton.next_power_of_2(I))

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

    _fused_swiglu_kernel[grid](
        x,
        gate_up, down,
        output,
        N, H, I,
        x.stride(0), x.stride(1),
        gate_up.stride(0), gate_up.stride(1),
        down.stride(0), down.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        BLOCK_I=BLOCK_I,
    )

    return output


def triton_expert_forward_int8(
    x: torch.Tensor,
    gate_up_int8: torch.Tensor,
    gate_up_scale: torch.Tensor,
    down_int8: torch.Tensor,
    down_scale: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """
    Fused SwiGLU expert forward with INT8 dequant via Triton.

    Dequantizes inline: weight_fp = int8_weight * scale.
    For now, dequantize per-expert in PyTorch then call the Triton kernel.
    Full inline dequant kernel is future work.

    Args:
        x: (N, H) input tokens
        gate_up_int8: (H, 2*I) int8
        gate_up_scale: (H,) float scale
        down_int8: (I, H) int8
        down_scale: (I,) float scale
        num_tokens: number of tokens

    Returns:
        output: (N, H)
    """
    if not is_triton_available():
        return None

    # Dequantize to compute dtype
    gate_up_fp = gate_up_int8.float() * gate_up_scale.unsqueeze(-1)
    gate_up_fp = gate_up_fp.to(x.dtype)

    down_fp = down_int8.float() * down_scale.unsqueeze(-1)
    down_fp = down_fp.to(x.dtype)

    return triton_expert_forward(x, gate_up_fp, down_fp, num_tokens)
