"""
vllm-i64 :: Triton Fused Expert Kernel

Fused Gate+Up → SwiGLU → Down kernel for token-routed MLP.
Eliminates intermediate memory reads/writes vs. the 3-op PyTorch path.

Three entry points:
  - triton_expert_forward():      FP16/BF16/FP32
  - triton_expert_forward_int8(): INT8 inline dequant (loads int8+scale, no FP materialization)

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


    # =================================================================
    # INT8 inline dequant kernel
    #
    # Loads INT8 weights (1 byte each) + per-channel float32 scale,
    # dequantizes in-register: w_fp = int8_val * scale[row].
    # Saves 2x memory bandwidth vs loading pre-dequantized FP16
    # and avoids materializing the full FP weight tensor.
    # =================================================================

    @triton.jit
    def _fused_swiglu_int8_kernel(
        # Input activations (fp16/bf16/fp32)
        x_ptr,
        # INT8 weights + scales
        gate_up_int8_ptr,   # (H, 2*I) int8
        gate_up_scale_ptr,  # (H,) float32 — per-channel scale
        down_int8_ptr,      # (I, H) int8
        down_scale_ptr,     # (I,) float32 — per-channel scale
        # Output
        output_ptr,
        # Dimensions
        N, H, I,
        # Strides for x (N, H)
        stride_x_n, stride_x_h,
        # Strides for gate_up_int8 (H, 2*I)
        stride_gu_h, stride_gu_i,
        # Strides for down_int8 (I, H)
        stride_d_i, stride_d_h,
        # Strides for output (N, H)
        stride_o_n, stride_o_h,
        # Block sizes
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        """
        Fused SwiGLU with INT8 inline dequantization.

        Same computation as _fused_swiglu_kernel but:
        - Loads INT8 weight bytes (1B each vs 2B for fp16)
        - Loads per-channel scale factors (one float per row)
        - Dequantizes in-register: w_fp = w_int8.to(f32) * scale
        - 2x less memory bandwidth for weight loads
        - No intermediate FP tensor materialized in global memory
        """
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)

        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H

        acc = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)

        for i_start in range(0, I, BLOCK_I):
            i_offsets = i_start + tl.arange(0, BLOCK_I)
            i_mask = i_offsets < I

            gate_acc = tl.zeros((BLOCK_N, BLOCK_I), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_N, BLOCK_I), dtype=tl.float32)

            for h_start in range(0, H, BLOCK_H):
                hk_offsets = h_start + tl.arange(0, BLOCK_H)
                hk_mask = hk_offsets < H

                # Load x tile: (BLOCK_N, BLOCK_H) — native dtype
                x_ptrs = x_ptr + n_offsets[:, None] * stride_x_n + hk_offsets[None, :] * stride_x_h
                x_tile = tl.load(x_ptrs, mask=n_mask[:, None] & hk_mask[None, :], other=0.0).to(tl.float32)

                # Load per-channel scale for this H tile: (BLOCK_H,)
                gu_scale_ptrs = gate_up_scale_ptr + hk_offsets
                gu_scale = tl.load(gu_scale_ptrs, mask=hk_mask, other=0.0)  # (BLOCK_H,)

                # Load INT8 gate weights: (BLOCK_H, BLOCK_I)
                gu_gate_ptrs = gate_up_int8_ptr + hk_offsets[:, None] * stride_gu_h + i_offsets[None, :] * stride_gu_i
                gu_gate_int8 = tl.load(gu_gate_ptrs, mask=hk_mask[:, None] & i_mask[None, :], other=0)
                # Dequant: w_fp = int8_val * scale[row]
                gu_gate_fp = gu_gate_int8.to(tl.float32) * gu_scale[:, None]

                # Load INT8 up weights: (BLOCK_H, BLOCK_I)
                gu_up_ptrs = gate_up_int8_ptr + hk_offsets[:, None] * stride_gu_h + (i_offsets[None, :] + I) * stride_gu_i
                gu_up_int8 = tl.load(gu_up_ptrs, mask=hk_mask[:, None] & i_mask[None, :], other=0)
                gu_up_fp = gu_up_int8.to(tl.float32) * gu_scale[:, None]

                # Accumulate: (BLOCK_N, BLOCK_H) @ (BLOCK_H, BLOCK_I)
                gate_acc += tl.dot(x_tile, gu_gate_fp)
                up_acc += tl.dot(x_tile, gu_up_fp)

            # SwiGLU activation
            sigmoid_gate = tl.sigmoid(gate_acc)
            silu_gate = gate_acc * sigmoid_gate
            inter = silu_gate * up_acc

            # Down projection: load INT8 + dequant inline
            d_ptrs = down_int8_ptr + i_offsets[:, None] * stride_d_i + h_offsets[None, :] * stride_d_h
            d_int8 = tl.load(d_ptrs, mask=i_mask[:, None] & h_mask[None, :], other=0)

            # Per-channel scale for down: (BLOCK_I,)
            d_scale_ptrs = down_scale_ptr + i_offsets
            d_scale = tl.load(d_scale_ptrs, mask=i_mask, other=0.0)

            d_fp = d_int8.to(tl.float32) * d_scale[:, None]

            acc += tl.dot(inter, d_fp)

        # Store output in original dtype
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
    Fused SwiGLU expert forward with INT8 inline dequant via Triton.

    Loads INT8 weights (1 byte each) + per-channel scale, dequantizes
    in-register. Saves 2x memory bandwidth vs pre-dequantized FP16
    and avoids materializing the full FP weight tensor in global memory.

    Args:
        x: (N, H) input tokens
        gate_up_int8: (H, 2*I) int8 weights
        gate_up_scale: (H,) float32 per-channel scale
        down_int8: (I, H) int8 weights
        down_scale: (I,) float32 per-channel scale
        num_tokens: number of tokens

    Returns:
        output: (N, H) or None if Triton unavailable
    """
    if not is_triton_available():
        return None

    H = x.shape[1]
    I = down_int8.shape[0]
    N = num_tokens

    output = torch.empty(N, H, device=x.device, dtype=x.dtype)

    BLOCK_N = min(32, triton.next_power_of_2(N))
    BLOCK_H = min(64, triton.next_power_of_2(H))
    BLOCK_I = min(64, triton.next_power_of_2(I))

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H, BLOCK_H))

    _fused_swiglu_int8_kernel[grid](
        x,
        gate_up_int8, gate_up_scale,
        down_int8, down_scale,
        output,
        N, H, I,
        x.stride(0), x.stride(1),
        gate_up_int8.stride(0), gate_up_int8.stride(1),
        down_int8.stride(0), down_int8.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        BLOCK_I=BLOCK_I,
    )

    return output
