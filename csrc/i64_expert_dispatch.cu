/**
 * vllm-i64 :: Expert Dispatch Kernel
 *
 * Batched SwiGLU MLP for token-routed experts.
 * Each expert processes its assigned tokens via fused gate+up matmul.
 *
 * Layout after scatter:
 *   hidden_scattered = [tokens_e0 | tokens_e1 | ... | tokens_eN]
 *   Each segment is contiguous → efficient batched matmul.
 *
 * INL - 2025
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>

// =============================================================================
// Batched expert MLP — one GEMM per expert, fused gate+up
// =============================================================================

/**
 * Execute all experts in sequence using cuBLAS GEMM.
 *
 * For each expert e:
 *   tokens_e = hidden_scattered[offset_e : offset_e + count_e]
 *   gate_up  = tokens_e @ W_gate_up[e]     (fused)
 *   inter    = silu(gate) * up
 *   output_e = inter @ W_down[e]
 *
 * @param handle         cuBLAS handle
 * @param hidden_in      [total_tokens, hidden_dim] fp16 — scattered by expert
 * @param hidden_out     [total_tokens, hidden_dim] fp16 — output (scattered order)
 * @param gate_up_weights [num_experts, hidden_dim, 2 * expert_inter] fp16
 * @param down_weights    [num_experts, expert_inter, hidden_dim] fp16
 * @param expert_offsets  [num_experts] int32 — start index per expert
 * @param expert_counts   [num_experts] int32 — token count per expert
 * @param num_experts     Number of experts
 * @param hidden_dim      Model dimension
 * @param expert_inter    Expert intermediate dimension (d_ff / N)
 * @param stream          CUDA stream
 */
extern "C" void i64_expert_mlp(
    cublasHandle_t handle,
    const half* hidden_in,
    half* hidden_out,
    const half* gate_up_weights,
    const half* down_weights,
    const int32_t* expert_offsets,
    const int32_t* expert_counts,
    int32_t num_experts,
    int32_t hidden_dim,
    int32_t expert_inter,
    cudaStream_t stream
);

// =============================================================================
// SiLU + Hadamard kernel (fused activation)
// =============================================================================

/**
 * Fused SiLU activation + Hadamard product.
 *
 * Input:  gate_up [num_tokens, 2 * expert_inter]
 *         gate = gate_up[:, :expert_inter]
 *         up   = gate_up[:, expert_inter:]
 *
 * Output: intermediate [num_tokens, expert_inter]
 *         intermediate = silu(gate) * up
 *
 * This is the ONLY place where float activation happens.
 * Everything else (routing, scatter, gather, indexing) is integer.
 */
__global__ void i64_silu_hadamard_kernel(
    const half* __restrict__ gate_up,
    half* __restrict__ intermediate,
    const int32_t num_tokens,
    const int32_t expert_inter
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    for (int d = threadIdx.x; d < expert_inter; d += blockDim.x) {
        const int gate_idx = token_idx * 2 * expert_inter + d;
        const int up_idx   = token_idx * 2 * expert_inter + expert_inter + d;
        const int out_idx  = token_idx * expert_inter + d;

        float g = __half2float(gate_up[gate_idx]);
        float u = __half2float(gate_up[up_idx]);

        // SiLU = x * sigmoid(x)
        float silu_g = g / (1.0f + expf(-g));

        intermediate[out_idx] = __float2half(silu_g * u);
    }
}

// =============================================================================
// Full expert dispatch pipeline
// =============================================================================

/**
 * Process all experts on their assigned tokens.
 *
 * Integer control flow:
 *   for e in 0..N:
 *     count = expert_counts[e]       // i32
 *     offset = expert_offsets[e]     // i32
 *     if count == 0: continue        // i32 comparison
 *
 * Float compute (only inside expert):
 *     gate_up = tokens[offset:offset+count] @ W_gate_up[e]   // fp16 GEMM
 *     inter = silu(gate) * up                                  // fp16 activation
 *     output = inter @ W_down[e]                               // fp16 GEMM
 *
 * Everything outside the 3 lines above is pure integer.
 */
extern "C" void i64_expert_dispatch(
    cublasHandle_t handle,
    const half* hidden_scattered,
    half* output_scattered,
    const half* gate_up_weights,    // [N, hidden_dim, 2*expert_inter]
    const half* down_weights,       // [N, expert_inter, hidden_dim]
    const int32_t* expert_offsets,  // [N]
    const int32_t* expert_counts,   // [N]
    int32_t num_experts,
    int32_t hidden_dim,
    int32_t expert_inter,
    cudaStream_t stream
) {
    cublasSetStream(handle, stream);

    const half alpha_h = __float2half(1.0f);
    const half beta_h  = __float2half(0.0f);

    // Temp buffer for gate_up output
    half* gate_up_buf;
    // Max tokens per expert (conservative)
    int32_t max_tokens = 0;
    int32_t h_counts[32];  // Max 32 experts
    int32_t h_offsets[32];
    cudaMemcpyAsync(h_counts, expert_counts, num_experts * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_offsets, expert_offsets, num_experts * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int e = 0; e < num_experts; e++) {
        if (h_counts[e] > max_tokens) max_tokens = h_counts[e];
    }

    cudaMallocAsync(&gate_up_buf, max_tokens * 2 * expert_inter * sizeof(half), stream);
    half* inter_buf;
    cudaMallocAsync(&inter_buf, max_tokens * expert_inter * sizeof(half), stream);

    // Process each expert — integer control, float compute
    for (int32_t e = 0; e < num_experts; e++) {
        const int32_t count = h_counts[e];
        const int32_t offset = h_offsets[e];

        // Skip empty experts (integer comparison)
        if (count == 0) continue;

        const half* tokens_e = hidden_scattered + (int64_t)offset * hidden_dim;
        half* output_e = output_scattered + (int64_t)offset * hidden_dim;

        // Expert weights (integer indexing into weight tensor)
        const half* w_gate_up = gate_up_weights + (int64_t)e * hidden_dim * 2 * expert_inter;
        const half* w_down = down_weights + (int64_t)e * expert_inter * hidden_dim;

        // --- FLOAT ZONE START ---

        // 1. Gate+Up projection: [count, hidden_dim] @ [hidden_dim, 2*expert_inter]
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            2 * expert_inter, count, hidden_dim,
            &alpha_h,
            w_gate_up, 2 * expert_inter,
            tokens_e, hidden_dim,
            &beta_h,
            gate_up_buf, 2 * expert_inter
        );

        // 2. SiLU + Hadamard
        const int threads = min(expert_inter, 256);
        i64_silu_hadamard_kernel<<<count, threads, 0, stream>>>(
            gate_up_buf, inter_buf, count, expert_inter
        );

        // 3. Down projection: [count, expert_inter] @ [expert_inter, hidden_dim]
        cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            hidden_dim, count, expert_inter,
            &alpha_h,
            w_down, hidden_dim,
            inter_buf, expert_inter,
            &beta_h,
            output_e, hidden_dim
        );

        // --- FLOAT ZONE END ---
    }

    cudaFreeAsync(gate_up_buf, stream);
    cudaFreeAsync(inter_buf, stream);
}
