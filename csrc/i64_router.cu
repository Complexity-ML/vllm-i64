/**
 * vllm-i64 :: i64 Router Kernel
 *
 * Integer-only token routing on GPU.
 * Replaces all float-based MoE routing with pure i64 operations.
 *
 * token_id % num_experts = expert_id (bit masking for power-of-2)
 *
 * INL - 2025
 */

#include <cuda_runtime.h>
#include <cstdint>

// =============================================================================
// Core routing kernel — pure integer, zero float
// =============================================================================

/**
 * Route tokens to experts using integer modulo (bit mask).
 *
 * For num_experts = 2^k, this is a single AND operation per token.
 * No float, no softmax, no learned router weights.
 *
 * @param token_ids     [num_tokens] int64 — input token IDs
 * @param expert_ids    [num_tokens] int32 — output expert assignments
 * @param token_order   [num_tokens] int32 — original position (for gather)
 * @param expert_counts [num_experts] int32 — atomic count per expert
 * @param num_tokens    Total tokens in batch
 * @param expert_mask   num_experts - 1 (bit mask)
 */
__global__ void i64_route_tokens_kernel(
    const int64_t* __restrict__ token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ token_order,
    int32_t* __restrict__ expert_counts,
    const int32_t num_tokens,
    const int32_t expert_mask
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) return;

    // Pure integer routing — single AND operation
    const int32_t expert = (int32_t)(token_ids[idx] & (int64_t)expert_mask);

    expert_ids[idx] = expert;
    token_order[idx] = idx;

    // Atomic count for scatter offsets
    atomicAdd(&expert_counts[expert], 1);
}

// =============================================================================
// Scatter kernel — group tokens by expert
// =============================================================================

/**
 * Scatter tokens into per-expert contiguous buffers.
 * This prepares data for batched expert MLP (one bmm per expert).
 *
 * Two-pass algorithm:
 *   Pass 1: i64_route_tokens_kernel (above) — count per expert
 *   Pass 2: prefix_sum on expert_counts → offsets
 *   Pass 3: this kernel — scatter tokens to contiguous slots
 *
 * @param hidden_in       [num_tokens, hidden_dim] fp16 — input activations
 * @param hidden_out      [num_tokens, hidden_dim] fp16 — scattered output
 * @param expert_ids      [num_tokens] int32
 * @param scatter_indices [num_tokens] int32 — output: position in expert buffer
 * @param expert_offsets  [num_experts] int32 — prefix sum of expert_counts
 * @param expert_counters [num_experts] int32 — atomic counters (init to 0)
 * @param num_tokens      Total tokens
 * @param hidden_dim      Model dimension
 */
__global__ void i64_scatter_kernel(
    const half* __restrict__ hidden_in,
    half* __restrict__ hidden_out,
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ scatter_indices,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ expert_counters,
    const int32_t num_tokens,
    const int32_t hidden_dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int32_t expert = expert_ids[token_idx];

    // Atomic increment to get slot within expert's buffer
    int32_t slot;
    if (threadIdx.x == 0) {
        slot = atomicAdd(&expert_counters[expert], 1);
        scatter_indices[token_idx] = expert_offsets[expert] + slot;
    }
    __syncthreads();

    // Broadcast slot to all threads in block
    __shared__ int32_t shared_slot;
    if (threadIdx.x == 0) {
        shared_slot = expert_offsets[expert] + slot;
    }
    __syncthreads();

    // Copy hidden states (coalesced within block)
    const int out_idx = shared_slot;
    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        hidden_out[out_idx * hidden_dim + d] = hidden_in[token_idx * hidden_dim + d];
    }
}

// =============================================================================
// Gather kernel — reorder results back to original token order
// =============================================================================

/**
 * Gather expert outputs back to original token order.
 * Pure integer indexing — no float involved.
 *
 * @param expert_out    [num_tokens, hidden_dim] fp16 — expert outputs (scattered order)
 * @param output        [num_tokens, hidden_dim] fp16 — final output (original order)
 * @param scatter_indices [num_tokens] int32 — where each token went
 * @param num_tokens    Total tokens
 * @param hidden_dim    Model dimension
 */
__global__ void i64_gather_kernel(
    const half* __restrict__ expert_out,
    half* __restrict__ output,
    const int32_t* __restrict__ scatter_indices,
    const int32_t num_tokens,
    const int32_t hidden_dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    const int32_t src_idx = scatter_indices[token_idx];

    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        output[token_idx * hidden_dim + d] = expert_out[src_idx * hidden_dim + d];
    }
}

// =============================================================================
// Prefix sum on expert counts (small array, single block)
// =============================================================================

__global__ void i64_prefix_sum_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    const int32_t num_experts
) {
    // Single thread — num_experts is tiny (4-16)
    if (threadIdx.x != 0) return;

    int32_t sum = 0;
    for (int i = 0; i < num_experts; i++) {
        expert_offsets[i] = sum;
        sum += expert_counts[i];
    }
}

// =============================================================================
// Host-callable wrapper functions
// =============================================================================

extern "C" {

/**
 * Full i64 routing pipeline:
 *   1. Route (integer bit mask)
 *   2. Prefix sum (expert offsets)
 *   3. Scatter (group by expert)
 *
 * All integer except the hidden state copy in scatter.
 */
void i64_route_and_scatter(
    const int64_t* token_ids,       // [num_tokens]
    const half* hidden_in,          // [num_tokens, hidden_dim]
    half* hidden_out,               // [num_tokens, hidden_dim] scattered
    int32_t* expert_ids,            // [num_tokens] output
    int32_t* scatter_indices,       // [num_tokens] output
    int32_t* expert_offsets,        // [num_experts] output
    int32_t num_tokens,
    int32_t num_experts,
    int32_t hidden_dim,
    cudaStream_t stream
) {
    const int32_t expert_mask = num_experts - 1;

    // Temp buffers
    int32_t* expert_counts;
    int32_t* expert_counters;
    int32_t* token_order;
    cudaMallocAsync(&expert_counts, num_experts * sizeof(int32_t), stream);
    cudaMallocAsync(&expert_counters, num_experts * sizeof(int32_t), stream);
    cudaMallocAsync(&token_order, num_tokens * sizeof(int32_t), stream);
    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int32_t), stream);
    cudaMemsetAsync(expert_counters, 0, num_experts * sizeof(int32_t), stream);

    // 1. Route — pure i64
    const int threads = 256;
    const int blocks_route = (num_tokens + threads - 1) / threads;
    i64_route_tokens_kernel<<<blocks_route, threads, 0, stream>>>(
        token_ids, expert_ids, token_order, expert_counts,
        num_tokens, expert_mask
    );

    // 2. Prefix sum — tiny kernel
    i64_prefix_sum_kernel<<<1, 1, 0, stream>>>(
        expert_counts, expert_offsets, num_experts
    );

    // 3. Scatter — group tokens by expert
    const int threads_scatter = min(hidden_dim, 256);
    i64_scatter_kernel<<<num_tokens, threads_scatter, 0, stream>>>(
        hidden_in, hidden_out, expert_ids, scatter_indices,
        expert_offsets, expert_counters,
        num_tokens, hidden_dim
    );

    cudaFreeAsync(expert_counts, stream);
    cudaFreeAsync(expert_counters, stream);
    cudaFreeAsync(token_order, stream);
}

/**
 * Gather results back to original order.
 */
void i64_gather(
    const half* expert_out,
    half* output,
    const int32_t* scatter_indices,
    int32_t num_tokens,
    int32_t hidden_dim,
    cudaStream_t stream
) {
    const int threads = min(hidden_dim, 256);
    i64_gather_kernel<<<num_tokens, threads, 0, stream>>>(
        expert_out, output, scatter_indices,
        num_tokens, hidden_dim
    );
}

}  // extern "C"
