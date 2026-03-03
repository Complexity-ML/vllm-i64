/**
 * vllm-i64 :: PyTorch C++ Extension Binding
 *
 * Bridges the i64 CUDA kernels to PyTorch tensor operations.
 * Uses torch::Tensor for automatic memory management and device handling.
 *
 * Ops:
 *   route_tokens(token_ids, num_experts)          → expert_ids
 *   silu_hadamard(gate_up, expert_inter)           → intermediate  (fp32/fp16/bf16)
 *   scatter_by_expert(hidden, expert_ids, N)       → (scattered, indices, offsets, counts)
 *   gather_by_expert(expert_out, scatter_indices)  → output
 *   atomic_scatter(hidden, expert_ids, N)          → (scattered, scatter_map, offsets, counts)
 *   fused_route_scatter(token_ids, hidden, N)      → (expert_ids, scattered, scatter_map, offsets, counts)
 *
 * Integrates kernels from csrc/i64_router.cu (atomic scatter) and
 * csrc/i64_expert_dispatch.cu (SiLU+Hadamard) into PyTorch-compatible bindings.
 *
 * INL - 2025
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>


// =========================================================================
// Route tokens kernel
// =========================================================================

__global__ void route_tokens_kernel(
    const int64_t* __restrict__ token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t num_tokens,
    int32_t expert_mask
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) return;
    expert_ids[idx] = (int32_t)(token_ids[idx] & (int64_t)expert_mask);
}


torch::Tensor route_tokens(torch::Tensor token_ids, int64_t num_experts) {
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA tensor");
    TORCH_CHECK(token_ids.dtype() == torch::kInt64, "token_ids must be int64");

    int32_t num_tokens = token_ids.size(0);
    int32_t expert_mask = (int32_t)(num_experts - 1);

    auto expert_ids = torch::empty({num_tokens}, torch::TensorOptions()
        .dtype(torch::kInt32).device(token_ids.device()));

    const int threads = 256;
    const int blocks = (num_tokens + threads - 1) / threads;

    route_tokens_kernel<<<blocks, threads>>>(
        token_ids.data_ptr<int64_t>(),
        expert_ids.data_ptr<int32_t>(),
        num_tokens,
        expert_mask
    );

    return expert_ids;
}


// =========================================================================
// Fused SiLU + Hadamard kernels (fp32, fp16, bf16)
//
// Ported from csrc/i64_expert_dispatch.cu, extended with bf16 support.
// =========================================================================

__global__ void silu_hadamard_kernel(
    const float* __restrict__ gate_up,
    float* __restrict__ output,
    int32_t num_tokens,
    int32_t expert_inter
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    for (int d = threadIdx.x; d < expert_inter; d += blockDim.x) {
        int gate_idx = token_idx * 2 * expert_inter + d;
        int up_idx   = token_idx * 2 * expert_inter + expert_inter + d;
        int out_idx  = token_idx * expert_inter + d;

        float g = gate_up[gate_idx];
        float u = gate_up[up_idx];
        float silu_g = g / (1.0f + expf(-g));
        output[out_idx] = silu_g * u;
    }
}


__global__ void silu_hadamard_half_kernel(
    const __half* __restrict__ gate_up,
    __half* __restrict__ output,
    int32_t num_tokens,
    int32_t expert_inter
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    for (int d = threadIdx.x; d < expert_inter; d += blockDim.x) {
        int gate_idx = token_idx * 2 * expert_inter + d;
        int up_idx   = token_idx * 2 * expert_inter + expert_inter + d;
        int out_idx  = token_idx * expert_inter + d;

        float g = __half2float(gate_up[gate_idx]);
        float u = __half2float(gate_up[up_idx]);
        float silu_g = g / (1.0f + expf(-g));
        output[out_idx] = __float2half(silu_g * u);
    }
}


__global__ void silu_hadamard_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    __nv_bfloat16* __restrict__ output,
    int32_t num_tokens,
    int32_t expert_inter
) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    for (int d = threadIdx.x; d < expert_inter; d += blockDim.x) {
        int gate_idx = token_idx * 2 * expert_inter + d;
        int up_idx   = token_idx * 2 * expert_inter + expert_inter + d;
        int out_idx  = token_idx * expert_inter + d;

        float g = __bfloat162float(gate_up[gate_idx]);
        float u = __bfloat162float(gate_up[up_idx]);
        float silu_g = g / (1.0f + expf(-g));
        output[out_idx] = __float2bfloat16(silu_g * u);
    }
}


torch::Tensor silu_hadamard(torch::Tensor gate_up, int64_t expert_inter) {
    TORCH_CHECK(gate_up.is_cuda(), "gate_up must be CUDA tensor");
    TORCH_CHECK(gate_up.dim() == 2, "gate_up must be 2D (tokens, 2*expert_inter)");

    int32_t num_tokens = gate_up.size(0);
    int32_t inter = (int32_t)expert_inter;

    auto output = torch::empty({num_tokens, inter}, gate_up.options());

    int threads = std::min(inter, 256);

    if (gate_up.dtype() == torch::kFloat16) {
        silu_hadamard_half_kernel<<<num_tokens, threads>>>(
            reinterpret_cast<const __half*>(gate_up.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            num_tokens, inter
        );
    } else if (gate_up.dtype() == torch::kBFloat16) {
        silu_hadamard_bf16_kernel<<<num_tokens, threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            num_tokens, inter
        );
    } else {
        silu_hadamard_kernel<<<num_tokens, threads>>>(
            gate_up.data_ptr<float>(),
            output.data_ptr<float>(),
            num_tokens, inter
        );
    }

    return output;
}


// =========================================================================
// Scatter / Gather (PyTorch-level, using argsort)
// =========================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
scatter_by_expert(torch::Tensor hidden, torch::Tensor expert_ids, int64_t num_experts) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA tensor");

    // Sort by expert_id for contiguous expert buffers
    auto sorted_indices = expert_ids.argsort(/*stable=*/true);
    auto scattered = hidden.index_select(0, sorted_indices);
    auto sorted_expert_ids = expert_ids.index_select(0, sorted_indices);

    auto expert_counts = torch::bincount(sorted_expert_ids, {}, num_experts).to(torch::kInt32);
    auto expert_offsets = torch::zeros({num_experts}, torch::TensorOptions()
        .dtype(torch::kInt32).device(hidden.device()));
    torch::cumsum_out(expert_offsets, expert_counts, 0);
    // Shift to get start offsets
    auto offsets = torch::zeros({num_experts}, expert_offsets.options());
    if (num_experts > 1) {
        offsets.slice(0, 1) = expert_offsets.slice(0, 0, num_experts - 1);
    }

    // Build reverse mapping
    auto scatter_indices = torch::zeros_like(sorted_indices);
    scatter_indices.scatter_(0, sorted_indices,
        torch::arange(sorted_indices.size(0), sorted_indices.options()));

    return {scattered, scatter_indices.to(torch::kInt32), offsets, expert_counts};
}


torch::Tensor gather_by_expert(torch::Tensor expert_out, torch::Tensor scatter_indices) {
    return expert_out.index_select(0, scatter_indices.to(torch::kInt64));
}


// =========================================================================
// Atomic scatter — O(N) parallel scatter without sorting
//
// Ported from csrc/i64_router.cu. Uses atomic counters to assign each
// token a contiguous slot within its expert's buffer. Avoids O(N log N)
// argsort for large prefill batches.
// =========================================================================

__global__ void atomic_scatter_kernel(
    const float* __restrict__ hidden_in,
    float* __restrict__ hidden_out,
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ scatter_map,
    int32_t* __restrict__ expert_counters,
    const int32_t* __restrict__ expert_offsets,
    int32_t num_tokens,
    int32_t hidden_dim
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    // Thread 0 atomically claims a slot for this token
    __shared__ int32_t out_idx;
    if (threadIdx.x == 0) {
        int32_t expert = expert_ids[token_idx];
        int32_t slot = atomicAdd(&expert_counters[expert], 1);
        out_idx = expert_offsets[expert] + slot;
        scatter_map[token_idx] = out_idx;
    }
    __syncthreads();

    // All threads cooperate to copy hidden state
    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        hidden_out[out_idx * hidden_dim + d] = hidden_in[token_idx * hidden_dim + d];
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
atomic_scatter(torch::Tensor hidden, torch::Tensor expert_ids, int64_t num_experts) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA tensor");
    TORCH_CHECK(hidden.dtype() == torch::kFloat32,
                "atomic_scatter currently supports float32 (use scatter_by_expert for fp16/bf16)");

    int32_t num_tokens = hidden.size(0);
    int32_t hidden_dim = hidden.size(1);

    // Expert counts via bincount (O(N), GPU)
    auto expert_counts = torch::bincount(expert_ids.to(torch::kInt64),
                                          {}, num_experts).to(torch::kInt32);

    // Prefix sum for offsets
    auto expert_offsets = torch::zeros({num_experts}, torch::TensorOptions()
        .dtype(torch::kInt32).device(hidden.device()));
    if (num_experts > 1) {
        auto cumsum = torch::cumsum(expert_counts, 0);
        expert_offsets.slice(0, 1).copy_(cumsum.slice(0, 0, num_experts - 1));
    }

    // Atomic counters (zeroed)
    auto expert_counters = torch::zeros({num_experts}, torch::TensorOptions()
        .dtype(torch::kInt32).device(hidden.device()));

    // Output buffers
    auto scattered = torch::empty_like(hidden);
    auto scatter_map = torch::empty({num_tokens}, torch::TensorOptions()
        .dtype(torch::kInt32).device(hidden.device()));

    int threads = std::min(hidden_dim, 256);
    atomic_scatter_kernel<<<num_tokens, threads>>>(
        hidden.data_ptr<float>(),
        scattered.data_ptr<float>(),
        expert_ids.data_ptr<int32_t>(),
        scatter_map.data_ptr<int32_t>(),
        expert_counters.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(),
        num_tokens, hidden_dim
    );

    return {scattered, scatter_map, expert_offsets, expert_counts};
}


// =========================================================================
// Fused route + scatter — combines routing and scatter in one call
//
// Eliminates the intermediate expert_ids tensor allocation when both
// routing and scattering are needed (the common case in prefill).
// =========================================================================

__global__ void fused_route_scatter_kernel(
    const int64_t* __restrict__ token_ids,
    const float* __restrict__ hidden_in,
    float* __restrict__ hidden_out,
    int32_t* __restrict__ expert_ids_out,
    int32_t* __restrict__ scatter_map,
    int32_t* __restrict__ expert_counters,
    const int32_t* __restrict__ expert_offsets,
    int32_t num_tokens,
    int32_t hidden_dim,
    int32_t expert_mask
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    __shared__ int32_t out_idx;
    if (threadIdx.x == 0) {
        // Route: pure integer bit mask
        int32_t expert = (int32_t)(token_ids[token_idx] & (int64_t)expert_mask);
        expert_ids_out[token_idx] = expert;

        // Scatter: atomic slot assignment
        int32_t slot = atomicAdd(&expert_counters[expert], 1);
        out_idx = expert_offsets[expert] + slot;
        scatter_map[token_idx] = out_idx;
    }
    __syncthreads();

    // Copy hidden state to scattered position
    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        hidden_out[out_idx * hidden_dim + d] = hidden_in[token_idx * hidden_dim + d];
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_route_scatter(torch::Tensor token_ids, torch::Tensor hidden, int64_t num_experts) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA tensor");
    TORCH_CHECK(hidden.dtype() == torch::kFloat32,
                "fused_route_scatter currently supports float32");

    int32_t num_tokens = hidden.size(0);
    int32_t hidden_dim = hidden.size(1);
    int32_t expert_mask = (int32_t)(num_experts - 1);

    // First pass: route to get expert_ids for bincount
    auto expert_ids = torch::empty({num_tokens}, torch::TensorOptions()
        .dtype(torch::kInt32).device(hidden.device()));

    {
        const int threads = 256;
        const int blocks = (num_tokens + threads - 1) / threads;
        route_tokens_kernel<<<blocks, threads>>>(
            token_ids.data_ptr<int64_t>(),
            expert_ids.data_ptr<int32_t>(),
            num_tokens, expert_mask
        );
    }

    // Expert counts + offsets
    auto expert_counts = torch::bincount(expert_ids.to(torch::kInt64),
                                          {}, num_experts).to(torch::kInt32);
    auto expert_offsets = torch::zeros({num_experts}, torch::TensorOptions()
        .dtype(torch::kInt32).device(hidden.device()));
    if (num_experts > 1) {
        auto cumsum = torch::cumsum(expert_counts, 0);
        expert_offsets.slice(0, 1).copy_(cumsum.slice(0, 0, num_experts - 1));
    }

    auto expert_counters = torch::zeros({num_experts}, torch::TensorOptions()
        .dtype(torch::kInt32).device(hidden.device()));

    // Second pass: fused route + scatter
    auto scattered = torch::empty_like(hidden);
    auto scatter_map = torch::empty({num_tokens}, torch::TensorOptions()
        .dtype(torch::kInt32).device(hidden.device()));

    int threads = std::min(hidden_dim, 256);
    fused_route_scatter_kernel<<<num_tokens, threads>>>(
        token_ids.data_ptr<int64_t>(),
        hidden.data_ptr<float>(),
        scattered.data_ptr<float>(),
        expert_ids.data_ptr<int32_t>(),
        scatter_map.data_ptr<int32_t>(),
        expert_counters.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(),
        num_tokens, hidden_dim, expert_mask
    );

    return {expert_ids, scattered, scatter_map, expert_offsets, expert_counts};
}


// =========================================================================
// Module registration
// =========================================================================

TORCH_LIBRARY(vllm_i64, m) {
    m.def("route_tokens", &route_tokens);
    m.def("silu_hadamard", &silu_hadamard);
    m.def("scatter_by_expert", &scatter_by_expert);
    m.def("gather_by_expert", &gather_by_expert);
    m.def("atomic_scatter", &atomic_scatter);
    m.def("fused_route_scatter", &fused_route_scatter);
}
