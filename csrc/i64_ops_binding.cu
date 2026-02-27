/**
 * vllm-i64 :: PyTorch C++ Extension Binding
 *
 * Bridges the i64 CUDA kernels to PyTorch tensor operations.
 * Uses torch::Tensor for automatic memory management and device handling.
 *
 * This file provides the torch.ops interface:
 *   torch.ops.vllm_i64.route_tokens(token_ids, num_experts) → expert_ids
 *   torch.ops.vllm_i64.scatter(hidden, expert_ids, num_experts) → (scattered, indices, offsets, counts)
 *   torch.ops.vllm_i64.gather(expert_out, scatter_indices) → output
 *   torch.ops.vllm_i64.silu_hadamard(gate_up, expert_inter) → intermediate
 *
 * INL - 2025
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
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
// Fused SiLU + Hadamard kernel
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


torch::Tensor silu_hadamard(torch::Tensor gate_up, int64_t expert_inter) {
    TORCH_CHECK(gate_up.is_cuda(), "gate_up must be CUDA tensor");
    TORCH_CHECK(gate_up.dim() == 2, "gate_up must be 2D (tokens, 2*expert_inter)");

    int32_t num_tokens = gate_up.size(0);
    int32_t inter = (int32_t)expert_inter;

    auto output = torch::empty({num_tokens, inter},
        gate_up.options());

    int threads = std::min(inter, 256);

    if (gate_up.dtype() == torch::kFloat16) {
        silu_hadamard_half_kernel<<<num_tokens, threads>>>(
            reinterpret_cast<const __half*>(gate_up.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
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
// Module registration
// =========================================================================

TORCH_LIBRARY(vllm_i64, m) {
    m.def("route_tokens", &route_tokens);
    m.def("silu_hadamard", &silu_hadamard);
    m.def("scatter_by_expert", &scatter_by_expert);
    m.def("gather_by_expert", &gather_by_expert);
}
