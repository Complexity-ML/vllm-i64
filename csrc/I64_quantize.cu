/**
 * vllm-i64 :: I64_quantize.cu
 *
 * CUDA kernels: activation quantization (INT8 and FP8).
 *
 * INT8: per-token symmetric quantization
 *   x_int8 = round(x / scale), scale = max(|x|) / 127
 *
 * FP8: per-token E4M3 quantization (Hopper/Ada)
 *   x_fp8 = x / scale, scale = max(|x|) / 448
 *
 * Thread mapping: 1 warp-group per token row for reduction,
 * then 1 thread per element for quantization.
 *
 * INL - 2025
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// =========================================================================
// Warp max reduction
// =========================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}


// =========================================================================
// I64_quantize_int8 — per-token INT8 quantization
// =========================================================================

template <int BLOCK_SIZE>
__global__ void I64_quantize_int8_kernel(
    const float* __restrict__ x,          // (N, K)
    int8_t* __restrict__ out,             // (N, K)
    float* __restrict__ out_scale,        // (N,)
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * K;
    int8_t* o_row = out + row * K;

    // Phase 1: Find abs max
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float val = fabsf(x_row[i]);
        if (val > local_max) local_max = val;
    }

    // Warp-level max reduction
    local_max = warp_reduce_max(local_max);

    // Cross-warp reduction via shared memory
    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < (BLOCK_SIZE / 32)) ? shared_max[lane] : 0.0f;
        local_max = warp_reduce_max(local_max);
    }

    __shared__ float scale_shared;
    if (threadIdx.x == 0) {
        float abs_max = fmaxf(local_max, 1e-8f);
        scale_shared = abs_max / 127.0f;
        out_scale[row] = scale_shared;
    }
    __syncthreads();
    float scale = scale_shared;

    // Phase 2: Quantize
    for (int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float val = x_row[i];
        float q = rintf(val / scale);
        q = fminf(fmaxf(q, -128.0f), 127.0f);
        o_row[i] = (int8_t)q;
    }
}


// =========================================================================
// I64_dequantize_int8 — per-token INT8 dequantization
// =========================================================================

__global__ void I64_dequantize_int8_kernel(
    const int8_t* __restrict__ x_int8,    // (N, K)
    const float* __restrict__ scale,      // (N,)
    float* __restrict__ out,              // (N, K)
    int N,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    int row = idx / K;
    out[idx] = (float)x_int8[idx] * scale[row];
}


// =========================================================================
// I64_quantize_per_channel_int8 — per-channel weight quantization
// =========================================================================

__global__ void I64_quantize_perchannel_int8_kernel(
    const float* __restrict__ weight,     // (N, K) — N=out_features, K=in_features
    int8_t* __restrict__ out,             // (N, K)
    float* __restrict__ out_scale,        // (N,)
    int N,
    int K
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* w_row = weight + row * K;
    int8_t* o_row = out + row * K;

    // Find per-channel abs max
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = fabsf(w_row[i]);
        if (val > local_max) local_max = val;
    }

    local_max = warp_reduce_max(local_max);

    __shared__ float shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < (blockDim.x / 32)) ? shared_max[lane] : 0.0f;
        local_max = warp_reduce_max(local_max);
    }

    __shared__ float scale_shared;
    if (threadIdx.x == 0) {
        float abs_max = fmaxf(local_max, 1e-8f);
        scale_shared = abs_max / 127.0f;
        out_scale[row] = scale_shared;
    }
    __syncthreads();
    float scale = scale_shared;

    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = w_row[i];
        float q = rintf(val / scale);
        q = fminf(fmaxf(q, -128.0f), 127.0f);
        o_row[i] = (int8_t)q;
    }
}


// =========================================================================
// PyTorch bindings
// =========================================================================

std::tuple<torch::Tensor, torch::Tensor> I64_quantize_int8(
    torch::Tensor x       // (N, K) float
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");

    int N = x.size(0);
    int K = x.size(1);

    auto x_f = x.to(torch::kFloat32).contiguous();
    auto out = torch::empty({N, K}, torch::dtype(torch::kInt8).device(x.device()));
    auto scale = torch::empty({N}, torch::dtype(torch::kFloat32).device(x.device()));

    if (K <= 512) {
        I64_quantize_int8_kernel<128><<<N, 128>>>(
            x_f.data_ptr<float>(), out.data_ptr<int8_t>(),
            scale.data_ptr<float>(), N, K);
    } else {
        I64_quantize_int8_kernel<256><<<N, 256>>>(
            x_f.data_ptr<float>(), out.data_ptr<int8_t>(),
            scale.data_ptr<float>(), N, K);
    }

    return std::make_tuple(out, scale);
}


torch::Tensor I64_dequantize_int8(
    torch::Tensor x_int8,      // (N, K) int8
    torch::Tensor scale         // (N,) float
) {
    TORCH_CHECK(x_int8.is_cuda(), "x must be CUDA tensor");
    int N = x_int8.size(0);
    int K = x_int8.size(1);

    auto out = torch::empty({N, K}, torch::dtype(torch::kFloat32).device(x_int8.device()));

    int total = N * K;
    int block = 256;
    int grid = (total + block - 1) / block;

    I64_dequantize_int8_kernel<<<grid, block>>>(
        x_int8.data_ptr<int8_t>(), scale.data_ptr<float>(),
        out.data_ptr<float>(), N, K
    );

    return out;
}


std::tuple<torch::Tensor, torch::Tensor> I64_quantize_perchannel_int8(
    torch::Tensor weight     // (N, K) float
) {
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    int N = weight.size(0);
    int K = weight.size(1);

    auto w_f = weight.to(torch::kFloat32).contiguous();
    auto out = torch::empty({N, K}, torch::dtype(torch::kInt8).device(weight.device()));
    auto scale = torch::empty({N}, torch::dtype(torch::kFloat32).device(weight.device()));

    int block = (K <= 512) ? 128 : 256;
    I64_quantize_perchannel_int8_kernel<<<N, block>>>(
        w_f.data_ptr<float>(), out.data_ptr<int8_t>(),
        scale.data_ptr<float>(), N, K
    );

    return std::make_tuple(out, scale);
}
