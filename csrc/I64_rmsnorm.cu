/**
 * vllm-i64 :: I64_rmsnorm.cu
 *
 * CUDA kernel: fused RMSNorm + optional INT8 quantization.
 *
 * Standard: 3 kernel launches (variance, normalize, weight multiply).
 * Fused: 1 kernel launch, 1 read + 1 write of hidden states.
 *
 * Two variants:
 *   I64_rmsnorm_forward:       float output (standard path)
 *   I64_rmsnorm_quant_forward: INT8 output + per-token scale (integer pipeline)
 *
 * Thread mapping: 1 warp-group (128 threads) per token row.
 * Warp shuffle reduction for variance computation.
 *
 * INL - 2025
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// =========================================================================
// Warp reduction utilities
// =========================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// =========================================================================
// I64_rmsnorm_forward — float output
// =========================================================================

template <int BLOCK_SIZE>
__global__ void I64_rmsnorm_kernel(
    const float* __restrict__ x,          // (N, H)
    const float* __restrict__ weight,     // (H,)
    float* __restrict__ out,              // (N, H)
    int N,
    int H,
    float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * H;
    float* o_row = out + row * H;

    // Each thread handles H/BLOCK_SIZE elements
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) {
        float val = x_row[i];
        var_sum += val * val;
    }

    // Block-level reduction via shared memory
    __shared__ float shared[32];  // one per warp
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    var_sum = warp_reduce_sum(var_sum);
    if (lane == 0) shared[warp_id] = var_sum;
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        var_sum = (lane < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
        var_sum = warp_reduce_sum(var_sum);
    }

    __shared__ float rrms_shared;
    if (threadIdx.x == 0) {
        rrms_shared = rsqrtf(var_sum / (float)H + eps);
    }
    __syncthreads();

    float rrms = rrms_shared;

    // Normalize + weight multiply + store
    for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) {
        float val = x_row[i] * rrms * weight[i];
        o_row[i] = val;
    }
}

// =========================================================================
// I64_rmsnorm_quant_forward — INT8 output + per-token scale
// =========================================================================

template <int BLOCK_SIZE>
__global__ void I64_rmsnorm_quant_kernel(
    const float* __restrict__ x,          // (N, H)
    const float* __restrict__ weight,     // (H,)
    int8_t* __restrict__ out_int8,        // (N, H)
    float* __restrict__ out_scale,        // (N,)
    int N,
    int H,
    float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x_row = x + row * H;
    int8_t* o_row = out_int8 + row * H;

    // Phase 1: Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) {
        float val = x_row[i];
        var_sum += val * val;
    }

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    var_sum = warp_reduce_sum(var_sum);
    if (lane == 0) shared[warp_id] = var_sum;
    __syncthreads();

    if (warp_id == 0) {
        var_sum = (lane < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
        var_sum = warp_reduce_sum(var_sum);
    }

    __shared__ float rrms_shared;
    if (threadIdx.x == 0) {
        rrms_shared = rsqrtf(var_sum / (float)H + eps);
    }
    __syncthreads();
    float rrms = rrms_shared;

    // Phase 2: Normalize + find abs max
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) {
        float val = x_row[i] * rrms * weight[i];
        float abs_val = fabsf(val);
        if (abs_val > local_max) local_max = abs_val;
    }

    // Reduce max across block
    __shared__ float max_shared[32];
    local_max = warp_reduce_sum(local_max);  // warp max via sum (ok for max too if we use __shfl_down)
    // Actually need warp max, not sum — redo with max
    local_max = 0.0f;
    for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) {
        float val = fabsf(x_row[i] * rrms * weight[i]);
        if (val > local_max) local_max = val;
    }
    // Warp max reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }
    if (lane == 0) max_shared[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < (BLOCK_SIZE / 32)) ? max_shared[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            if (other > local_max) local_max = other;
        }
    }

    __shared__ float scale_shared;
    if (threadIdx.x == 0) {
        float abs_max = fmaxf(local_max, 1e-8f);
        scale_shared = abs_max / 127.0f;
        out_scale[row] = scale_shared;
    }
    __syncthreads();
    float scale = scale_shared;

    // Phase 3: Quantize + store
    for (int i = threadIdx.x; i < H; i += BLOCK_SIZE) {
        float val = x_row[i] * rrms * weight[i];
        float q = rintf(val / scale);
        q = fminf(fmaxf(q, -128.0f), 127.0f);
        o_row[i] = (int8_t)q;
    }
}


// =========================================================================
// PyTorch bindings
// =========================================================================

torch::Tensor I64_rmsnorm_forward(
    torch::Tensor x,          // (N, H) float
    torch::Tensor weight,     // (H,) float
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");

    int N = x.size(0);
    int H = x.size(1);

    auto out = torch::empty_like(x);
    auto x_f = x.to(torch::kFloat32).contiguous();
    auto w_f = weight.to(torch::kFloat32).contiguous();

    if (H <= 256) {
        I64_rmsnorm_kernel<128><<<N, 128>>>(
            x_f.data_ptr<float>(), w_f.data_ptr<float>(),
            out.data_ptr<float>(), N, H, (float)eps);
    } else if (H <= 2048) {
        I64_rmsnorm_kernel<256><<<N, 256>>>(
            x_f.data_ptr<float>(), w_f.data_ptr<float>(),
            out.data_ptr<float>(), N, H, (float)eps);
    } else {
        I64_rmsnorm_kernel<512><<<N, 512>>>(
            x_f.data_ptr<float>(), w_f.data_ptr<float>(),
            out.data_ptr<float>(), N, H, (float)eps);
    }

    return out;
}


std::tuple<torch::Tensor, torch::Tensor> I64_rmsnorm_quant_forward(
    torch::Tensor x,
    torch::Tensor weight,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");

    int N = x.size(0);
    int H = x.size(1);

    auto out_int8 = torch::empty({N, H}, torch::dtype(torch::kInt8).device(x.device()));
    auto out_scale = torch::empty({N}, torch::dtype(torch::kFloat32).device(x.device()));
    auto x_f = x.to(torch::kFloat32).contiguous();
    auto w_f = weight.to(torch::kFloat32).contiguous();

    if (H <= 256) {
        I64_rmsnorm_quant_kernel<128><<<N, 128>>>(
            x_f.data_ptr<float>(), w_f.data_ptr<float>(),
            out_int8.data_ptr<int8_t>(), out_scale.data_ptr<float>(),
            N, H, (float)eps);
    } else if (H <= 2048) {
        I64_rmsnorm_quant_kernel<256><<<N, 256>>>(
            x_f.data_ptr<float>(), w_f.data_ptr<float>(),
            out_int8.data_ptr<int8_t>(), out_scale.data_ptr<float>(),
            N, H, (float)eps);
    } else {
        I64_rmsnorm_quant_kernel<512><<<N, 512>>>(
            x_f.data_ptr<float>(), w_f.data_ptr<float>(),
            out_int8.data_ptr<int8_t>(), out_scale.data_ptr<float>(),
            N, H, (float)eps);
    }

    return std::make_tuple(out_int8, out_scale);
}
