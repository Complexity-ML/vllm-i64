/**
 * vllm-i64 :: I64_softmax.cu
 *
 * CUDA kernel: integer softmax using Q7 LUT.
 *
 * Matches the CPU integer softmax path exactly:
 *   1. Quantize logits to Q7 (×128 → INT32)
 *   2. Subtract row-max for numerical stability
 *   3. Clamp to [-1024, 0]
 *   4. LUT exp(): 1025-entry table (exp(idx/128) × 2^16)
 *   5. Normalize: w_i = exp_i / sum(exp)
 *
 * Eliminates all float transcendentals — pure integer + LUT.
 * The LUT lives in constant memory for fast broadcast reads.
 *
 * INL - 2025
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Integer softmax constants
#define I64_Q_IN 128
#define I64_Q_OUT (1 << 16)
#define I64_LUT_MIN (-1024)
#define I64_LUT_SIZE 1025

// LUT in constant memory — fast broadcast to all threads
__constant__ int32_t I64_exp_lut[I64_LUT_SIZE];

// Host-side LUT (built once, copied to constant memory)
static bool lut_initialized = false;

void I64_init_softmax_lut() {
    if (lut_initialized) return;

    int32_t host_lut[I64_LUT_SIZE];
    for (int i = 0; i < I64_LUT_SIZE; i++) {
        float val = (float)(I64_LUT_MIN + i) / (float)I64_Q_IN;
        host_lut[i] = (int32_t)roundf(expf(val) * (float)I64_Q_OUT);
    }

    cudaMemcpyToSymbol(I64_exp_lut, host_lut, sizeof(host_lut));
    lut_initialized = true;
}


// =========================================================================
// Warp reduction utilities
// =========================================================================

__device__ __forceinline__ int32_t warp_reduce_max_i32(int32_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = max(val, other);
    }
    return val;
}

__device__ __forceinline__ int32_t warp_reduce_sum_i32(int32_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}


// =========================================================================
// I64_softmax_integer_kernel
// =========================================================================

template <int BLOCK_SIZE>
__global__ void I64_softmax_integer_kernel(
    const float* __restrict__ logits,     // (N, D)
    float* __restrict__ out,              // (N, D)
    int N,
    int D
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* l_row = logits + row * D;
    float* o_row = out + row * D;

    // Phase 1: Quantize to Q7 + find max
    int32_t local_max = INT32_MIN;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        int32_t q = __float2int_rn(l_row[i] * (float)I64_Q_IN);
        if (q > local_max) local_max = q;
    }

    // Warp + block reduction for max
    local_max = warp_reduce_max_i32(local_max);
    __shared__ int32_t shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) shared_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < (BLOCK_SIZE / 32)) ? shared_max[lane] : INT32_MIN;
        local_max = warp_reduce_max_i32(local_max);
    }

    __shared__ int32_t row_max_shared;
    if (threadIdx.x == 0) row_max_shared = local_max;
    __syncthreads();
    int32_t row_max = row_max_shared;

    // Phase 2: LUT exp + sum
    int32_t local_sum = 0;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        int32_t q = __float2int_rn(l_row[i] * (float)I64_Q_IN);
        int32_t shifted = q - row_max;

        // Clamp to LUT range
        shifted = max(shifted, I64_LUT_MIN);
        shifted = min(shifted, 0);

        // LUT lookup
        int32_t lut_idx = shifted - I64_LUT_MIN;
        local_sum += I64_exp_lut[lut_idx];
    }

    // Warp + block reduction for sum
    local_sum = warp_reduce_sum_i32(local_sum);
    __shared__ int32_t shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane < (BLOCK_SIZE / 32)) ? shared_sum[lane] : 0;
        local_sum = warp_reduce_sum_i32(local_sum);
    }

    __shared__ float inv_sum_shared;
    if (threadIdx.x == 0) {
        int32_t total = max(local_sum, 1);
        inv_sum_shared = 1.0f / (float)total;
    }
    __syncthreads();
    float inv_sum = inv_sum_shared;

    // Phase 3: Normalize and store
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        int32_t q = __float2int_rn(l_row[i] * (float)I64_Q_IN);
        int32_t shifted = q - row_max;
        shifted = max(shifted, I64_LUT_MIN);
        shifted = min(shifted, 0);

        int32_t lut_idx = shifted - I64_LUT_MIN;
        int32_t exp_val = I64_exp_lut[lut_idx];

        o_row[i] = (float)exp_val * inv_sum;
    }
}


// =========================================================================
// PyTorch binding
// =========================================================================

torch::Tensor I64_softmax_integer(
    torch::Tensor logits     // (N, D) float
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D");

    // Initialize LUT on first call
    I64_init_softmax_lut();

    int N = logits.size(0);
    int D = logits.size(1);

    auto logits_f = logits.to(torch::kFloat32).contiguous();
    auto out = torch::empty({N, D}, torch::dtype(torch::kFloat32).device(logits.device()));

    if (D <= 128) {
        I64_softmax_integer_kernel<64><<<N, 64>>>(
            logits_f.data_ptr<float>(), out.data_ptr<float>(), N, D);
    } else if (D <= 1024) {
        I64_softmax_integer_kernel<128><<<N, 128>>>(
            logits_f.data_ptr<float>(), out.data_ptr<float>(), N, D);
    } else {
        I64_softmax_integer_kernel<256><<<N, 256>>>(
            logits_f.data_ptr<float>(), out.data_ptr<float>(), N, D);
    }

    return out;
}
