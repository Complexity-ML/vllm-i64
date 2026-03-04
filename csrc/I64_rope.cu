/**
 * vllm-i64 :: I64_rope.cu
 *
 * CUDA kernel: fused rotary position embedding.
 *
 * Two variants:
 *   I64_rope_forward:         float cos/sin path
 *   I64_rope_integer_forward: Q14 INT16 cos/sin → Q7 input → INT32 rotation → dequant
 *
 * Thread mapping: 1 thread per (token, head, half_dim_element).
 * Fully parallel — no inter-thread communication needed.
 *
 * INL - 2025
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// =========================================================================
// I64_rope_forward — float cos/sin
// =========================================================================

__global__ void I64_rope_float_kernel(
    const float* __restrict__ x,          // (N, num_heads, head_dim)
    const float* __restrict__ cos_ptr,    // (N, head_dim)
    const float* __restrict__ sin_ptr,    // (N, head_dim)
    float* __restrict__ out,              // (N, num_heads, head_dim)
    int N,
    int num_heads,
    int head_dim,
    int half_dim
) {
    // Grid: (N * num_heads * half_dim) total threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * num_heads * half_dim;
    if (idx >= total) return;

    int d = idx % half_dim;
    int tmp = idx / half_dim;
    int h = tmp % num_heads;
    int n = tmp / num_heads;

    // Offsets into x: (n, h, d) and (n, h, d + half_dim)
    int base = n * num_heads * head_dim + h * head_dim;
    float x1 = x[base + d];
    float x2 = x[base + d + half_dim];

    // cos/sin indexed by (n, d) — first half of head_dim
    int cs_idx = n * head_dim + d;
    float c = cos_ptr[cs_idx];
    float s = sin_ptr[cs_idx];

    // Rotate
    out[base + d] = x1 * c - x2 * s;
    out[base + d + half_dim] = x2 * c + x1 * s;
}


// =========================================================================
// I64_rope_integer_forward — Q14 INT16 cos/sin, integer rotation
// =========================================================================

__global__ void I64_rope_integer_kernel(
    const float* __restrict__ x,          // (N, num_heads, head_dim)
    const int16_t* __restrict__ cos_q14,  // (N, head_dim) INT16
    const int16_t* __restrict__ sin_q14,  // (N, head_dim) INT16
    float* __restrict__ out,              // (N, num_heads, head_dim)
    int N,
    int num_heads,
    int head_dim,
    int half_dim,
    float dequant_scale    // 1.0 / (Q_ROPE_IN * Q_ROPE) = 1.0 / (128 * 16384)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * num_heads * half_dim;
    if (idx >= total) return;

    int d = idx % half_dim;
    int tmp = idx / half_dim;
    int h = tmp % num_heads;
    int n = tmp / num_heads;

    int base = n * num_heads * head_dim + h * head_dim;
    float x1_f = x[base + d];
    float x2_f = x[base + d + half_dim];

    // Quantize input to Q7
    int32_t x1_q = __float2int_rn(x1_f * 128.0f);
    int32_t x2_q = __float2int_rn(x2_f * 128.0f);

    // Load Q14 cos/sin
    int cs_idx = n * head_dim + d;
    int32_t c = (int32_t)cos_q14[cs_idx];
    int32_t s = (int32_t)sin_q14[cs_idx];

    // Integer rotation: Q7 × Q14 → Q21
    int32_t r1 = x1_q * c - x2_q * s;
    int32_t r2 = x2_q * c + x1_q * s;

    // Dequant
    out[base + d] = (float)r1 * dequant_scale;
    out[base + d + half_dim] = (float)r2 * dequant_scale;
}


// =========================================================================
// PyTorch bindings
// =========================================================================

torch::Tensor I64_rope_forward(
    torch::Tensor x,          // (N, num_heads, head_dim)
    torch::Tensor cos_vals,   // (N, head_dim)
    torch::Tensor sin_vals    // (N, head_dim)
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, heads, head_dim)");

    int N = x.size(0);
    int num_heads = x.size(1);
    int head_dim = x.size(2);
    int half_dim = head_dim / 2;

    auto out = torch::empty_like(x);
    auto x_f = x.to(torch::kFloat32).contiguous();
    auto cos_f = cos_vals.to(torch::kFloat32).contiguous();
    auto sin_f = sin_vals.to(torch::kFloat32).contiguous();

    int total = N * num_heads * half_dim;
    int block = 256;
    int grid = (total + block - 1) / block;

    I64_rope_float_kernel<<<grid, block>>>(
        x_f.data_ptr<float>(), cos_f.data_ptr<float>(), sin_f.data_ptr<float>(),
        out.data_ptr<float>(), N, num_heads, head_dim, half_dim
    );

    return out;
}


torch::Tensor I64_rope_integer_forward(
    torch::Tensor x,
    torch::Tensor cos_q14,
    torch::Tensor sin_q14
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, heads, head_dim)");

    int N = x.size(0);
    int num_heads = x.size(1);
    int head_dim = x.size(2);
    int half_dim = head_dim / 2;

    auto out = torch::empty_like(x);
    auto x_f = x.to(torch::kFloat32).contiguous();
    auto cos_i = cos_q14.to(torch::kInt16).contiguous();
    auto sin_i = sin_q14.to(torch::kInt16).contiguous();

    float dequant_scale = 1.0f / (128.0f * 16384.0f);

    int total = N * num_heads * half_dim;
    int block = 256;
    int grid = (total + block - 1) / block;

    I64_rope_integer_kernel<<<grid, block>>>(
        x_f.data_ptr<float>(), cos_i.data_ptr<int16_t>(), sin_i.data_ptr<int16_t>(),
        out.data_ptr<float>(), N, num_heads, head_dim, half_dim, dequant_scale
    );

    return out;
}
