/**
 * vllm-i64 :: I64_binding.cu
 *
 * Unified PyTorch C++ extension binding for all I64 GPU kernels.
 *
 * Kernels:
 *   I64_rmsnorm:     Fused RMSNorm + optional INT8 quantization
 *   I64_rope:        Fused rotary embedding (float + integer Q14)
 *   I64_quantize:    INT8 activation/weight quantization
 *   I64_softmax:     Integer softmax (Q7 LUT)
 *   I64_gemm:        Fused dequant+GEMM (INT8), fused gate+up+SiLU
 *
 * Build: torch.utils.cpp_extension.load(sources=[...all .cu files...])
 *
 * INL - 2025
 */

#include <torch/extension.h>

// Forward declarations from individual .cu files

// I64_rmsnorm.cu
torch::Tensor I64_rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps);
std::tuple<torch::Tensor, torch::Tensor> I64_rmsnorm_quant_forward(
    torch::Tensor x, torch::Tensor weight, double eps);

// I64_rope.cu
torch::Tensor I64_rope_forward(torch::Tensor x, torch::Tensor cos_vals, torch::Tensor sin_vals);
torch::Tensor I64_rope_integer_forward(torch::Tensor x, torch::Tensor cos_q14, torch::Tensor sin_q14);

// I64_quantize.cu
std::tuple<torch::Tensor, torch::Tensor> I64_quantize_int8(torch::Tensor x);
torch::Tensor I64_dequantize_int8(torch::Tensor x_int8, torch::Tensor scale);
std::tuple<torch::Tensor, torch::Tensor> I64_quantize_perchannel_int8(torch::Tensor weight);

// I64_softmax.cu
torch::Tensor I64_softmax_integer(torch::Tensor logits);
void I64_init_softmax_lut();

// I64_gemm.cu
torch::Tensor I64_gemm_dequant_int8(
    torch::Tensor x, torch::Tensor w_int8, torch::Tensor w_scale);
torch::Tensor I64_gemm_silu_int8(
    torch::Tensor x,
    torch::Tensor gate_int8, torch::Tensor gate_scale,
    torch::Tensor up_int8, torch::Tensor up_scale);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "vllm-i64 GPU kernels — integer-first inference";

    // RMSNorm
    m.def("rmsnorm_forward", &I64_rmsnorm_forward,
          "I64 fused RMSNorm (CUDA)", py::arg("x"), py::arg("weight"), py::arg("eps"));
    m.def("rmsnorm_quant_forward", &I64_rmsnorm_quant_forward,
          "I64 fused RMSNorm + INT8 quantization (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("eps"));

    // RoPE
    m.def("rope_forward", &I64_rope_forward,
          "I64 fused RoPE float (CUDA)", py::arg("x"), py::arg("cos"), py::arg("sin"));
    m.def("rope_integer_forward", &I64_rope_integer_forward,
          "I64 fused RoPE integer Q14 (CUDA)", py::arg("x"), py::arg("cos_q14"), py::arg("sin_q14"));

    // Quantization
    m.def("quantize_int8", &I64_quantize_int8,
          "I64 per-token INT8 quantization (CUDA)", py::arg("x"));
    m.def("dequantize_int8", &I64_dequantize_int8,
          "I64 INT8 dequantization (CUDA)", py::arg("x_int8"), py::arg("scale"));
    m.def("quantize_perchannel_int8", &I64_quantize_perchannel_int8,
          "I64 per-channel INT8 weight quantization (CUDA)", py::arg("weight"));

    // Softmax
    m.def("softmax_integer", &I64_softmax_integer,
          "I64 integer softmax Q7 LUT (CUDA)", py::arg("logits"));
    m.def("init_softmax_lut", &I64_init_softmax_lut,
          "Initialize integer softmax LUT in constant memory");

    // GEMM
    m.def("gemm_dequant_int8", &I64_gemm_dequant_int8,
          "I64 fused INT8 dequant + GEMM (CUDA)",
          py::arg("x"), py::arg("w_int8"), py::arg("w_scale"));
    m.def("gemm_silu_int8", &I64_gemm_silu_int8,
          "I64 fused gate+up GEMM + SiLU (CUDA)",
          py::arg("x"), py::arg("gate_int8"), py::arg("gate_scale"),
          py::arg("up_int8"), py::arg("up_scale"));
}
