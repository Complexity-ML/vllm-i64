cmake_minimum_required(VERSION 3.18)
project(vllm_i64_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find PyTorch
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

# CUDA architectures (SM 70+)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

# =========================================================================
# i64 Router kernel
# =========================================================================
add_library(i64_router SHARED
    csrc/i64_router.cu
)
target_link_libraries(i64_router
    ${TORCH_LIBRARIES}
    CUDA::cublas
)
target_include_directories(i64_router PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)
set_target_properties(i64_router PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""
    OUTPUT_NAME "i64_router"
)

# =========================================================================
# i64 Expert Dispatch kernel
# =========================================================================
add_library(i64_expert_dispatch SHARED
    csrc/i64_expert_dispatch.cu
)
target_link_libraries(i64_expert_dispatch
    ${TORCH_LIBRARIES}
    CUDA::cublas
    CUDA::cublasLt
)
target_include_directories(i64_expert_dispatch PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)
set_target_properties(i64_expert_dispatch PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""
    OUTPUT_NAME "i64_expert_dispatch"
)

# =========================================================================
# Install
# =========================================================================
install(TARGETS i64_router i64_expert_dispatch
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
