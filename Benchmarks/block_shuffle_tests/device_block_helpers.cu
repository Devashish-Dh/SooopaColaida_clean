#include <cuda_runtime.h>

#include <cstdint>

extern "C" __device__ __noinline__ unsigned sc_test_prepass_helper_block_x() {
    return blockIdx.x;
}

extern "C" __device__ __noinline__ std::uint64_t
sc_test_prepass_helper_linear_block(std::uint64_t scale, std::uint64_t bias) {
    return static_cast<std::uint64_t>(blockIdx.x) * scale + bias;
}
