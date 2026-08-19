#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static constexpr int kIterations = 100000;

__global__ void cta_cta_different_blocks(unsigned int *value) {
    if (threadIdx.x != 0 || blockIdx.x >= 2)
        return;

    for (int i = 0; i < kIterations; ++i)
        atomicAdd_block(value, 1u);
}

static void checkCuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess)
        return;

    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
}

int main() {
    unsigned int *device_value = nullptr;
    unsigned int host_value = 0;

    checkCuda(cudaMalloc(&device_value, sizeof(*device_value)), "cudaMalloc(device_value)");
    checkCuda(cudaMemset(device_value, 0, sizeof(*device_value)), "cudaMemset(device_value)");

    cta_cta_different_blocks<<<2, 1>>>(device_value);
    checkCuda(cudaGetLastError(), "launch cta_cta_different_blocks");
    checkCuda(cudaDeviceSynchronize(), "sync cta_cta_different_blocks");

    checkCuda(cudaMemcpy(&host_value, device_value, sizeof(host_value), cudaMemcpyDeviceToHost),
              "cudaMemcpy(device_value)");
    checkCuda(cudaFree(device_value), "cudaFree(device_value)");

    const unsigned int hardware_expected = 2u * static_cast<unsigned int>(kIterations);
    std::printf(
        "different-CTA block-scope atomics: final=%u hardware_expected_if_globally_serialized=%u\n",
        host_value,
        hardware_expected);

    return 0;
}
