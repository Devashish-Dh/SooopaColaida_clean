#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static constexpr int kAtomicIterations = 20000;
static constexpr int kStoreIterations = 128;

__device__ __forceinline__ void compilerBarrier() {
    asm volatile("" ::: "memory");
}

__global__ void weak_store_vs_cta_atomic(unsigned int *value) {
    if (threadIdx.x != 0 || blockIdx.x >= 2)
        return;

    if (blockIdx.x == 0) {
        for (int i = 0; i < kStoreIterations; ++i) {
            compilerBarrier();
            *value = 0x13579bdu + static_cast<unsigned int>(i);
            compilerBarrier();
        }
    } else {
        for (int i = 0; i < kAtomicIterations; ++i)
            atomicAdd_block(value, 1u);
    }
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

    weak_store_vs_cta_atomic<<<2, 1>>>(device_value);
    checkCuda(cudaGetLastError(), "launch weak_store_vs_cta_atomic");
    checkCuda(cudaDeviceSynchronize(), "sync weak_store_vs_cta_atomic");

    checkCuda(cudaMemcpy(&host_value, device_value, sizeof(host_value), cudaMemcpyDeviceToHost),
              "cudaMemcpy(device_value)");
    checkCuda(cudaFree(device_value), "cudaFree(device_value)");

    std::printf(
        "weak store vs CTA atomic: final=0x%08x atomic_iterations=%d weak_store_iterations=%d\n",
        host_value,
        kAtomicIterations,
        kStoreIterations);

    return 0;
}
