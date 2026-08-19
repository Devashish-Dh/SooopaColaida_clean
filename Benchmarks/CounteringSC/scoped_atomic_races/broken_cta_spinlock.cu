#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static constexpr int kIterations = 4000;

__device__ __forceinline__ void compilerBarrier() {
    asm volatile("" ::: "memory");
}

__global__ void broken_cta_spinlock(unsigned int *lock, unsigned int *counter) {
    if (threadIdx.x != 0 || blockIdx.x >= 2)
        return;

    for (int i = 0; i < kIterations; ++i) {
        while (atomicCAS_block(lock, 0u, 1u) != 0u)
            __nanosleep(32);

        compilerBarrier();
        unsigned int old = *counter;
        *counter = old + 1u;
        compilerBarrier();

        atomicExch_block(lock, 0u);
    }
}

static void checkCuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess)
        return;

    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
}

int main() {
    unsigned int *device_lock = nullptr;
    unsigned int *device_counter = nullptr;
    unsigned int host_lock = 0;
    unsigned int host_counter = 0;

    checkCuda(cudaMalloc(&device_lock, sizeof(*device_lock)), "cudaMalloc(device_lock)");
    checkCuda(cudaMalloc(&device_counter, sizeof(*device_counter)), "cudaMalloc(device_counter)");
    checkCuda(cudaMemset(device_lock, 0, sizeof(*device_lock)), "cudaMemset(device_lock)");
    checkCuda(cudaMemset(device_counter, 0, sizeof(*device_counter)), "cudaMemset(device_counter)");

    broken_cta_spinlock<<<2, 1>>>(device_lock, device_counter);
    checkCuda(cudaGetLastError(), "launch broken_cta_spinlock");
    checkCuda(cudaDeviceSynchronize(), "sync broken_cta_spinlock");

    checkCuda(cudaMemcpy(&host_lock, device_lock, sizeof(host_lock), cudaMemcpyDeviceToHost),
              "cudaMemcpy(device_lock)");
    checkCuda(cudaMemcpy(&host_counter, device_counter, sizeof(host_counter), cudaMemcpyDeviceToHost),
              "cudaMemcpy(device_counter)");

    checkCuda(cudaFree(device_counter), "cudaFree(device_counter)");
    checkCuda(cudaFree(device_lock), "cudaFree(device_lock)");

    const unsigned int hardware_expected = 2u * static_cast<unsigned int>(kIterations);
    std::printf(
        "broken CTA-scope spinlock: counter=%u hardware_expected_if_lock_serializes=%u final_lock=%u\n",
        host_counter,
        hardware_expected,
        host_lock);

    return 0;
}
