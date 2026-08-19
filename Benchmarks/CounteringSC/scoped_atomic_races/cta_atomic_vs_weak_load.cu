#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static constexpr int kAtomicIterations = 20000;
static constexpr int kLoadIterations = 512;

__device__ __forceinline__ void compilerBarrier() {
    asm volatile("" ::: "memory");
}

__global__ void cta_atomic_vs_weak_load(unsigned int *value, unsigned long long *sink) {
    if (threadIdx.x != 0 || blockIdx.x >= 2)
        return;

    if (blockIdx.x == 0) {
        unsigned long long sum = 0;
        for (int i = 0; i < kLoadIterations; ++i) {
            compilerBarrier();
            sum += *value;
            compilerBarrier();
        }
        *sink = sum;
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
    unsigned long long *device_sink = nullptr;
    unsigned int host_value = 0;
    unsigned long long host_sink = 0;

    checkCuda(cudaMalloc(&device_value, sizeof(*device_value)), "cudaMalloc(device_value)");
    checkCuda(cudaMalloc(&device_sink, sizeof(*device_sink)), "cudaMalloc(device_sink)");
    checkCuda(cudaMemset(device_value, 0, sizeof(*device_value)), "cudaMemset(device_value)");
    checkCuda(cudaMemset(device_sink, 0, sizeof(*device_sink)), "cudaMemset(device_sink)");

    cta_atomic_vs_weak_load<<<2, 1>>>(device_value, device_sink);
    checkCuda(cudaGetLastError(), "launch cta_atomic_vs_weak_load");
    checkCuda(cudaDeviceSynchronize(), "sync cta_atomic_vs_weak_load");

    checkCuda(cudaMemcpy(&host_value, device_value, sizeof(host_value), cudaMemcpyDeviceToHost),
              "cudaMemcpy(device_value)");
    checkCuda(cudaMemcpy(&host_sink, device_sink, sizeof(host_sink), cudaMemcpyDeviceToHost),
              "cudaMemcpy(device_sink)");

    checkCuda(cudaFree(device_sink), "cudaFree(device_sink)");
    checkCuda(cudaFree(device_value), "cudaFree(device_value)");

    std::printf(
        "CTA atomic vs weak load: final=%u load_checksum=%llu atomic_iterations=%d load_iterations=%d\n",
        host_value,
        host_sink,
        kAtomicIterations,
        kLoadIterations);

    return 0;
}
