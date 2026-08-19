#include <cuda_runtime.h>
#include <cstdio>

__device__ __noinline__ int generic_load(int *ptr) {
    return *ptr;
}

__device__ __noinline__ void generic_store(int *ptr, int value) {
    *ptr = value;
}

__global__ void generic_clobbered_read_kernel(int *value, int *sink, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;

    for (int i = 0; i < iterations; ++i) {
        if ((threadIdx.x & 1) == 0) {
            acc += generic_load(value);
        } else {
            generic_store(value, tid + i + 1);
        }
    }

    if ((threadIdx.x & 1) == 0) {
        sink[tid] = acc;
    }
}

int main() {
    constexpr int blocks = 64;
    constexpr int threads = 128;
    constexpr int iterations = 4096;

    int *value = nullptr;
    int *sink = nullptr;
    cudaMalloc(&value, sizeof(int));
    cudaMalloc(&sink, blocks * threads * sizeof(int));
    cudaMemset(value, 0, sizeof(int));

    generic_clobbered_read_kernel<<<blocks, threads>>>(value, sink, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("generic clobbered-read: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("generic clobbered-read: no trap observed in this run\n");
    cudaFree(sink);
    cudaFree(value);
    return 0;
}
