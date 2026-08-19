#include <cuda_runtime.h>
#include <cstdio>

__device__ int g_cross_block_value = 1;

__global__ void cross_block_clobbered_read_kernel(int *sink, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;

    for (int i = 0; i < iterations; ++i) {
        if ((blockIdx.x & 1) == 0) {
            g_cross_block_value = tid + i + 1;
        } else {
            acc += g_cross_block_value;
        }
    }

    if ((blockIdx.x & 1) != 0) {
        sink[tid] = acc;
    }
}

int main() {
    constexpr int blocks = 128;
    constexpr int threads = 64;
    constexpr int iterations = 4096;

    int *sink = nullptr;
    cudaMalloc(&sink, blocks * threads * sizeof(int));

    cross_block_clobbered_read_kernel<<<blocks, threads>>>(sink, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("cross-block clobbered-read: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("cross-block clobbered-read: no trap observed in this run\n");
    cudaFree(sink);
    return 0;
}
