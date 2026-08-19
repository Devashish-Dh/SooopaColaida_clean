#include <cuda_runtime.h>
#include <cstdio>

__global__ void updating_kernel(int *value, int iterations, int bias) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < iterations; ++i) {
        int old = value[0];
        value[0] = old + tid + bias + 1;
    }
}

int main() {
    constexpr int blocks = 64;
    constexpr int threads = 128;
    constexpr int iterations = 4096;

    int *value = nullptr;
    cudaMalloc(&value, sizeof(int));
    cudaMemset(value, 0, sizeof(int));

    cudaStream_t stream_a;
    cudaStream_t stream_b;
    cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking);

    updating_kernel<<<blocks, threads, 0, stream_a>>>(value, iterations, 1);
    updating_kernel<<<blocks, threads, 0, stream_b>>>(value, iterations, 1000003);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("cross-kernel lost-update: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("cross-kernel lost-update: no trap observed in this run\n");
    cudaStreamDestroy(stream_b);
    cudaStreamDestroy(stream_a);
    cudaFree(value);
    return 0;
}
