#include <cuda_runtime.h>
#include <cstdio>

__global__ void shared_clobbered_read_kernel(int *sink, int iterations) {
    __shared__ int shared_value;

    if (threadIdx.x == 0) {
        shared_value = 1;
    }
    __syncthreads();

    int acc = 0;
    for (int i = 0; i < iterations; ++i) {
        if ((threadIdx.x & 1) == 0) {
            acc += shared_value;
        } else {
            shared_value = threadIdx.x + i + 1;
        }
    }

    if ((threadIdx.x & 1) == 0) {
        sink[threadIdx.x] = acc;
    }
}

int main() {
    constexpr int threads = 256;
    constexpr int iterations = 4096;

    int *sink = nullptr;
    cudaMalloc(&sink, threads * sizeof(int));

    shared_clobbered_read_kernel<<<1, threads>>>(sink, iterations);
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("shared clobbered-read: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("shared clobbered-read: no trap observed in this run\n");
    cudaFree(sink);
    return 0;
}
