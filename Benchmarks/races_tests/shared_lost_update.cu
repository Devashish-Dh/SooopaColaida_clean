#include <cuda_runtime.h>
#include <cstdio>

__global__ void shared_lost_update_kernel(int *sink) {
    __shared__ int shared_value;

    shared_value = threadIdx.x + 1;
    __syncthreads();

    if (threadIdx.x == 0) {
        sink[0] = shared_value;
    }
}

int main() {
    int *sink = nullptr;
    cudaMalloc(&sink, sizeof(int));

    shared_lost_update_kernel<<<1, 256>>>(sink);
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("shared lost-update: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("shared lost-update: no trap observed in this run\n");
    cudaFree(sink);
    return 0;
}
