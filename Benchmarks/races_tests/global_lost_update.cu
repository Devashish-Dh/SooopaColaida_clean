#include <cuda_runtime.h>
#include <cstdio>

__device__ int g_value = 0;

__global__ void global_lost_update_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    g_value = tid + 1;
}

int main() {
    global_lost_update_kernel<<<1, 256>>>();
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("global lost-update: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("global lost-update: no trap observed in this run\n");
    return 0;
}
