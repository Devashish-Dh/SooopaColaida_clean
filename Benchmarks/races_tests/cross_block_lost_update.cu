#include <cuda_runtime.h>
#include <cstdio>

__device__ int g_cross_block_value = 0;

__global__ void cross_block_lost_update_kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    g_cross_block_value = tid + 1;
}

int main() {
    cross_block_lost_update_kernel<<<128, 64>>>();
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("cross-block lost-update: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("cross-block lost-update: no trap observed in this run\n");
    return 0;
}
