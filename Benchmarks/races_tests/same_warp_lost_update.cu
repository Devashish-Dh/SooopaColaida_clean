#include <cuda_runtime.h>
#include <cstdio>

__device__ int g_warp_value = 0;

__global__ void same_warp_lost_update_kernel() {
    // Distinct values are intentional. The current detector can catch these
    // with the universal lost-update check. Same-value writes require the
    // separate intra-warp address-match detector, which is not implemented yet.
    g_warp_value = threadIdx.x + 1;
}

int main() {
    same_warp_lost_update_kernel<<<1, 32>>>();
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("same-warp lost-update: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("same-warp lost-update: no trap observed in this run\n");
    return 0;
}
