#include <cuda_runtime.h>
#include <cstdio>

__device__ __noinline__ void generic_store(int *ptr, int value) {
    *ptr = value;
}

__global__ void generic_lost_update_kernel(int *value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    generic_store(value, tid + 1);
}

int main() {
    int *value = nullptr;
    cudaMalloc(&value, sizeof(int));
    cudaMemset(value, 0, sizeof(int));

    generic_lost_update_kernel<<<1, 256>>>(value);
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("generic lost-update: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("generic lost-update: no trap observed in this run\n");
    cudaFree(value);
    return 0;
}
