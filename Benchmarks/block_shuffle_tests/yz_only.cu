#include <cuda_runtime.h>

__global__ void yz_only(unsigned *out) {
    if (threadIdx.x == 0) {
        unsigned slot = blockIdx.y + gridDim.y * blockIdx.z;
        out[slot] = blockIdx.y + 1000u * blockIdx.z;
    }
}
