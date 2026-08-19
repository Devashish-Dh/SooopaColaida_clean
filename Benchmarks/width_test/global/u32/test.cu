__device__ unsigned int sc_u32_src[64];
__device__ unsigned int sc_u32_dst[64];

extern "C" __global__ void width_u32_kernel() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64)
        return;

    unsigned int value = sc_u32_src[tid];
    sc_u32_dst[tid] = value;
}
