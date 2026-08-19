__device__ unsigned short sc_u16_src[64];
__device__ unsigned short sc_u16_dst[64];

extern "C" __global__ void width_u16_kernel() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64)
        return;

    unsigned short value = sc_u16_src[tid];
    sc_u16_dst[tid] = value;
}
