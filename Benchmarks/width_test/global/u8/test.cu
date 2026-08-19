__device__ unsigned char sc_u8_src[64];
__device__ unsigned char sc_u8_dst[64];

extern "C" __global__ void width_u8_kernel() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64)
        return;

    unsigned char value = sc_u8_src[tid];
    sc_u8_dst[tid] = value;
}
