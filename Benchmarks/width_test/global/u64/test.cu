__device__ unsigned long long sc_u64_src[64];
__device__ unsigned long long sc_u64_dst[64];

extern "C" __global__ void width_u64_kernel() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64)
        return;

    unsigned long long value = sc_u64_src[tid];
    sc_u64_dst[tid] = value;
}
