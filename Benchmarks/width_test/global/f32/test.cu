__device__ float sc_f32_src[64];
__device__ float sc_f32_dst[64];

extern "C" __global__ void width_f32_kernel() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64)
        return;

    float value = sc_f32_src[tid];
    sc_f32_dst[tid] = value;
}
