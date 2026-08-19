__device__ double sc_f64_src[64];
__device__ double sc_f64_dst[64];

extern "C" __global__ void width_f64_kernel() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 64)
        return;

    double value = sc_f64_src[tid];
    sc_f64_dst[tid] = value;
}
