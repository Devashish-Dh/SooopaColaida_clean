#include <cuda_runtime.h>

__global__ void inline_ptx_ctaid_x(unsigned *out) {
    unsigned physical;
    asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(physical));
    if (threadIdx.x == 0)
        out[0] = physical;
}
