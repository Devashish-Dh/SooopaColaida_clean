using value_t = unsigned int;

__device__ __noinline__
value_t generic_load_u32(value_t *ptr) {
    // Keep the actual memory operation inside a non-inlined helper whose
    // pointer parameter remains a generic CUDA/NVPTX address.
    return *ptr;
}

__device__ __noinline__
void generic_store_u32(value_t *ptr, value_t value) {
    // This is the generic weak store candidate.
    *ptr = value;
}

extern "C" __global__
void width_generic_u32_kernel(value_t *input, value_t *output) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    value_t value = generic_load_u32(input + tid);
    generic_store_u32(output + tid, value);
}
