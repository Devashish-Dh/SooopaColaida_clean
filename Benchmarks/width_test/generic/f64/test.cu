using value_t = double;

__device__ __noinline__
value_t generic_load_f64(value_t *ptr) {
    // Keep the actual memory operation inside a non-inlined helper whose
    // pointer parameter remains a generic CUDA/NVPTX address.
    return *ptr;
}

__device__ __noinline__
void generic_store_f64(value_t *ptr, value_t value) {
    // This is the generic weak store candidate.
    *ptr = value;
}

extern "C" __global__
void width_generic_f64_kernel(value_t *input, value_t *output) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    value_t value = generic_load_f64(input + tid);
    generic_store_f64(output + tid, value);
}
