#include <cuda_runtime.h>

#include <cstdint>

// This translation unit is intentionally linked AFTER the SuperCollider pass.
// It gives the test harness an unshuffled observation of the physical CTA ID.
extern "C" __device__ __noinline__ unsigned sc_test_physical_block_x() {
    return blockIdx.x;
}

// PTX %gridid is a 64-bit temporal grid-launch identifier. This helper is also
// linked after the pass so the test can independently reproduce the mapping.
extern "C" __device__ __noinline__ std::uint64_t sc_test_gridid() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(value));
    return value;
}
