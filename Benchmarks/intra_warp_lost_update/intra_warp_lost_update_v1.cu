#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__device__ int g_ilu_global_target = 0;

__device__ __noinline__ void generic_local_store_noinline(int *ptr, int value) {
    *ptr = value;
}

__global__ void ilu_global_same_value_kernel(int *sink) {
    // Positive ILU case: all 32 lanes of one warp write the same value to the
    // same statically-global address. The ordinary value-based LU detector can
    // miss this because every writer stores the same value; ILU must report it.
    g_ilu_global_target = 17;

    if (threadIdx.x == 0)
        sink[0] = g_ilu_global_target;
}

__global__ void ilu_shared_same_value_kernel(int *sink) {
    __shared__ int shared_target;

    // Positive ILU case in shared memory: every active lane targets the same
    // shared address and stores the same value.
    shared_target = 23;

    __syncthreads();
    if (threadIdx.x == 0)
        sink[1] = shared_target;
}

__global__ void ilu_generic_global_same_value_kernel(int *target, int *sink) {
    // Positive ILU case through a generic pointer that resolves to global
    // memory at runtime. This exercises the generic participant-mask path.
    target[0] = 31;

    if (threadIdx.x == 0)
        sink[2] = target[0];
}

__global__ void ilu_unique_address_negative_kernel(int *target) {
    // Negative control: every lane writes the same value, but to a different
    // address. match.any.sync must see a population count of one per address.
    target[threadIdx.x] = 47;
}

__global__ void ilu_generic_local_negative_kernel(int *sink) {
    int local_value = 0;

    // Negative control for generic -> local. The noinline helper keeps the
    // store behind a generic pointer in a separate device function. Local
    // pointers must be excluded from ILU participation.
    generic_local_store_noinline(&local_value, 59);
    sink[threadIdx.x] = local_value;
}

static void checkCuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess)
        return;

    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
}

int main() {
    constexpr int kWarpSize = 32;

    int *target = nullptr;
    int *sink = nullptr;

    checkCuda(cudaMalloc(&target, kWarpSize * sizeof(int)), "cudaMalloc(target)");
    checkCuda(cudaMalloc(&sink, kWarpSize * sizeof(int)), "cudaMalloc(sink)");
    checkCuda(cudaMemset(target, 0, kWarpSize * sizeof(int)), "cudaMemset(target)");
    checkCuda(cudaMemset(sink, 0, kWarpSize * sizeof(int)), "cudaMemset(sink)");

    // Exactly one warp makes the positive cases deterministic for ILU.
    ilu_global_same_value_kernel<<<1, kWarpSize>>>(sink);
    checkCuda(cudaGetLastError(), "launch ilu_global_same_value_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync ilu_global_same_value_kernel");

    ilu_shared_same_value_kernel<<<1, kWarpSize>>>(sink);
    checkCuda(cudaGetLastError(), "launch ilu_shared_same_value_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync ilu_shared_same_value_kernel");

    ilu_generic_global_same_value_kernel<<<1, kWarpSize>>>(target, sink);
    checkCuda(cudaGetLastError(), "launch ilu_generic_global_same_value_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync ilu_generic_global_same_value_kernel");

    ilu_unique_address_negative_kernel<<<1, kWarpSize>>>(target);
    checkCuda(cudaGetLastError(), "launch ilu_unique_address_negative_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync ilu_unique_address_negative_kernel");

    ilu_generic_local_negative_kernel<<<1, kWarpSize>>>(sink);
    checkCuda(cudaGetLastError(), "launch ilu_generic_local_negative_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync ilu_generic_local_negative_kernel");

    int host_sink[kWarpSize] = {};
    checkCuda(
        cudaMemcpy(host_sink, sink, sizeof(host_sink), cudaMemcpyDeviceToHost),
        "cudaMemcpy(sink)");

    if (host_sink[0] != 59 || host_sink[1] != 59 || host_sink[2] != 59) {
        std::fprintf(
            stderr,
            "unexpected sink values after local negative control: %d %d %d\n",
            host_sink[0],
            host_sink[1],
            host_sink[2]);
        return 2;
    }

    checkCuda(cudaFree(sink), "cudaFree(sink)");
    checkCuda(cudaFree(target), "cudaFree(target)");

    std::printf(
        "ILU test completed. Expect ILU reports for global/shared/generic-global "
        "same-address stores and no ILU report for unique-address or generic-local stores.\n");
    return 0;
}
