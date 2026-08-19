#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__device__ int g_ilu_global_target = 0;

__device__ __noinline__ void generic_local_store_noinline(
    int *ptr,
    int value) {
    *ptr = value;
}

__global__ void ilu_global_same_value_kernel(int *sink) {
    // Positive ILU case:
    // Only lanes 2 and 3 write the same value to the same global address.
    //
    // Expected ILU lane mask:
    //   { l2 l3 }
    if (threadIdx.x == 2 || threadIdx.x == 3) {
        g_ilu_global_target = 17;
    }

    if (threadIdx.x == 0)
        sink[0] = g_ilu_global_target;
}

__global__ void ilu_shared_same_value_kernel(int *sink) {
    __shared__ int shared_target;

    // Initialize the shared value first.
    if (threadIdx.x == 0)
        shared_target = 0;

    __syncthreads();

    // Positive ILU case:
    // Only lanes 5, 6, 7, and 8 write the same shared address.
    //
    // Expected ILU lane mask:
    //   { l5 l6 l7 l8 }
    if (threadIdx.x >= 5 && threadIdx.x <= 8) {
        shared_target = 23;
    }

    __syncthreads();

    if (threadIdx.x == 0)
        sink[1] = shared_target;
}

__global__ void ilu_generic_global_same_value_kernel(
    int *target,
    int *sink) {

    // Positive ILU case through a generic pointer resolving to global memory.
    //
    // Only lanes 10-14 perform the store.
    //
    // Expected ILU lane mask:
    //   { l10 l11 l12 l13 l14 }
    if (threadIdx.x >= 10 && threadIdx.x <= 14) {
        target[0] = 31;
    }

    if (threadIdx.x == 0)
        sink[2] = target[0];
}

__global__ void ilu_unique_address_negative_kernel(int *target) {
    // Negative control:
    // all lanes execute a store, but every lane uses a unique address.
    //
    // Every match.any group should contain exactly one lane.
    target[threadIdx.x] = 47;
}

__global__ void ilu_generic_local_negative_kernel(int *sink) {
    int local_value = 0;

    // Negative control for generic -> local.
    // Every lane invokes the helper, but each pointer refers to that
    // thread's private local-memory object.
    generic_local_store_noinline(&local_value, 59);

    sink[threadIdx.x] = local_value;
}

static void checkCuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess)
        return;

    std::fprintf(
        stderr,
        "%s failed: %s\n",
        what,
        cudaGetErrorString(status));
    std::exit(1);
}

int main() {
    constexpr int kWarpSize = 32;

    int *target = nullptr;
    int *sink = nullptr;

    checkCuda(
        cudaMalloc(&target, kWarpSize * sizeof(int)),
        "cudaMalloc(target)");

    checkCuda(
        cudaMalloc(&sink, kWarpSize * sizeof(int)),
        "cudaMalloc(sink)");

    checkCuda(
        cudaMemset(target, 0, kWarpSize * sizeof(int)),
        "cudaMemset(target)");

    checkCuda(
        cudaMemset(sink, 0, kWarpSize * sizeof(int)),
        "cudaMemset(sink)");

    // 2-lane global collision:
    //
    // Expected:
    //   Lanes in warp: { l2 l3 }
    ilu_global_same_value_kernel<<<1, kWarpSize>>>(sink);
    checkCuda(
        cudaGetLastError(),
        "launch ilu_global_same_value_kernel");
    checkCuda(
        cudaDeviceSynchronize(),
        "sync ilu_global_same_value_kernel");

    // 4-lane shared-memory collision:
    //
    // Expected:
    //   Lanes in warp: { l5 l6 l7 l8 }
    ilu_shared_same_value_kernel<<<1, kWarpSize>>>(sink);
    checkCuda(
        cudaGetLastError(),
        "launch ilu_shared_same_value_kernel");
    checkCuda(
        cudaDeviceSynchronize(),
        "sync ilu_shared_same_value_kernel");

    // 5-lane generic -> global collision:
    //
    // Expected:
    //   Lanes in warp: { l10 l11 l12 l13 l14 }
    ilu_generic_global_same_value_kernel<<<1, kWarpSize>>>(
        target,
        sink);
    checkCuda(
        cudaGetLastError(),
        "launch ilu_generic_global_same_value_kernel");
    checkCuda(
        cudaDeviceSynchronize(),
        "sync ilu_generic_global_same_value_kernel");

    // Negative: unique addresses.
    ilu_unique_address_negative_kernel<<<1, kWarpSize>>>(target);
    checkCuda(
        cudaGetLastError(),
        "launch ilu_unique_address_negative_kernel");
    checkCuda(
        cudaDeviceSynchronize(),
        "sync ilu_unique_address_negative_kernel");

    // Negative: generic pointers resolving to local memory.
    ilu_generic_local_negative_kernel<<<1, kWarpSize>>>(sink);
    checkCuda(
        cudaGetLastError(),
        "launch ilu_generic_local_negative_kernel");
    checkCuda(
        cudaDeviceSynchronize(),
        "sync ilu_generic_local_negative_kernel");

    int host_sink[kWarpSize] = {};

    checkCuda(
        cudaMemcpy(
            host_sink,
            sink,
            sizeof(host_sink),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(sink)");

    if (host_sink[0] != 59 ||
        host_sink[1] != 59 ||
        host_sink[2] != 59) {

        std::fprintf(
            stderr,
            "unexpected sink values after local negative control: "
            "%d %d %d\n",
            host_sink[0],
            host_sink[1],
            host_sink[2]);
        return 2;
    }

    checkCuda(cudaFree(sink), "cudaFree(sink)");
    checkCuda(cudaFree(target), "cudaFree(target)");

    std::printf(
        "ILU subset-lane test completed.\n"
        "Expected positive ILU masks:\n"
        "  global         : { l2 l3 }\n"
        "  shared         : { l5 l6 l7 l8 }\n"
        "  generic-global : { l10 l11 l12 l13 l14 }\n"
        "Expected no ILU report for unique-address or generic-local "
        "stores.\n");

    return 0;
}