#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(x) do {                                           \
    cudaError_t e = (x);                                             \
    if (e != cudaSuccess) {                                          \
        std::fprintf(                                                \
            stderr,                                                  \
            "CUDA error %s:%d: %s\n",                                \
            __FILE__,                                                \
            __LINE__,                                                \
            cudaGetErrorString(e));                                  \
        std::exit(1);                                                \
    }                                                                \
} while (0)

#define CHECK_DRV(x) do {                                            \
    CUresult e = (x);                                                \
    if (e != CUDA_SUCCESS) {                                         \
        const char *s = nullptr;                                     \
        cuGetErrorString(e, &s);                                     \
        std::fprintf(                                                \
            stderr,                                                  \
            "Driver error %s:%d: %s\n",                              \
            __FILE__,                                                \
            __LINE__,                                                \
            s ? s : "unknown");                                     \
        std::exit(1);                                                \
    }                                                                \
} while (0)

__global__ void alias_no_fence(
    uint32_t *A,
    uint32_t *B,
    uint32_t *observed) {
    int tid = threadIdx.x;

    // Different warps so the intra-warp detector cannot rescue us.
    if (tid == 0) {
        *A = 0x12345678;
    }

    if (tid == 32) {
        // Scheduling perturbation only. No synchronization.
        for (int i = 0; i < 100; ++i)
            __nanosleep(1000);

        // A and B are different virtual addresses mapping the same physical
        // GPU memory.
        uint32_t x = *B;

        observed[0] = x;
    }
}

__global__ void alias_single_thread_with_fence(
    uint32_t *A,
    uint32_t *B,
    uint32_t *observed) {
    if (threadIdx.x != 0)
        return;

    *A = 0x87654321;

    // Required ordering mechanism between virtual aliases.
    asm volatile(
        "fence.proxy.alias;"
        :
        :
        : "memory");

    observed[0] = *B;
}

int main() {
    CHECK_CUDA(cudaSetDevice(0));

    CHECK_DRV(cuInit(0));

    CUdevice dev;
    CHECK_DRV(cuDeviceGet(&dev, 0));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t granularity = 0;
    CHECK_DRV(cuMemGetAllocationGranularity(
        &granularity,
        &prop,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    size_t size = granularity;

    CUmemGenericAllocationHandle handle;
    CHECK_DRV(cuMemCreate(
        &handle,
        size,
        &prop,
        0));

    CUdeviceptr vaA = 0;
    CUdeviceptr vaB = 0;

    CHECK_DRV(cuMemAddressReserve(
        &vaA,
        size,
        granularity,
        0,
        0));

    CHECK_DRV(cuMemAddressReserve(
        &vaB,
        size,
        granularity,
        0,
        0));

    // Map the same physical allocation twice.
    CHECK_DRV(cuMemMap(
        vaA,
        size,
        0,
        handle,
        0));

    CHECK_DRV(cuMemMap(
        vaB,
        size,
        0,
        handle,
        0));

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CHECK_DRV(cuMemSetAccess(
        vaA,
        size,
        &access,
        1));

    CHECK_DRV(cuMemSetAccess(
        vaB,
        size,
        &access,
        1));

    uint32_t *A = reinterpret_cast<uint32_t *>(vaA);
    uint32_t *B = reinterpret_cast<uint32_t *>(vaB);

    uint32_t *observed = nullptr;
    CHECK_CUDA(cudaMalloc(&observed, sizeof(uint32_t)));

    std::printf("A virtual address = %p\n", static_cast<void *>(A));
    std::printf("B virtual address = %p\n", static_cast<void *>(B));
    std::printf("same VA?          = %s\n", A == B ? "YES" : "NO");

    // Establish initial physical contents before the test operation.
    CHECK_CUDA(cudaMemset(A, 0, size));
    CHECK_CUDA(cudaDeviceSynchronize());

    uint32_t zero = 0;
    CHECK_CUDA(cudaMemcpy(
        observed,
        &zero,
        sizeof(zero),
        cudaMemcpyHostToDevice));

    std::printf("\n=== no fence: cross-alias test ===\n");

    alias_no_fence<<<1, 64>>>(A, B, observed);

    CHECK_CUDA(cudaDeviceSynchronize());

    uint32_t h = 0;
    CHECK_CUDA(cudaMemcpy(
        &h,
        observed,
        sizeof(h),
        cudaMemcpyDeviceToHost));

    std::printf("B observed 0x%08x\n", h);

    // Reset at a device-operation boundary.
    CHECK_CUDA(cudaMemset(A, 0, size));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        observed,
        &zero,
        sizeof(zero),
        cudaMemcpyHostToDevice));

    std::printf("\n=== fence.proxy.alias control ===\n");

    alias_single_thread_with_fence<<<1, 1>>>(A, B, observed);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(
        &h,
        observed,
        sizeof(h),
        cudaMemcpyDeviceToHost));

    std::printf("B observed 0x%08x\n", h);

    CHECK_CUDA(cudaFree(observed));

    CHECK_DRV(cuMemUnmap(vaA, size));
    CHECK_DRV(cuMemUnmap(vaB, size));

    CHECK_DRV(cuMemAddressFree(vaA, size));
    CHECK_DRV(cuMemAddressFree(vaB, size));

    CHECK_DRV(cuMemRelease(handle));

    return 0;
}
