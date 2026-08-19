#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda/ptx>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kWarpSize = 32;
constexpr int kStrideBytes = 16;
constexpr int kSourceBytes = kWarpSize * kStrideBytes;
constexpr int kBulkBytes = 256;

__global__ void cp_async_4_kernel(
    const unsigned char *src,
    unsigned long long *sink) {

    __shared__ __align__(16) unsigned char smem[kSourceBytes];

    const unsigned lane = threadIdx.x;
    const unsigned offset = lane * kStrideBytes;

    // Classic cp.async characterization: 4-byte global -> shared copy.
    __pipeline_memcpy_async(
        smem + offset,
        src + offset,
        4);
    __pipeline_commit();
    __pipeline_wait_prior(0);

    if (lane == 0) {
        const unsigned int value =
            *reinterpret_cast<const unsigned int *>(smem + offset);
        sink[0] = static_cast<unsigned long long>(value);
    }
}

__global__ void cp_async_8_kernel(
    const unsigned char *src,
    unsigned long long *sink) {

    __shared__ __align__(16) unsigned char smem[kSourceBytes];

    const unsigned lane = threadIdx.x;
    const unsigned offset = lane * kStrideBytes;

    // Classic cp.async characterization: 8-byte global -> shared copy.
    __pipeline_memcpy_async(
        smem + offset,
        src + offset,
        8);
    __pipeline_commit();
    __pipeline_wait_prior(0);

    if (lane == 0) {
        sink[1] = *reinterpret_cast<const unsigned long long *>(
            smem + offset);
    }
}

__global__ void cp_async_16_kernel(
    const unsigned char *src,
    unsigned long long *sink) {

    __shared__ __align__(16) unsigned char smem[kSourceBytes];

    const unsigned lane = threadIdx.x;
    const unsigned offset = lane * kStrideBytes;

    // Classic cp.async characterization: 16-byte global -> shared copy.
    __pipeline_memcpy_async(
        smem + offset,
        src + offset,
        16);
    __pipeline_commit();
    __pipeline_wait_prior(0);

    if (lane == 0) {
        const ulonglong2 value =
            *reinterpret_cast<const ulonglong2 *>(smem + offset);
        sink[2] = value.x;
        sink[3] = value.y;
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

__global__ void cp_async_bulk_shared_to_global_kernel(
    unsigned char *dst) {

    __shared__ alignas(16) unsigned char smem[kBulkBytes];

    for (int i = threadIdx.x; i < kBulkBytes; i += blockDim.x)
        smem[i] = static_cast<unsigned char>((i * 7 + 3) & 0xff);

    __syncthreads();

    // Make ordinary shared-memory writes visible to the async proxy before
    // the shared -> global TMA transfer is initiated.
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    __syncthreads();

    if (threadIdx.x == 0) {
        // Paper-supported bulk direction: shared::cta -> global.
        // This should lower to:
        //   cp.async.bulk.global.shared::cta.bulk_group
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_global,
            cuda::ptx::space_shared,
            dst,
            smem,
            kBulkBytes);

        cuda::ptx::cp_async_bulk_commit_group();

        // Wait until the bulk group has completed reading shared memory.
        cuda::ptx::cp_async_bulk_wait_group_read(
            cuda::ptx::n32_t<0>());
    }
}

#endif

void checkCuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess)
        return;

    std::fprintf(
        stderr,
        "%s failed: %s\n",
        what,
        cudaGetErrorString(status));
    std::exit(1);
}

} // namespace

int main() {
    // Runtime validation is intentionally limited to classic cp.async.
    // The current RTX A6000 is sm_86; cp.async.bulk requires sm_90+.

    unsigned char hostSource[kSourceBytes] = {};
    for (int i = 0; i < kSourceBytes; ++i)
        hostSource[i] = static_cast<unsigned char>((i * 13 + 7) & 0xff);

    unsigned char *deviceSource = nullptr;
    unsigned long long *deviceSink = nullptr;

    checkCuda(
        cudaMalloc(&deviceSource, sizeof(hostSource)),
        "cudaMalloc(deviceSource)");
    checkCuda(
        cudaMalloc(&deviceSink, 4 * sizeof(unsigned long long)),
        "cudaMalloc(deviceSink)");

    checkCuda(
        cudaMemcpy(
            deviceSource,
            hostSource,
            sizeof(hostSource),
            cudaMemcpyHostToDevice),
        "cudaMemcpy(deviceSource)");
    checkCuda(
        cudaMemset(
            deviceSink,
            0,
            4 * sizeof(unsigned long long)),
        "cudaMemset(deviceSink)");

    cp_async_4_kernel<<<1, kWarpSize>>>(deviceSource, deviceSink);
    checkCuda(cudaGetLastError(), "launch cp_async_4_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync cp_async_4_kernel");

    cp_async_8_kernel<<<1, kWarpSize>>>(deviceSource, deviceSink);
    checkCuda(cudaGetLastError(), "launch cp_async_8_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync cp_async_8_kernel");

    cp_async_16_kernel<<<1, kWarpSize>>>(deviceSource, deviceSink);
    checkCuda(cudaGetLastError(), "launch cp_async_16_kernel");
    checkCuda(cudaDeviceSynchronize(), "sync cp_async_16_kernel");

    unsigned long long hostSink[4] = {};
    checkCuda(
        cudaMemcpy(
            hostSink,
            deviceSink,
            sizeof(hostSink),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(hostSink)");

    std::uint32_t expected4 = 0;
    std::uint64_t expected8 = 0;
    std::uint64_t expected16Lo = 0;
    std::uint64_t expected16Hi = 0;

    std::memcpy(&expected4, hostSource, sizeof(expected4));
    std::memcpy(&expected8, hostSource, sizeof(expected8));
    std::memcpy(&expected16Lo, hostSource, sizeof(expected16Lo));
    std::memcpy(
        &expected16Hi,
        hostSource + sizeof(expected16Lo),
        sizeof(expected16Hi));

    if (hostSink[0] != expected4 ||
        hostSink[1] != expected8 ||
        hostSink[2] != expected16Lo ||
        hostSink[3] != expected16Hi) {

        std::fprintf(
            stderr,
            "cp.async result mismatch:\n"
            "  4B     got=0x%llx expected=0x%llx\n"
            "  8B     got=0x%llx expected=0x%llx\n"
            "  16B.lo got=0x%llx expected=0x%llx\n"
            "  16B.hi got=0x%llx expected=0x%llx\n",
            hostSink[0],
            static_cast<unsigned long long>(expected4),
            hostSink[1],
            static_cast<unsigned long long>(expected8),
            hostSink[2],
            static_cast<unsigned long long>(expected16Lo),
            hostSink[3],
            static_cast<unsigned long long>(expected16Hi));
        return 2;
    }

    checkCuda(cudaFree(deviceSink), "cudaFree(deviceSink)");
    checkCuda(cudaFree(deviceSource), "cudaFree(deviceSource)");

    std::printf(
        "Classic cp.async 4/8/16-byte runtime characterization completed successfully.\n"
        "Bulk shared->global is compile/PTX-only in this directory because it requires sm_90+.\n");
    return 0;
}
