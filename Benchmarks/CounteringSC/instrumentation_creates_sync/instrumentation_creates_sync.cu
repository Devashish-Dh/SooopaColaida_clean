#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ void store_release_sys(int *p, int value) {
    asm volatile(
        "st.release.sys.global.u32 [%0], %1;"
        :
        : "l"(p), "r"(value)
        : "memory");
}

__device__ __forceinline__ void acquire_fence_sys() {
    asm volatile(
        "fence.acquire.sys;"
        :
        :
        : "memory");
}

__global__ void instrumentation_creates_sync(
    int *data,
    int *flag,
    int *observed_flag,
    int *observed_data) {
    if (threadIdx.x == 0) {
        // Ordinary weak store.
        *data = 42;

        // Strong release store.
        store_release_sys(flag, 1);
    }

    if (threadIdx.x == 32) {
        // Only increase likelihood that producer runs first.
        // This is NOT synchronization.
        for (int i = 0; i < 100; ++i)
            __nanosleep(1000);

        // IMPORTANT: ordinary weak load.
        int f = *flag;

        if (f == 1) {
            // Original program contains this fence.
            acquire_fence_sys();

            // Ordinary weak load.
            int d = *data;

            *observed_flag = f;
            *observed_data = d;
        }
    }
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
    int *data = nullptr;
    int *flag = nullptr;
    int *of = nullptr;
    int *od = nullptr;

    checkCuda(cudaMalloc(&data, sizeof(int)), "cudaMalloc(data)");
    checkCuda(cudaMalloc(&flag, sizeof(int)), "cudaMalloc(flag)");
    checkCuda(cudaMalloc(&of, sizeof(int)), "cudaMalloc(of)");
    checkCuda(cudaMalloc(&od, sizeof(int)), "cudaMalloc(od)");

    int zero = 0;
    int minus_one = -1;

    for (int iteration = 0; iteration < 100000; ++iteration) {
        checkCuda(
            cudaMemcpy(data, &zero, sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy(data)");
        checkCuda(
            cudaMemcpy(flag, &zero, sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy(flag)");
        checkCuda(
            cudaMemcpy(of, &minus_one, sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy(of)");
        checkCuda(
            cudaMemcpy(od, &minus_one, sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy(od)");

        instrumentation_creates_sync<<<1, 64>>>(data, flag, of, od);

        checkCuda(cudaGetLastError(), "launch instrumentation_creates_sync");
        checkCuda(cudaDeviceSynchronize(), "sync instrumentation_creates_sync");

        int hf = -1;
        int hd = -1;

        checkCuda(
            cudaMemcpy(&hf, of, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy(hf)");
        checkCuda(
            cudaMemcpy(&hd, od, sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy(hd)");

        if (hf == 1 && hd == 0) {
            std::printf(
                "Observed weak message-passing outcome at iteration %d\n",
                iteration);
            break;
        }
    }

    checkCuda(cudaFree(data), "cudaFree(data)");
    checkCuda(cudaFree(flag), "cudaFree(flag)");
    checkCuda(cudaFree(of), "cudaFree(of)");
    checkCuda(cudaFree(od), "cudaFree(od)");

    return 0;
}
