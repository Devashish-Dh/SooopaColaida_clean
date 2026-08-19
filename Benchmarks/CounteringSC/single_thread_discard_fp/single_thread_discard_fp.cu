#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void single_thread_fp(uint32_t *p, uint32_t *out) {
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;

    // PTX requires the discarded region to be 128-byte aligned.
    asm volatile(
        "discard.global.L2 [%0], 128;"
        :
        : "l"(p)
        : "memory");

    // Ordinary weak load. SuperCollider should instrument this.
    uint32_t x = p[0];

    out[0] = x;
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
    void *raw = nullptr;
    uint32_t *out = nullptr;

    checkCuda(cudaMalloc(&raw, 256 + 127), "cudaMalloc(raw)");
    checkCuda(cudaMalloc(&out, sizeof(uint32_t)), "cudaMalloc(out)");

    uintptr_t r = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned = (r + 127) & ~uintptr_t(127);
    uint32_t *p = reinterpret_cast<uint32_t *>(aligned);

    single_thread_fp<<<1, 1>>>(p, out);
    checkCuda(cudaGetLastError(), "launch single_thread_fp");
    checkCuda(cudaDeviceSynchronize(), "sync single_thread_fp");

    uint32_t result = 0;
    checkCuda(
        cudaMemcpy(&result, out, sizeof(result), cudaMemcpyDeviceToHost),
        "cudaMemcpy(out)");

    std::printf("result = 0x%08x\n", result);

    checkCuda(cudaFree(out), "cudaFree(out)");
    checkCuda(cudaFree(raw), "cudaFree(raw)");

    return 0;
}
