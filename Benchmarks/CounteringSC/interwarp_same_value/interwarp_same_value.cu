#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void interwarp_same_value(int *p) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0 && (warp == 0 || warp == 1))
        *p = 123;
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
    int *device_value = nullptr;
    int host_value = 0;

    checkCuda(
        cudaMalloc(&device_value, sizeof(*device_value)),
        "cudaMalloc(device_value)");

    checkCuda(
        cudaMemset(device_value, 0, sizeof(*device_value)),
        "cudaMemset(device_value)");

    interwarp_same_value<<<1, 64>>>(device_value);
    checkCuda(cudaGetLastError(), "launch interwarp_same_value");
    checkCuda(cudaDeviceSynchronize(), "sync interwarp_same_value");

    checkCuda(
        cudaMemcpy(
            &host_value,
            device_value,
            sizeof(host_value),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy(device_value)");

    checkCuda(cudaFree(device_value), "cudaFree(device_value)");

    if (host_value != 123) {
        std::fprintf(
            stderr,
            "interwarp same-value ABA: expected 123, saw %d\n",
            host_value);
        return 2;
    }

    std::printf(
        "interwarp same-value ABA completed: "
        "warp 0 lane 0 and warp 1 lane 0 both wrote 123.\n");

    return 0;
}
