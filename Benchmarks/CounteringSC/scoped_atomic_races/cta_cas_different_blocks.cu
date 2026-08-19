#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr int kSlots = 4096;

__global__ void cta_cas_different_blocks(unsigned int *slots) {
    if (threadIdx.x != 0 || blockIdx.x >= 2)
        return;

    const unsigned int desired = blockIdx.x == 0 ? 1u : 2u;

    for (int i = 0; i < kSlots; ++i)
        atomicCAS_block(&slots[i], 0u, desired);
}

static void checkCuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess)
        return;

    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
}

int main() {
    unsigned int *device_slots = nullptr;
    std::vector<unsigned int> host_slots(kSlots, 0u);

    checkCuda(cudaMalloc(&device_slots, kSlots * sizeof(*device_slots)), "cudaMalloc(device_slots)");
    checkCuda(cudaMemset(device_slots, 0, kSlots * sizeof(*device_slots)), "cudaMemset(device_slots)");

    cta_cas_different_blocks<<<2, 1>>>(device_slots);
    checkCuda(cudaGetLastError(), "launch cta_cas_different_blocks");
    checkCuda(cudaDeviceSynchronize(), "sync cta_cas_different_blocks");

    checkCuda(cudaMemcpy(host_slots.data(), device_slots, kSlots * sizeof(*device_slots), cudaMemcpyDeviceToHost),
              "cudaMemcpy(device_slots)");
    checkCuda(cudaFree(device_slots), "cudaFree(device_slots)");

    int won_by_block0 = 0;
    int won_by_block1 = 0;
    int unexpected = 0;

    for (unsigned int v : host_slots) {
        if (v == 1u)
            ++won_by_block0;
        else if (v == 2u)
            ++won_by_block1;
        else
            ++unexpected;
    }

    std::printf(
        "CTA-scope CAS across CTAs: block0_wins=%d block1_wins=%d unexpected=%d slots=%d\n",
        won_by_block0,
        won_by_block1,
        unexpected,
        kSlots);

    return 0;
}
