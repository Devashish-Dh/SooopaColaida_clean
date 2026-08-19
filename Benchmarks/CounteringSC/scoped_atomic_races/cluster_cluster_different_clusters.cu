#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static constexpr int kIterations = 100000;

__global__ void cluster_cluster_different_clusters(unsigned int *value) {
    if (threadIdx.x != 0 || blockIdx.x >= 2)
        return;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    for (int i = 0; i < kIterations; ++i) {
        __nv_atomic_fetch_add(
            value,
            1u,
            __NV_ATOMIC_RELAXED,
            __NV_THREAD_SCOPE_CLUSTER);
    }
#else
#error "cluster_cluster_different_clusters requires sm_90 or newer"
#endif
}

static void checkCuda(cudaError_t status, const char *what) {
    if (status == cudaSuccess)
        return;

    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
}

int main() {
    unsigned int *device_value = nullptr;
    unsigned int host_value = 0;

    checkCuda(cudaMalloc(&device_value, sizeof(*device_value)), "cudaMalloc(device_value)");
    checkCuda(cudaMemset(device_value, 0, sizeof(*device_value)), "cudaMemset(device_value)");

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(2, 1, 1);
    config.blockDim = dim3(1, 1, 1);

    cudaLaunchAttribute attribute = {};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = 1;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;

    config.attrs = &attribute;
    config.numAttrs = 1;

    checkCuda(
        cudaLaunchKernelEx(&config, cluster_cluster_different_clusters, device_value),
        "cudaLaunchKernelEx(cluster_cluster_different_clusters)");
    checkCuda(cudaDeviceSynchronize(), "sync cluster_cluster_different_clusters");

    checkCuda(cudaMemcpy(&host_value, device_value, sizeof(host_value), cudaMemcpyDeviceToHost),
              "cudaMemcpy(device_value)");
    checkCuda(cudaFree(device_value), "cudaFree(device_value)");

    const unsigned int hardware_expected = 2u * static_cast<unsigned int>(kIterations);
    std::printf(
        "cluster-scope atomics across different 1-CTA clusters: final=%u hardware_expected_if_globally_serialized=%u\n",
        host_value,
        hardware_expected);

    return 0;
}
