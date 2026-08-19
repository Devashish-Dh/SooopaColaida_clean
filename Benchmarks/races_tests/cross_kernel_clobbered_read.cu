#include <cuda_runtime.h>
#include <cstdio>

__global__ void writer_kernel(int *value, int iterations, int bias) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < iterations; ++i) {
        int other = value[(i + 1) & 1];
        value[i & 1] = other + tid + bias + i;
    }
}

__global__ void reader_kernel(int *value, int *sink, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int acc = 0;

    for (int i = 0; i < iterations; ++i) {
        acc += value[i & 1];
    }

    sink[tid] = acc;
}

int main() {
    constexpr int blocks = 64;
    constexpr int threads = 128;
    constexpr int iterations = 4096;

    int *value = nullptr;
    int *sink = nullptr;
    cudaMalloc(&value, 2 * sizeof(int));
    cudaMalloc(&sink, blocks * threads * sizeof(int));
    cudaMemset(value, 0, 2 * sizeof(int));

    cudaStream_t writer_stream;
    cudaStream_t reader_stream;
    cudaStreamCreateWithFlags(&writer_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&reader_stream, cudaStreamNonBlocking);

    writer_kernel<<<blocks, threads, 0, writer_stream>>>(value, iterations, 17);
    reader_kernel<<<blocks, threads, 0, reader_stream>>>(value, sink, iterations);

    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        std::printf("cross-kernel clobbered-read: detector trapped: %s\n", cudaGetErrorString(err));
        return 0;
    }

    std::printf("cross-kernel clobbered-read: no trap observed in this run\n");
    cudaStreamDestroy(reader_stream);
    cudaStreamDestroy(writer_stream);
    cudaFree(sink);
    cudaFree(value);
    return 0;
}
