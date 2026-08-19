#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdio>

// Global device memory.
__device__ int g_global[64];

// Dedicated storage for atomic load/store tests.
// Keep this separate from g_global so we don't intentionally mix
// atomic and non-atomic accesses to the same objects.
__device__ int g_atomic[64];

// Constant memory.
__constant__ int g_constant[64];

// Force a generic-pointer load/store through a device function.
__device__ __noinline__
int generic_load(int *ptr)
{
    return *ptr;
}

__device__ __noinline__
void generic_store(int *ptr, int value)
{
    *ptr = value;
}

__global__
void memory_ops_kernel(int *input, int *output, int n)
{
    __shared__ int shared_data[64];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= n)
        return;

    // ------------------------------------------------------------
    // 1. Ordinary global load/store through kernel pointer arguments
    // ------------------------------------------------------------

    int global_load = input[idx];

    output[idx] = global_load;


    // ------------------------------------------------------------
    // 2. Ordinary global load/store through a __device__ variable
    // ------------------------------------------------------------

    int device_global_load = g_global[tid];

    g_global[tid] = device_global_load + 1;


    // ------------------------------------------------------------
    // 3. Shared-memory load/store
    // ------------------------------------------------------------

    shared_data[tid] = global_load;

    __syncthreads();

    int shared_load = shared_data[tid];


    // ------------------------------------------------------------
    // 4. Constant-memory load
    // ------------------------------------------------------------

    int constant_load = g_constant[tid];


    // ------------------------------------------------------------
    // 5. Volatile local/private memory
    // ------------------------------------------------------------

    volatile int local_data[8];

    int local_idx = tid & 7;

    local_data[local_idx] = shared_load;

    int local_load = local_data[local_idx];


    // ------------------------------------------------------------
    // 6. Volatile global load/store
    // ------------------------------------------------------------

    volatile int *volatile_global = g_global;

    int volatile_global_load = volatile_global[tid];

    volatile_global[tid] = volatile_global_load + 1;


    // ------------------------------------------------------------
    // 7. Volatile shared load/store
    // ------------------------------------------------------------

    volatile int *volatile_shared = shared_data;

    int volatile_shared_load = volatile_shared[tid];

    volatile_shared[tid] = volatile_shared_load + 1;


    // ------------------------------------------------------------
    // 8. Generic pointer load/store
    // ------------------------------------------------------------

    int generic_value = generic_load(input + idx);

    generic_store(output + idx, generic_value);


    // ------------------------------------------------------------
    // 9. Atomic read-modify-write
    // ------------------------------------------------------------

    int atomic_old =
        atomicAdd(&g_global[tid], 1);


    // ------------------------------------------------------------
    // 10. Compare-and-swap
    // ------------------------------------------------------------

    int expected = atomic_old;

    int cas_old =
        atomicCAS(&g_global[tid],
                  expected,
                  expected + 1);


    // ------------------------------------------------------------
    // 11. Memory fences
    // ------------------------------------------------------------

    __threadfence_block();

    __threadfence();

    __threadfence_system();


    // ------------------------------------------------------------
    // 12. Ordinary NON-VOLATILE local/private load/store
    //
    // Dynamic indexing prevents mem2reg from trivially promoting
    // the whole array to SSA scalars.
    // ------------------------------------------------------------

    int plain_local_data[8];

    int plain_local_idx = (tid + global_load) & 7;

    plain_local_data[plain_local_idx] =
        shared_load + device_global_load;

    int plain_local_load =
        plain_local_data[plain_local_idx];


    // ------------------------------------------------------------
    // 13. Atomic load/store with explicit memory ordering
    //
    // We want to see these become LLVM atomic LoadInst/StoreInst,
    // and inspect their ordering and sync scope.
    // ------------------------------------------------------------

    cuda::atomic_ref<int, cuda::thread_scope_device>
        atomic_ref(g_atomic[tid]);


    // ------------------------------------------------------------
    // 13a. Relaxed atomic load
    // ------------------------------------------------------------

    int atomic_relaxed_load =
        atomic_ref.load(cuda::memory_order_relaxed);


    // ------------------------------------------------------------
    // 13b. Acquire atomic load
    // ------------------------------------------------------------

    int atomic_acquire_load =
        atomic_ref.load(cuda::memory_order_acquire);


    // ------------------------------------------------------------
    // 13c. Relaxed atomic store
    // ------------------------------------------------------------

    atomic_ref.store(
        atomic_relaxed_load + 1,
        cuda::memory_order_relaxed);


    // ------------------------------------------------------------
    // 13d. Release atomic store
    // ------------------------------------------------------------

    atomic_ref.store(
        atomic_acquire_load + 2,
        cuda::memory_order_release);


    // ------------------------------------------------------------
    // Keep everything live.
    // ------------------------------------------------------------

    output[idx] =
        global_load +
        device_global_load +
        shared_load +
        constant_load +
        local_load +
        volatile_global_load +
        volatile_shared_load +
        generic_value +
        atomic_old +
        cas_old +
        plain_local_load +
        atomic_relaxed_load +
        atomic_acquire_load;
}

int main()
{
    constexpr int N = 64;

    int h_input[N];
    int h_output[N];

    for (int i = 0; i < N; ++i) {
        h_input[i] = i;
        h_output[i] = 0;
    }

    int *d_input = nullptr;
    int *d_output = nullptr;

    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    cudaMemcpy(
        d_input,
        h_input,
        N * sizeof(int),
        cudaMemcpyHostToDevice);

    memory_ops_kernel<<<1, N>>>(
        d_input,
        d_output,
        N);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_output,
        d_output,
        N * sizeof(int),
        cudaMemcpyDeviceToHost);

    printf("output[0] = %d\n", h_output[0]);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}