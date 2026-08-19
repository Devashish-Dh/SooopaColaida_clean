extern "C" __global__
void width_shared_u64_kernel(volatile unsigned long long *sink) {
    __shared__ unsigned long long data[64];

    unsigned int tid = threadIdx.x;
    unsigned int other = (tid + 1) & 63;

    // Each thread writes its own shared-memory slot.
    data[tid] = static_cast<unsigned long long>(tid + 1);

    // Force real inter-thread communication through shared memory.
    __syncthreads();

    // Read a slot written by a different thread. This prevents the O1
    // pipeline from forwarding the store directly to the load in SSA.
    unsigned long long value = data[other];

    // Keep the loaded value observable. The volatile global store should not
    // become a normal SuperCollider weak-memory candidate.
    sink[tid] = value;
}
