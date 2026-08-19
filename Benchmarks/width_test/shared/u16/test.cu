extern "C" __global__
void width_shared_u16_kernel(volatile unsigned short *sink) {
    __shared__ unsigned short data[64];

    unsigned int tid = threadIdx.x;
    unsigned int other = (tid + 1) & 63;

    // Each thread writes its own shared-memory slot.
    data[tid] = static_cast<unsigned short>(tid + 1);

    // Force real inter-thread communication through shared memory.
    __syncthreads();

    // Read a slot written by a different thread. This prevents the O1
    // pipeline from forwarding the store directly to the load in SSA.
    unsigned short value = data[other];

    // Keep the loaded value observable. The volatile global store should not
    // become a normal SuperCollider weak-memory candidate.
    sink[tid] = value;
}
