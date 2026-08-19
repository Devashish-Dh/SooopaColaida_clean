extern "C" __global__
void width_shared_u8_kernel(volatile unsigned char *sink) {
    __shared__ unsigned char data[64];

    unsigned int tid = threadIdx.x;
    unsigned int other = (tid + 1) & 63;

    // Each thread writes its own shared-memory slot.
    data[tid] = static_cast<unsigned char>(tid + 1);

    // Force real inter-thread communication through shared memory.
    __syncthreads();

    // Read a slot written by a different thread. This prevents the O1
    // pipeline from forwarding the store directly to the load in SSA.
    unsigned char value = data[other];

    // Keep the loaded value observable. The volatile global store should not
    // become a normal SuperCollider weak-memory candidate.
    sink[tid] = value;
}
