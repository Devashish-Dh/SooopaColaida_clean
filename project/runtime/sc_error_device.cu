#include "sc_runtime_abi.h"

#include <cuda_runtime.h>

#include <cstdint>

using supercollider::SC_MAX_ERROR_SITES;
using supercollider::SC_RUNTIME_MAGIC;
using supercollider::SC_RUNTIME_STATE_BYTES;
using supercollider::SC_RUNTIME_VERSION;
using supercollider::SCRuntimeErrorRecord;
using supercollider::SCRuntimeState;

extern "C" __device__ __align__(8)
unsigned char __sc_runtime_state_storage[SC_RUNTIME_STATE_BYTES] = {};

namespace {

__device__ __forceinline__ SCRuntimeState *asRuntimeState(
    volatile void *Pointer) {
    return reinterpret_cast<SCRuntimeState *>(
        const_cast<void *>(Pointer));
}

__device__ __forceinline__ void initializeRuntimeHeader(
    SCRuntimeState *State) {
    unsigned int *Magic = reinterpret_cast<unsigned int *>(&State->Magic);
    unsigned int Previous = atomicCAS(Magic, 0u, SC_RUNTIME_MAGIC);

    if (Previous == 0u) {
        State->Version = SC_RUNTIME_VERSION;
        __threadfence_system();
    }
}

__device__ __forceinline__ void recordError(
    SCRuntimeState *State,
    std::uint64_t SiteID,
    std::uint64_t Address,
    std::uint32_t RaceType,
    std::uint32_t LaneMask) {
    initializeRuntimeHeader(State);

    for (std::size_t Index = 0; Index < SC_MAX_ERROR_SITES; ++Index) {
        SCRuntimeErrorRecord *Record = &State->Errors[Index];
        unsigned long long *Key = reinterpret_cast<unsigned long long *>(
            &Record->SiteID);

        unsigned long long Existing = atomicCAS(
            Key,
            0ULL,
            static_cast<unsigned long long>(SiteID));

        if (Existing != 0ULL && Existing != SiteID)
            continue;

        if (Existing == 0ULL) {
            Record->FirstAddress = Address;
            Record->RaceType = RaceType;

            Record->ThreadX = threadIdx.x;
            Record->ThreadY = threadIdx.y;
            Record->ThreadZ = threadIdx.z;

            Record->BlockX = blockIdx.x;
            Record->BlockY = blockIdx.y;
            Record->BlockZ = blockIdx.z;

            Record->LaneMask = LaneMask;

            __threadfence_system();
        }

        atomicAdd(
            reinterpret_cast<unsigned long long *>(&Record->Occurrences),
            1ULL);
        return;
    }

    atomicExch(
        reinterpret_cast<unsigned int *>(&State->Overflow),
        1u);
    atomicAdd(
        reinterpret_cast<unsigned long long *>(&State->DroppedOccurrences),
        1ULL);
}

} // namespace

extern "C" __device__ __noinline__
void sc_err(
    volatile void *pRuntimeStr,
    std::uint64_t SiteID,
    std::uint64_t Address,
    std::uint32_t RaceType) {

    if (pRuntimeStr == nullptr) {
        asm volatile("brkpt;");
        return;
    }

    recordError(
        asRuntimeState(pRuntimeStr),
        SiteID,
        Address,
        RaceType,
        0u);
}

extern "C" __device__ __noinline__
void sc_err_ilu(
    volatile void *pRuntimeStr,
    std::uint64_t SiteID,
    std::uint64_t Address,
    std::uint32_t RaceType,
    std::uint32_t LaneMask) {

    if (pRuntimeStr == nullptr) {
        asm volatile("brkpt;");
        return;
    }

    recordError(
        asRuntimeState(pRuntimeStr),
        SiteID,
        Address,
        RaceType,
        LaneMask);
}
