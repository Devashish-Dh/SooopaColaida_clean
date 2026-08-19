#pragma once

#include <cstdint>

namespace supercollider {

// LLVM NVPTX address-space identifiers used by CUDA device IR.
constexpr unsigned SC_GENERIC_ADDRESS_SPACE = 0;
constexpr unsigned SC_GLOBAL_ADDRESS_SPACE = 1;
constexpr unsigned SC_SHARED_ADDRESS_SPACE = 3;
constexpr unsigned SC_CONSTANT_ADDRESS_SPACE = 4;
constexpr unsigned SC_LOCAL_ADDRESS_SPACE = 5;
constexpr unsigned SC_INVALID_ADDRESS_SPACE = ~0u;

// Bound recursive provenance walks so malformed or cyclic IR cannot cause
// unbounded analysis.
constexpr unsigned SC_MAX_PROVENANCE_DEPTH = 32;

// Effective memory space associated with a memory operation after provenance
// analysis. Generic means the concrete state space cannot be determined
// statically and may resolve to global, shared, or local memory at runtime.
enum class SCMemorySpace {
    Generic,
    Global,
    Shared,
    Constant,
    Local,
    Unknown,
};

// Ordinary weak memory operations currently handled by the pass.
enum class SCMemoryOpKind {
    Load,
    Store,
};

// Reserve the high SiteID bit for the independent intra-warp lost-update
// reporting namespace so it cannot aggregate into an ordinary lost-update
// record for the same original store.
constexpr std::uint64_t SC_INTRA_WARP_SITE_TAG = 1ULL << 63;

// Race categories consumed by the device and host reporting runtimes.
enum class SCRaceType : std::uint32_t {
    ClobberedRead = 0,
    LostUpdate = 1,
    IntraWarpLostUpdate = 2,
    AsyncCopy = 3,
};

// Aggregate counts used by the pass diagnostics for accepted weak accesses.
struct CandidateSummary {
    std::uint64_t Loads = 0;
    std::uint64_t Stores = 0;

    std::uint64_t GlobalLoads = 0;
    std::uint64_t GlobalStores = 0;
    std::uint64_t SharedLoads = 0;
    std::uint64_t SharedStores = 0;
    std::uint64_t GenericLoads = 0;
    std::uint64_t GenericStores = 0;
};

} // namespace supercollider
