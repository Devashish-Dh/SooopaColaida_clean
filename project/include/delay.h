#pragma once

#include "llvm/IR/DebugLoc.h"

#include <cstdint>

namespace llvm {
class CallInst;
class Instruction;
class Value;
}

namespace supercollider {

// The SSA value used as the nanosleep duration together with the inserted
// nanosleep call. SleepCall is null when MaximumDelayNs is zero, in which case
// DelayNs is the constant zero value.
struct RuntimeDelay {
    llvm::Value *DelayNs = nullptr;
    llvm::CallInst *SleepCall = nullptr;
};

// Insert a bounded runtime delay immediately before InsertBefore.
//
// The delay is generated entirely in device IR. Runtime execution identity
// (thread, block, launch dimensions, lane/warp/SM placement, and launch ID)
// is combined with runtime clock values and a compile-time instrumentation
// SiteID. A SplitMix64 finalizer avalanches the combined seed, and the result
// is reduced to [0, MaximumDelayNs] before being passed as the requested delay
// to llvm.nvvm.nanosleep. PTX nanosleep timing is approximate, so this bounds
// the instruction argument rather than measured wall-clock sleep time.
//
// This mechanism is stateless: no device allocation, RNG state, helper runtime,
// or CUDA source modification is required.
RuntimeDelay insertRuntimeDelay(
    llvm::Instruction &InsertBefore,
    std::uint32_t MaximumDelayNs,
    std::uint64_t SiteID,
    const llvm::DebugLoc &DebugLocation);

} // namespace supercollider
