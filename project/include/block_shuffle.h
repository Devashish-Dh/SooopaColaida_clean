#pragma once

#include "config.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace llvm {
class CallInst;
class Module;
}

namespace supercollider {

struct BlockShuffleSummary {
    std::uint64_t BlockIndexReads = 0;
    std::uint64_t Rewritten = 0;
    std::uint64_t Functions = 0;
    std::uint64_t InlineAsmCTAIDReferences = 0;
};

// Freeze all application blockIdx.x reads before detector instrumentation adds
// any of its own special-register reads. Handwritten inline PTX that directly
// reads %ctaid.x is rejected because it cannot be safely redirected to the
// logical affine mapping at LLVM IR level.
bool collectBlockShuffleCandidates(
    llvm::Module &M,
    llvm::SmallVectorImpl<llvm::CallInst *> &Candidates,
    BlockShuffleSummary &Summary,
    std::string &ErrorMessage);

// Rewrite the frozen application blockIdx.x reads using:
//
//   logical_x = (a * physical_x + b) mod gridDim.x
//
// The per-launch coefficients are derived from %gridid and the configured
// seed. The selected multiplier is always coprime with gridDim.x, so the map is
// bijective for every fixed (blockIdx.y, blockIdx.z) plane.
bool instrumentBlockShuffling(
    llvm::ArrayRef<llvm::CallInst *> Candidates,
    const SCConfig &Config,
    BlockShuffleSummary &Summary);

void printBlockShuffleSummary(
    const BlockShuffleSummary &Summary,
    const SCConfig &Config);

} // namespace supercollider
