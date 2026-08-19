#pragma once

#include "config.h"
#include "filter_memops.h"
#include "internal_defns.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace llvm {
class CallInst;
class LoadInst;
class Value;
}

namespace supercollider {

// Complete compiler-side representation of one instrumented ordinary weak
// memory operation. The record links the original access to every IR object
// implementing its detector and its reporting callback.
struct InstrumentedMemoryOperation {
    WeakMemoryOperation WeakOperation;
    SCRaceType RaceType = SCRaceType::ClobberedRead;

    llvm::Value *ExpectedValue = nullptr;
    llvm::Value *DelayValue = nullptr;
    llvm::CallInst *SleepCall = nullptr;
    llvm::LoadInst *StrongLoad = nullptr;
    llvm::Value *Mismatch = nullptr;
    llvm::CallInst *ReportCall = nullptr;
};

struct InstrumentationSummary {
    std::uint64_t Instrumented = 0;
    std::uint64_t FromLoads = 0;
    std::uint64_t FromStores = 0;

    std::uint64_t Global = 0;
    std::uint64_t Shared = 0;
    std::uint64_t Generic = 0;

    std::uint64_t DelayCalls = 0;
    std::uint64_t ZeroDelaySites = 0;
    std::uint64_t RaceGuards = 0;
    std::uint64_t ReportPaths = 0;

    std::uint64_t UnsupportedType = 0;
    std::uint64_t InvalidInsertionPoint = 0;
    std::uint64_t InvalidComparison = 0;
    std::uint64_t InvalidReporting = 0;
};

// Instrument every accepted weak memory operation with:
//
//   weak load  -> bounded delay -> strong load -> compare -> sc_err
//   weak store -> bounded delay -> strong load -> compare -> sc_err
//
// sc_err records the race and returns to the continuation so one execution can
// collect multiple distinct race sites instead of aborting at the first one.
bool instrumentWeakMemoryOperations(
    llvm::ArrayRef<WeakMemoryOperation> Operations,
    const SCConfig &Config,
    llvm::SmallVectorImpl<InstrumentedMemoryOperation> &Instrumented,
    InstrumentationSummary &Summary);

} // namespace supercollider
