#pragma once

#include "filter_memops.h"
#include "internal_defns.h"
#include "race_assert.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/DebugLoc.h"

#include <cstdint>
#include <string>

namespace llvm {
class BranchInst;
class CallInst;
class Instruction;
class Module;
class Value;
}

namespace supercollider {

struct SCErrorInsertion {
    llvm::CallInst *ReportCall = nullptr;
    llvm::BranchInst *ContinueBranch = nullptr;
};

struct SCReportingSite {
    std::uint64_t SiteID = 0;
    SCRaceType RaceType = SCRaceType::ClobberedRead;
    llvm::Instruction *Origin = nullptr;
};

// Replace the trap fallback in a race-failure block with a call matching the
// paper's reporting boundary:
//
//   sc_err(runtime pointer, site ID, racy address, race type)
//
// The runtime pointer references module-local fixed reporting storage. The
// host runtime later aggregates and prints records from that storage.
SCErrorInsertion insertSCErrorReport(
    RaceAssertResult &Assert,
    const WeakMemoryOperation &Operation,
    SCRaceType RaceType,
    const llvm::DebugLoc &DebugLocation);

// Generic reporting entry point for detectors whose racy address is already an
// SSA value rather than a WeakMemoryOperation pointer. Address may be either a
// pointer or an integer address and is normalized to the runtime's i64 ABI.
SCErrorInsertion insertSCErrorReportForSite(
    RaceAssertResult &Assert,
    std::uint64_t SiteID,
    llvm::Value *Address,
    SCRaceType RaceType,
    const llvm::DebugLoc &DebugLocation);

// Emit a compact source-correlation table into the instrumented device module.
// The host runtime uses it to map SiteID to function/file/line information.
void emitSCReportingMetadata(
    llvm::Module &M,
    llvm::ArrayRef<SCReportingSite> Sites);

// Link the CUDA-C++ device callback/runtime module after application memory
// instrumentation has completed. Linking happens after candidate collection so
// sc_err's own memory operations are never instrumented recursively.
bool linkSCDeviceRuntime(llvm::Module &M, std::string &ErrorMessage);

} // namespace supercollider
