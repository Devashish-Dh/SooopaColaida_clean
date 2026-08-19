#pragma once

#include "config.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace llvm {
class CallInst;
class Module;
class Value;
}

namespace supercollider {

enum class SCAsyncMemoryOpKind {
    ClassicCopy,
    BulkSharedToGlobal,
};

enum class SCAsyncOriginForm {
    PipelineMemcpyCall,
    InlineAsm,
};

// Canonical description of one asynchronous copy accepted by the collector.
// Site IDs share the ordinary reporting namespace and are assigned only after
// the weak load/store candidates have been frozen.
struct AsyncMemoryOperation {
    std::uint64_t SiteID = 0;

    llvm::CallInst *Origin = nullptr;
    llvm::Value *Destination = nullptr;
    llvm::Value *Source = nullptr;
    llvm::Value *ByteCount = nullptr;

    std::uint32_t StaticBytes = 0;
    SCAsyncMemoryOpKind Kind = SCAsyncMemoryOpKind::ClassicCopy;
    SCAsyncOriginForm OriginForm = SCAsyncOriginForm::InlineAsm;

    bool isClassic() const {
        return Kind == SCAsyncMemoryOpKind::ClassicCopy;
    }

    bool isBulkSharedToGlobal() const {
        return Kind == SCAsyncMemoryOpKind::BulkSharedToGlobal;
    }
};

struct AsyncInstrumentationSummary {
    std::uint64_t ClassicCandidates = 0;
    std::uint64_t BulkCandidates = 0;

    std::uint64_t ClassicInstrumented = 0;
    std::uint64_t BulkInstrumented = 0;

    std::uint64_t DelayCalls = 0;
    std::uint64_t ZeroDelaySites = 0;
    std::uint64_t ReportPaths = 0;

    std::uint64_t InvalidOperands = 0;
    std::uint64_t InvalidComparison = 0;
    std::uint64_t InvalidReporting = 0;
};

// Collect classic cp.async and paper-supported cp.async.bulk shared->global
// operations before any IR mutation. Classic CUDA __pipeline_memcpy_async call
// sites are recognized at the caller so source correlation remains attached to
// the application call site instead of a CUDA-header helper body. Direct
// cp.async inline assembly is also accepted outside those helper bodies.
void collectAsyncMemoryOperations(
    llvm::Module &M,
    std::uint64_t FirstSiteID,
    llvm::SmallVectorImpl<AsyncMemoryOperation> &Operations);

// Instrument the frozen asynchronous-copy candidates. Classic 4/8/16-byte
// copies use direct value comparison. Bulk shared->global copies use duplicate
// copies plus FNV-1a hashes over the global destination.
bool instrumentAsyncMemoryOperations(
    llvm::ArrayRef<AsyncMemoryOperation> Operations,
    const SCConfig &Config,
    AsyncInstrumentationSummary &Summary);

void printAsyncInstrumentationSummary(
    const AsyncInstrumentationSummary &Summary,
    const SCConfig &Config);

} // namespace supercollider
