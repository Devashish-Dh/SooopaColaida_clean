#pragma once

#include "filter_memops.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace supercollider {

struct IntraWarpLostUpdateSummary {
    std::uint64_t StoresConsidered = 0;
    std::uint64_t Instrumented = 0;

    std::uint64_t Global = 0;
    std::uint64_t Shared = 0;
    std::uint64_t Generic = 0;

    std::uint64_t GenericLocalGuards = 0;
    std::uint64_t ReportPaths = 0;

    std::uint64_t InvalidInsertionPoint = 0;
    std::uint64_t InvalidSiteID = 0;
};

// Insert the paper's independent intra-warp lost-update detector before each
// original weak store. The detector compares store addresses across the
// currently participating warp lanes with match.any.sync and reports when more
// than one participating lane targets the same address.
//
// Generic pointers receive an additional runtime isspacep.local check. A
// ballot-derived member mask excludes local-memory lanes before match.any.sync
// so every lane named in the synchronized member mask executes the intrinsic.
//
// This transformation does not modify or replace the ordinary post-store
// lost-update detector. The original store remains intact and can be processed
// later by instrumentWeakMemoryOperations().
bool instrumentIntraWarpLostUpdates(
    llvm::ArrayRef<WeakMemoryOperation> Operations,
    IntraWarpLostUpdateSummary &Summary);

void printIntraWarpLostUpdateSummary(
    const IntraWarpLostUpdateSummary &Summary);

} // namespace supercollider
