#pragma once

#include "config.h"
#include "filter_memops.h"
#include "instrument_memops.h"
#include "internal_defns.h"

namespace llvm {
class Function;
}

namespace supercollider {

void updateCandidateSummary(
    CandidateSummary &Summary,
    const WeakMemoryOperation &Operation);

// Print the configuration that controls the detector expansion for this pass
// invocation.
void printConfiguration(const SCConfig &Config);

// Print a weak memory operation that passed the instrumentation filter.
void printInstrumentableCandidate(
    const llvm::Function &F,
    const WeakMemoryOperation &Operation);

void printCandidateSummary(const CandidateSummary &Summary);

// Print the complete detector sequence associated with one weak memory access.
void printInstrumentedMemoryOperation(
    const InstrumentedMemoryOperation &Operation,
    const SCConfig &Config);

void printInstrumentationSummary(const InstrumentationSummary &Summary);

} // namespace supercollider
