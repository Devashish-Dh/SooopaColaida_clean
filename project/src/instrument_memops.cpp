#include "instrument_memops.h"

#include "delay.h"
#include "race_assert.h"
#include "sc_error.h"
#include "strong_memops.h"

#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"

namespace supercollider {

namespace {

void updateInstrumentationSummary(
    InstrumentationSummary &Summary,
    const WeakMemoryOperation &Operation,
    const InstrumentedMemoryOperation &InstrumentedOperation) {

    ++Summary.Instrumented;

    if (Operation.isLoad())
        ++Summary.FromLoads;
    else
        ++Summary.FromStores;

    switch (Operation.Space) {
    case SCMemorySpace::Global:
        ++Summary.Global;
        break;
    case SCMemorySpace::Shared:
        ++Summary.Shared;
        break;
    case SCMemorySpace::Generic:
        ++Summary.Generic;
        break;
    default:
        break;
    }

    if (InstrumentedOperation.SleepCall != nullptr)
        ++Summary.DelayCalls;
    else
        ++Summary.ZeroDelaySites;

    if (InstrumentedOperation.Mismatch != nullptr)
        ++Summary.RaceGuards;
    if (InstrumentedOperation.ReportCall != nullptr)
        ++Summary.ReportPaths;
}

llvm::Value *getExpectedValue(const WeakMemoryOperation &Operation) {
    if (Operation.isLoad())
        return llvm::dyn_cast_or_null<llvm::LoadInst>(Operation.Origin);

    if (auto *Store = llvm::dyn_cast_or_null<llvm::StoreInst>(Operation.Origin))
        return Store->getValueOperand();

    return nullptr;
}

SCRaceType getRaceType(const WeakMemoryOperation &Operation) {
    return Operation.isLoad()
               ? SCRaceType::ClobberedRead
               : SCRaceType::LostUpdate;
}

std::uint32_t getMaximumDelay(
    const WeakMemoryOperation &Operation,
    const SCConfig &Config) {

    return Operation.isLoad()
               ? Config.ReadDelayNs
               : Config.WriteDelayNs;
}

} // namespace

bool instrumentWeakMemoryOperations(
    llvm::ArrayRef<WeakMemoryOperation> Operations,
    const SCConfig &Config,
    llvm::SmallVectorImpl<InstrumentedMemoryOperation> &Instrumented,
    InstrumentationSummary &Summary) {

    bool Changed = false;

    for (const WeakMemoryOperation &Operation : Operations) {
        if (Operation.Origin == nullptr ||
            Operation.Pointer == nullptr ||
            Operation.ValueType == nullptr) {
            ++Summary.InvalidInsertionPoint;
            continue;
        }

        if (!isSupportedStrongLoadType(Operation.ValueType) ||
            !isSupportedStrongLoadAlignment(
                Operation.ValueType,
                Operation.Alignment) ||
            !isSupportedRaceComparisonType(Operation.ValueType)) {
            ++Summary.UnsupportedType;
            continue;
        }

        llvm::Value *ExpectedValue = getExpectedValue(Operation);
        if (ExpectedValue == nullptr ||
            ExpectedValue->getType() != Operation.ValueType) {
            ++Summary.InvalidComparison;
            continue;
        }

        llvm::Instruction *ContinueAt = Operation.Origin->getNextNode();
        if (ContinueAt == nullptr) {
            ++Summary.InvalidInsertionPoint;
            continue;
        }

        InstrumentedMemoryOperation Result;
        Result.WeakOperation = Operation;
        Result.RaceType = getRaceType(Operation);
        Result.ExpectedValue = ExpectedValue;

        RuntimeDelay Delay = insertRuntimeDelay(
            *ContinueAt,
            getMaximumDelay(Operation, Config),
            Operation.SiteID,
            Operation.Origin->getDebugLoc());
        Result.DelayValue = Delay.DelayNs;
        Result.SleepCall = Delay.SleepCall;

        Result.StrongLoad = createStrongSystemLoad(
            *ContinueAt,
            Operation.ValueType,
            Operation.Pointer,
            Operation.Alignment,
            Operation.Origin->getDebugLoc());

        if (Result.StrongLoad == nullptr) {
            ++Summary.UnsupportedType;
            continue;
        }

        RaceAssertResult Assert = insertRaceAssert(
            *ContinueAt,
            ExpectedValue,
            Result.StrongLoad,
            Operation.Origin->getDebugLoc());

        if (Assert.Mismatch == nullptr || Assert.TrapCall == nullptr) {
            ++Summary.InvalidComparison;
            continue;
        }

        Result.Mismatch = Assert.Mismatch;

        SCErrorInsertion Report = insertSCErrorReport(
            Assert,
            Operation,
            Result.RaceType,
            Operation.Origin->getDebugLoc());

        if (Report.ReportCall == nullptr || Report.ContinueBranch == nullptr) {
            // race_assert left the trap fallback intact if reporting insertion
            // failed, so the IR remains valid and still fails safely.
            ++Summary.InvalidReporting;
            continue;
        }

        Result.ReportCall = Report.ReportCall;

        Instrumented.push_back(Result);
        updateInstrumentationSummary(Summary, Operation, Result);
        Changed = true;
    }

    return Changed;
}

} // namespace supercollider
