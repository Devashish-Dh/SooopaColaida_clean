#include "debugging_prints.h"

#include "parsing_nvvm_ir.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

namespace supercollider {

namespace {

llvm::StringRef getMemorySpaceName(SCMemorySpace Space) {
    switch (Space) {
    case SCMemorySpace::Generic:
        return "generic";
    case SCMemorySpace::Global:
        return "global";
    case SCMemorySpace::Shared:
        return "shared";
    case SCMemorySpace::Constant:
        return "constant";
    case SCMemorySpace::Local:
        return "local";
    case SCMemorySpace::Unknown:
        return "unknown";
    }

    return "unknown";
}

llvm::StringRef getMemoryOperationName(SCMemoryOpKind Kind) {
    switch (Kind) {
    case SCMemoryOpKind::Load:
        return "load";
    case SCMemoryOpKind::Store:
        return "store";
    }

    return "unknown";
}

llvm::StringRef getRaceTypeName(SCRaceType RaceType) {
    switch (RaceType) {
    case SCRaceType::ClobberedRead:
        return "clobbered-read";
    case SCRaceType::LostUpdate:
        return "lost-update";
    case SCRaceType::IntraWarpLostUpdate:
        return "intra-warp-lost-update";
    case SCRaceType::AsyncCopy:
        return "async-copy";
    }

    return "unknown";
}

void printType(const llvm::Type *Type) {
    if (Type == nullptr) {
        llvm::errs() << "<null>";
        return;
    }

    Type->print(llvm::errs());
}

void printValue(const llvm::Value *Value) {
    if (Value != nullptr)
        Value->print(llvm::errs());
    else
        llvm::errs() << "<none>";
}

} // namespace

void updateCandidateSummary(
    CandidateSummary &Summary,
    const WeakMemoryOperation &Operation) {

    bool IsLoad = Operation.isLoad();

    if (IsLoad)
        ++Summary.Loads;
    else
        ++Summary.Stores;

    switch (Operation.Space) {
    case SCMemorySpace::Global:
        if (IsLoad)
            ++Summary.GlobalLoads;
        else
            ++Summary.GlobalStores;
        break;

    case SCMemorySpace::Shared:
        if (IsLoad)
            ++Summary.SharedLoads;
        else
            ++Summary.SharedStores;
        break;

    case SCMemorySpace::Generic:
        if (IsLoad)
            ++Summary.GenericLoads;
        else
            ++Summary.GenericStores;
        break;

    default:
        break;
    }
}

void printConfiguration(const SCConfig &Config) {
    llvm::errs() << "\n[SuperCollider] Configuration\n";
    llvm::errs() << "  rdelay max (ns) : " << Config.ReadDelayNs << "\n";
    llvm::errs() << "  wdelay max (ns) : " << Config.WriteDelayNs << "\n";
    llvm::errs() << "  intra-warp LU    : "
                 << (Config.EnableIntraWarpLostUpdate ? "enabled" : "disabled")
                 << "\n";
    llvm::errs() << "  classic cp.async : "
                 << (Config.EnableAsyncCopy ? "enabled" : "disabled")
                 << "\n";
    llvm::errs() << "  cp.async.bulk    : "
                 << (Config.EnableBulkAsyncCopy ? "enabled" : "disabled")
                 << "\n";
    llvm::errs() << "  block shuffle    : "
                 << (Config.EnableBlockShuffle ? "enabled" : "disabled")
                 << "\n";
    if (Config.EnableBlockShuffle)
        llvm::errs() << "  shuffle seed     : "
                     << Config.BlockShuffleSeed << "\n";
}

void printInstrumentableCandidate(
    const llvm::Function &F,
    const WeakMemoryOperation &Operation) {

    llvm::errs() << "\n[SC-CANDIDATE " << Operation.SiteID << "]\n";
    llvm::errs() << "  function : " << F.getName() << "\n";
    llvm::errs() << "  kind     : "
                 << getMemoryOperationName(Operation.Kind) << "\n";
    llvm::errs() << "  space    : "
                 << getMemorySpaceName(Operation.Space) << "\n";

    llvm::errs() << "  type     : ";
    printType(Operation.ValueType);
    llvm::errs() << "\n";

    llvm::errs() << "  pointer  : ";
    printValue(Operation.Pointer);
    llvm::errs() << "\n";

    const llvm::Value *Base = stripPointerDerivations(Operation.Pointer);
    llvm::errs() << "  base     : ";
    printValue(Base);
    llvm::errs() << "\n";

    llvm::errs() << "  llvm-ir  : ";
    if (Operation.Origin != nullptr)
        llvm::errs() << *Operation.Origin;
    else
        llvm::errs() << "<none>";
    llvm::errs() << "\n";
}

void printCandidateSummary(const CandidateSummary &Summary) {
    llvm::errs()
        << "\n[SuperCollider] Instrumentable weak-memory candidates\n";

    llvm::errs() << "  loads          : " << Summary.Loads << "\n";
    llvm::errs() << "  stores         : " << Summary.Stores << "\n";
    llvm::errs() << "  global L/S     : "
                 << Summary.GlobalLoads << " / " << Summary.GlobalStores << "\n";
    llvm::errs() << "  shared L/S     : "
                 << Summary.SharedLoads << " / " << Summary.SharedStores << "\n";
    llvm::errs() << "  generic L/S    : "
                 << Summary.GenericLoads << " / " << Summary.GenericStores << "\n";
}

void printInstrumentedMemoryOperation(
    const InstrumentedMemoryOperation &Operation,
    const SCConfig &Config) {

    const WeakMemoryOperation &Weak = Operation.WeakOperation;
    std::uint32_t MaximumDelay = Weak.isLoad()
                                     ? Config.ReadDelayNs
                                     : Config.WriteDelayNs;

    llvm::errs() << "\n[SC-INSTRUMENTED " << Weak.SiteID << "]\n";
    llvm::errs() << "  origin-kind : "
                 << getMemoryOperationName(Weak.Kind) << "\n";
    llvm::errs() << "  race-type   : "
                 << getRaceTypeName(Operation.RaceType) << "\n";
    llvm::errs() << "  space       : "
                 << getMemorySpaceName(Weak.Space) << "\n";

    llvm::errs() << "  type        : ";
    printType(Weak.ValueType);
    llvm::errs() << "\n";

    llvm::errs() << "  alignment   : " << Weak.Alignment.value() << "\n";
    llvm::errs() << "  delay max   : " << MaximumDelay << " ns\n";
    llvm::errs() << "  ordering    : monotonic\n";
    llvm::errs() << "  sync-scope  : system\n";

    llvm::errs() << "  original    : ";
    if (Weak.Origin != nullptr)
        llvm::errs() << *Weak.Origin;
    else
        llvm::errs() << "<none>";
    llvm::errs() << "\n";

    llvm::errs() << "  expected    : ";
    printValue(Operation.ExpectedValue);
    llvm::errs() << "\n";

    llvm::errs() << "  delay value : ";
    printValue(Operation.DelayValue);
    llvm::errs() << "\n";

    llvm::errs() << "  nanosleep   : ";
    printValue(Operation.SleepCall);
    llvm::errs() << "\n";

    llvm::errs() << "  strong-load : ";
    printValue(Operation.StrongLoad);
    llvm::errs() << "\n";

    llvm::errs() << "  mismatch    : ";
    printValue(Operation.Mismatch);
    llvm::errs() << "\n";

    llvm::errs() << "  sc_err      : ";
    printValue(Operation.ReportCall);
    llvm::errs() << "\n";
}

void printInstrumentationSummary(const InstrumentationSummary &Summary) {
    llvm::errs() << "\n[SuperCollider] Weak-memory instrumentation\n";
    llvm::errs() << "  instrumented total  : " << Summary.Instrumented << "\n";
    llvm::errs() << "  from weak loads     : " << Summary.FromLoads << "\n";
    llvm::errs() << "  from weak stores    : " << Summary.FromStores << "\n";
    llvm::errs() << "  global              : " << Summary.Global << "\n";
    llvm::errs() << "  shared              : " << Summary.Shared << "\n";
    llvm::errs() << "  generic             : " << Summary.Generic << "\n";
    llvm::errs() << "  nanosleep calls     : " << Summary.DelayCalls << "\n";
    llvm::errs() << "  zero-delay sites    : " << Summary.ZeroDelaySites << "\n";
    llvm::errs() << "  race guards         : " << Summary.RaceGuards << "\n";
    llvm::errs() << "  report paths        : " << Summary.ReportPaths << "\n";
    llvm::errs() << "  unsupported type    : " << Summary.UnsupportedType << "\n";
    llvm::errs() << "  invalid insert point: " << Summary.InvalidInsertionPoint << "\n";
    llvm::errs() << "  invalid comparison  : " << Summary.InvalidComparison << "\n";
    llvm::errs() << "  invalid reporting   : " << Summary.InvalidReporting << "\n";
}

} // namespace supercollider
