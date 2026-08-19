#pragma once

#include "internal_defns.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Alignment.h"

#include <cstdint>

namespace llvm {
class Instruction;
class LoadInst;
class Module;
class StoreInst;
class Type;
class Value;
}

namespace supercollider {

// Canonical description of one ordinary weak memory operation accepted by the
// filter. This record is shared by diagnostics and instrumentation so the pass
// classifies each original memory operation only once.
struct WeakMemoryOperation {
    // Stable within one pass invocation and assigned before any IR mutation.
    // The delay generator uses this ID to distinguish static instrumentation
    // sites even when they execute in the same CUDA thread.
    std::uint64_t SiteID = 0;

    llvm::Instruction *Origin = nullptr;
    llvm::Value *Pointer = nullptr;
    llvm::Type *ValueType = nullptr;
    llvm::Align Alignment = llvm::Align(1);
    SCMemorySpace Space = SCMemorySpace::Unknown;
    SCMemoryOpKind Kind = SCMemoryOpKind::Load;

    bool isLoad() const {
        return Kind == SCMemoryOpKind::Load;
    }

    bool isStore() const {
        return Kind == SCMemoryOpKind::Store;
    }
};

// SuperCollider instruments ordinary weak accesses in global, shared, and
// generic memory. Proven local and constant accesses are excluded here.
bool isInstrumentableSpace(SCMemorySpace Space);

// Check whether an LLVM load is an ordinary weak load supported by the current
// memory-access policy. Atomic and volatile loads are strong/special operations
// and are excluded.
bool isInstrumentableLoad(
    const llvm::LoadInst &LI,
    SCMemorySpace &Space);

// Check whether an LLVM store is an ordinary weak store supported by the
// current memory-access policy. Atomic and volatile stores are strong/special
// operations and are excluded.
bool isInstrumentableStore(
    const llvm::StoreInst &SI,
    SCMemorySpace &Space);

// Classify I as an instrumentable weak load/store and populate Operation when
// accepted. Other instruction kinds return false.
bool getInstrumentableWeakMemoryOperation(
    llvm::Instruction &I,
    WeakMemoryOperation &Operation);

// Collect all original weak-memory candidates before the IR is mutated. This
// prevents instrumentation inserted by later stages from becoming candidates
// in the same pass invocation.
void collectInstrumentableWeakMemoryOperations(
    llvm::Module &M,
    llvm::SmallVectorImpl<WeakMemoryOperation> &Operations);

} // namespace supercollider
