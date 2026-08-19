#include "filter_memops.h"

#include "parsing_nvvm_ir.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include <cstdint>

namespace supercollider {

bool isInstrumentableSpace(SCMemorySpace Space) {
    return Space == SCMemorySpace::Global ||
           Space == SCMemorySpace::Shared ||
           Space == SCMemorySpace::Generic;
}

bool isInstrumentableLoad(
    const llvm::LoadInst &LI,
    SCMemorySpace &Space) {

    if (LI.isAtomic() || LI.isVolatile())
        return false;

    Space = classifyPointerSpace(LI.getPointerOperand());
    return isInstrumentableSpace(Space);
}

bool isInstrumentableStore(
    const llvm::StoreInst &SI,
    SCMemorySpace &Space) {

    if (SI.isAtomic() || SI.isVolatile())
        return false;

    Space = classifyPointerSpace(SI.getPointerOperand());
    return isInstrumentableSpace(Space);
}

bool getInstrumentableWeakMemoryOperation(
    llvm::Instruction &I,
    WeakMemoryOperation &Operation) {

    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
        SCMemorySpace Space = SCMemorySpace::Unknown;
        if (!isInstrumentableLoad(*LI, Space))
            return false;

        Operation.Origin = LI;
        Operation.Pointer = LI->getPointerOperand();
        Operation.ValueType = LI->getType();
        Operation.Alignment = LI->getAlign();
        Operation.Space = Space;
        Operation.Kind = SCMemoryOpKind::Load;
        return true;
    }

    if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I)) {
        SCMemorySpace Space = SCMemorySpace::Unknown;
        if (!isInstrumentableStore(*SI, Space))
            return false;

        Operation.Origin = SI;
        Operation.Pointer = SI->getPointerOperand();
        Operation.ValueType = SI->getValueOperand()->getType();
        Operation.Alignment = SI->getAlign();
        Operation.Space = Space;
        Operation.Kind = SCMemoryOpKind::Store;
        return true;
    }

    return false;
}

void collectInstrumentableWeakMemoryOperations(
    llvm::Module &M,
    llvm::SmallVectorImpl<WeakMemoryOperation> &Operations) {

    std::uint64_t NextSiteID = 1;

    for (llvm::Function &F : M) {
        if (F.isDeclaration())
            continue;

        for (llvm::BasicBlock &BB : F) {
            for (llvm::Instruction &I : BB) {
                WeakMemoryOperation Operation;
                if (getInstrumentableWeakMemoryOperation(I, Operation)) {
                    Operation.SiteID = NextSiteID++;
                    Operations.push_back(Operation);
                }
            }
        }
    }
}

} // namespace supercollider
