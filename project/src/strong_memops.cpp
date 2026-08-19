#include "strong_memops.h"

#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"

namespace supercollider {

namespace {

std::uint64_t getRequiredAtomicAlignment(const llvm::Type *Type) {
    if (Type == nullptr)
        return 0;

    if (auto *IntegerType = llvm::dyn_cast<llvm::IntegerType>(Type)) {
        unsigned BitWidth = IntegerType->getBitWidth();
        if (BitWidth == 8 || BitWidth == 16 ||
            BitWidth == 32 || BitWidth == 64) {
            return BitWidth / 8;
        }
    }

    if (Type->isFloatTy())
        return 4;
    if (Type->isDoubleTy())
        return 8;

    return 0;
}

} // namespace

bool isSupportedStrongLoadType(const llvm::Type *Type) {
    if (Type == nullptr)
        return false;

    return Type->isIntegerTy(8) ||
           Type->isIntegerTy(16) ||
           Type->isIntegerTy(32) ||
           Type->isIntegerTy(64) ||
           Type->isFloatTy() ||
           Type->isDoubleTy();
}

bool isSupportedStrongLoadAlignment(
    const llvm::Type *Type,
    llvm::Align Alignment) {

    std::uint64_t Required = getRequiredAtomicAlignment(Type);
    return Required != 0 && Alignment.value() >= Required;
}

llvm::LoadInst *createStrongSystemLoad(
    llvm::Instruction &InsertBefore,
    llvm::Type *ValueType,
    llvm::Value *Pointer,
    llvm::Align Alignment,
    const llvm::DebugLoc &DebugLocation) {

    if (!isSupportedStrongLoadType(ValueType) ||
        !isSupportedStrongLoadAlignment(ValueType, Alignment) ||
        Pointer == nullptr) {
        return nullptr;
    }

    llvm::IRBuilder<> Builder(&InsertBefore);
    llvm::LoadInst *StrongLoad = Builder.CreateAlignedLoad(
        ValueType,
        Pointer,
        Alignment,
        false,
        "sc.strong.load");

    // Monotonic is LLVM's weakest atomic ordering. Combined with the system
    // synchronization scope, NVPTX lowers this duplicate read to a strong
    // system-scoped relaxed load rather than an ordinary weak load.
    StrongLoad->setAtomic(
        llvm::AtomicOrdering::Monotonic,
        llvm::SyncScope::System);

    StrongLoad->setDebugLoc(DebugLocation);
    return StrongLoad;
}

} // namespace supercollider
