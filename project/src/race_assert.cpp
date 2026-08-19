#include "race_assert.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

namespace supercollider {

bool isSupportedRaceComparisonType(const llvm::Type *Type) {
    if (Type == nullptr)
        return false;

    return Type->isIntegerTy(8) ||
           Type->isIntegerTy(16) ||
           Type->isIntegerTy(32) ||
           Type->isIntegerTy(64) ||
           Type->isFloatTy() ||
           Type->isDoubleTy();
}

RaceAssertResult insertRaceAssertFromMismatch(
    llvm::Instruction &ContinueAt,
    llvm::Value *Mismatch,
    const llvm::DebugLoc &DebugLocation) {

    RaceAssertResult Result;

    if (Mismatch == nullptr || !Mismatch->getType()->isIntegerTy(1))
        return Result;

    Result.Mismatch = Mismatch;

    llvm::BasicBlock *SourceBlock = ContinueAt.getParent();
    if (SourceBlock == nullptr)
        return RaceAssertResult{};

    llvm::Function *F = SourceBlock->getParent();
    llvm::Module *M = F != nullptr ? F->getParent() : nullptr;
    if (F == nullptr || M == nullptr)
        return RaceAssertResult{};

    Result.ContinuationBlock = SourceBlock->splitBasicBlock(
        &ContinueAt,
        "sc.race.continue");

    Result.FailureBlock = llvm::BasicBlock::Create(
        F->getContext(),
        "sc.race.fail",
        F,
        Result.ContinuationBlock);

    // splitBasicBlock installs an unconditional branch to the continuation.
    // Replace it with the detector's conditional branch.
    SourceBlock->getTerminator()->eraseFromParent();

    llvm::IRBuilder<> BranchBuilder(SourceBlock);
    BranchBuilder.SetCurrentDebugLocation(DebugLocation);
    Result.Branch = BranchBuilder.CreateCondBr(
        Result.Mismatch,
        Result.FailureBlock,
        Result.ContinuationBlock);

    llvm::IRBuilder<> FailureBuilder(Result.FailureBlock);
    FailureBuilder.SetCurrentDebugLocation(DebugLocation);

    llvm::Function *Trap = llvm::Intrinsic::getOrInsertDeclaration(
        M,
        llvm::Intrinsic::trap);
    Result.TrapCall = FailureBuilder.CreateCall(Trap);
    Result.TrapCall->setDebugLoc(DebugLocation);
    FailureBuilder.CreateUnreachable();

    return Result;
}

RaceAssertResult insertRaceAssert(
    llvm::Instruction &ContinueAt,
    llvm::Value *ExpectedValue,
    llvm::Value *ObservedValue,
    const llvm::DebugLoc &DebugLocation) {

    if (ExpectedValue == nullptr || ObservedValue == nullptr)
        return RaceAssertResult{};

    llvm::Type *Type = ExpectedValue->getType();
    if (Type != ObservedValue->getType() ||
        !isSupportedRaceComparisonType(Type)) {
        return RaceAssertResult{};
    }

    llvm::IRBuilder<> CompareBuilder(&ContinueAt);
    CompareBuilder.SetCurrentDebugLocation(DebugLocation);

    llvm::Value *Mismatch = nullptr;
    if (Type->isIntegerTy()) {
        Mismatch = CompareBuilder.CreateICmpNE(
            ExpectedValue,
            ObservedValue,
            "sc.race.mismatch");
    } else {
        // Ordered not-equal deliberately treats comparisons involving NaN as
        // non-reporting. This favors possible false negatives over introducing
        // a false race report solely from IEEE unordered comparison semantics.
        Mismatch = CompareBuilder.CreateFCmpONE(
            ExpectedValue,
            ObservedValue,
            "sc.race.mismatch");
    }

    return insertRaceAssertFromMismatch(
        ContinueAt,
        Mismatch,
        DebugLocation);
}

} // namespace supercollider
