#include "block_shuffle.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <string>

namespace supercollider {

namespace {

constexpr llvm::StringLiteral SC_BLOCK_SHUFFLE_PHYSICAL_MD =
    "supercollider.block_shuffle.physical";

constexpr std::array<std::uint64_t, 8> SC_SHUFFLE_PRIMES = {
    3ULL,
    5ULL,
    7ULL,
    11ULL,
    13ULL,
    17ULL,
    19ULL,
    23ULL,
};

llvm::Value *readNVVMSpecialRegister(
    llvm::IRBuilder<> &Builder,
    llvm::Module &M,
    llvm::Intrinsic::ID IntrinsicID,
    llvm::StringRef Name) {

    llvm::Function *Declaration =
        llvm::Intrinsic::getOrInsertDeclaration(&M, IntrinsicID);
    return Builder.CreateCall(Declaration, {}, Name);
}

llvm::Value *asI64(llvm::IRBuilder<> &Builder, llvm::Value *Value) {
    if (Value->getType()->isIntegerTy(64))
        return Value;

    return Builder.CreateZExtOrTrunc(
        Value,
        Builder.getInt64Ty());
}

// Stateless SplitMix64 finalizer. Arithmetic wraps modulo 2^64 by design.
llvm::Value *finalizeSplitMix64(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Input,
    llvm::StringRef Name) {

    llvm::Value *X = Input;

    X = Builder.CreateXor(
        X,
        Builder.CreateLShr(X, Builder.getInt64(30)));
    X = Builder.CreateMul(
        X,
        Builder.getInt64(0xbf58476d1ce4e5b9ULL));

    X = Builder.CreateXor(
        X,
        Builder.CreateLShr(X, Builder.getInt64(27)));
    X = Builder.CreateMul(
        X,
        Builder.getInt64(0x94d049bb133111ebULL));

    return Builder.CreateXor(
        X,
        Builder.CreateLShr(X, Builder.getInt64(31)),
        Name);
}

llvm::Value *selectPrimeMultiplier(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Hash) {

    static_assert(
        (SC_SHUFFLE_PRIMES.size() & (SC_SHUFFLE_PRIMES.size() - 1)) == 0,
        "shuffle prime table size must be a power of two");

    llvm::Value *Index = Builder.CreateAnd(
        Hash,
        Builder.getInt64(SC_SHUFFLE_PRIMES.size() - 1ULL),
        "sc.shuffle.prime.index");

    llvm::Value *Selected = Builder.getInt64(SC_SHUFFLE_PRIMES[0]);

    for (std::size_t I = 1; I < SC_SHUFFLE_PRIMES.size(); ++I) {
        llvm::Value *IsIndex = Builder.CreateICmpEQ(
            Index,
            Builder.getInt64(I));
        Selected = Builder.CreateSelect(
            IsIndex,
            Builder.getInt64(SC_SHUFFLE_PRIMES[I]),
            Selected,
            "sc.shuffle.prime.select");
    }

    return Selected;
}

bool isPhysicalShuffleRead(const llvm::CallInst &Call) {
    return Call.getMetadata(SC_BLOCK_SHUFFLE_PHYSICAL_MD) != nullptr;
}

bool isCTAIDXRead(const llvm::CallInst &Call) {
    const llvm::Function *Callee = Call.getCalledFunction();
    if (Callee == nullptr || !Callee->isIntrinsic())
        return false;

    return Callee->getIntrinsicID() ==
           llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x;
}

bool inlineAsmReadsCTAIDX(const llvm::CallBase &Call) {
    const llvm::Value *Called = Call.getCalledOperand();
    if (Called == nullptr)
        return false;

    Called = Called->stripPointerCasts();
    const auto *Asm = llvm::dyn_cast<llvm::InlineAsm>(Called);
    if (Asm == nullptr)
        return false;

    return Asm->getAsmString().contains("ctaid.x");
}

llvm::Value *createLogicalBlockX(
    llvm::CallInst &OriginalRead,
    const SCConfig &Config) {

    llvm::Module &M = *OriginalRead.getFunction()->getParent();
    llvm::IRBuilder<> Builder(&OriginalRead);
    Builder.SetCurrentDebugLocation(OriginalRead.getDebugLoc());

    llvm::Function *CTAIDDeclaration =
        llvm::Intrinsic::getOrInsertDeclaration(
            &M,
            llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x);

    auto *Physical = llvm::cast<llvm::CallInst>(
        Builder.CreateCall(
            CTAIDDeclaration,
            {},
            "sc.shuffle.physical.x"));
    Physical->setMetadata(
        SC_BLOCK_SHUFFLE_PHYSICAL_MD,
        llvm::MDNode::get(
            M.getContext(),
            llvm::ArrayRef<llvm::Metadata *>{}));

    llvm::Value *GridDimX = readNVVMSpecialRegister(
        Builder,
        M,
        llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x,
        "sc.shuffle.grid.x");

    llvm::Value *GridID = readNVVMSpecialRegister(
        Builder,
        M,
        llvm::Intrinsic::nvvm_read_ptx_sreg_gridid,
        "sc.shuffle.gridid");

    llvm::Value *Physical64 = asI64(Builder, Physical);
    llvm::Value *GridDim64 = asI64(Builder, GridDimX);
    llvm::Value *GridID64 = asI64(Builder, GridID);

    llvm::Value *ShuffleEnabled = Builder.CreateICmpUGT(
        GridDim64,
        Builder.getInt64(1),
        "sc.shuffle.active");

    // Keep the generated modulo operations defined even if malformed input IR
    // claims a zero-sized grid. Valid CUDA launches always have gridDim.x >= 1.
    llvm::Value *SafeGridDim = Builder.CreateSelect(
        ShuffleEnabled,
        GridDim64,
        Builder.getInt64(1),
        "sc.shuffle.safe.grid.x");

    llvm::Value *Seed = Builder.CreateXor(
        GridID64,
        Builder.getInt64(Config.BlockShuffleSeed),
        "sc.shuffle.seed");

    llvm::Value *MultiplierHash = finalizeSplitMix64(
        Builder,
        Builder.CreateXor(
            Seed,
            Builder.getInt64(0x243f6a8885a308d3ULL)),
        "sc.shuffle.a.hash");

    llvm::Value *OffsetHash = finalizeSplitMix64(
        Builder,
        Builder.CreateXor(
            Seed,
            Builder.getInt64(0x13198a2e03707344ULL)),
        "sc.shuffle.b.hash");

    llvm::Value *Prime = selectPrimeMultiplier(
        Builder,
        MultiplierHash);

    // If prime p does not divide N, gcd(p, N) is one. Otherwise use N-1,
    // which is always coprime with N for N > 1. Therefore every generated
    // multiplier is valid without a device-side Euclidean-GCD loop.
    llvm::Value *PrimeDividesGrid = Builder.CreateICmpEQ(
        Builder.CreateURem(
            SafeGridDim,
            Prime,
            "sc.shuffle.grid.mod.prime"),
        Builder.getInt64(0),
        "sc.shuffle.prime.divides");

    llvm::Value *FallbackMultiplier = Builder.CreateSub(
        SafeGridDim,
        Builder.getInt64(1),
        "sc.shuffle.a.fallback");

    llvm::Value *Multiplier = Builder.CreateSelect(
        PrimeDividesGrid,
        FallbackMultiplier,
        Prime,
        "sc.shuffle.a");

    llvm::Value *Offset = Builder.CreateURem(
        OffsetHash,
        SafeGridDim,
        "sc.shuffle.b");

    llvm::Value *Affine = Builder.CreateAdd(
        Builder.CreateMul(
            Multiplier,
            Physical64,
            "sc.shuffle.ax"),
        Offset,
        "sc.shuffle.ax.plus.b");

    llvm::Value *Logical64 = Builder.CreateURem(
        Affine,
        SafeGridDim,
        "sc.shuffle.logical.x64");

    llvm::Value *Logical32 = Builder.CreateTrunc(
        Logical64,
        Physical->getType(),
        "sc.shuffle.logical.x32");

    return Builder.CreateSelect(
        ShuffleEnabled,
        Logical32,
        Physical,
        "sc.shuffle.logical.x");
}

} // namespace

bool collectBlockShuffleCandidates(
    llvm::Module &M,
    llvm::SmallVectorImpl<llvm::CallInst *> &Candidates,
    BlockShuffleSummary &Summary,
    std::string &ErrorMessage) {

    ErrorMessage.clear();
    llvm::SmallPtrSet<llvm::Function *, 16> Functions;

    for (llvm::Function &F : M) {
        if (F.isDeclaration())
            continue;

        for (llvm::BasicBlock &BB : F) {
            for (llvm::Instruction &I : BB) {
                auto *Call = llvm::dyn_cast<llvm::CallBase>(&I);
                if (Call != nullptr && inlineAsmReadsCTAIDX(*Call))
                    ++Summary.InlineAsmCTAIDReferences;

                auto *CallInst = llvm::dyn_cast<llvm::CallInst>(&I);
                if (CallInst == nullptr ||
                    isPhysicalShuffleRead(*CallInst) ||
                    !isCTAIDXRead(*CallInst)) {
                    continue;
                }

                Candidates.push_back(CallInst);
                Functions.insert(&F);
            }
        }
    }

    Summary.BlockIndexReads = Candidates.size();
    Summary.Functions = Functions.size();

    if (Summary.InlineAsmCTAIDReferences != 0) {
        ErrorMessage =
            "block shuffling cannot safely rewrite this module because "
            "handwritten inline PTX reads %ctaid.x; all application "
            "blockIdx.x reads must use the same logical affine mapping";
        return false;
    }

    return true;
}

bool instrumentBlockShuffling(
    llvm::ArrayRef<llvm::CallInst *> Candidates,
    const SCConfig &Config,
    BlockShuffleSummary &Summary) {

    if (!Config.EnableBlockShuffle)
        return false;

    for (llvm::CallInst *OriginalRead : Candidates) {
        if (OriginalRead == nullptr || OriginalRead->getParent() == nullptr)
            continue;

        llvm::Value *LogicalBlockX = createLogicalBlockX(
            *OriginalRead,
            Config);
        if (LogicalBlockX == nullptr)
            continue;

        OriginalRead->replaceAllUsesWith(LogicalBlockX);
        OriginalRead->eraseFromParent();
        ++Summary.Rewritten;
    }

    return Summary.Rewritten != 0;
}

void printBlockShuffleSummary(
    const BlockShuffleSummary &Summary,
    const SCConfig &Config) {

    if (!Config.EnableBlockShuffle)
        return;

    llvm::errs() << "\n[SuperCollider] Block shuffling\n";
    llvm::errs() << "  blockIdx.x reads     : "
                 << Summary.BlockIndexReads << "\n";
    llvm::errs() << "  rewritten            : "
                 << Summary.Rewritten << "\n";
    llvm::errs() << "  functions            : "
                 << Summary.Functions << "\n";
    llvm::errs() << "  inline-asm ctaid.x   : "
                 << Summary.InlineAsmCTAIDReferences << "\n";
    llvm::errs() << "  shuffle seed         : "
                 << Config.BlockShuffleSeed << "\n";
}

} // namespace supercollider
