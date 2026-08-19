#include "fnv1a_hash.h"

#include "internal_defns.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"

#include <cstdint>

namespace supercollider {

namespace {

constexpr const char *SC_FNV1A_GLOBAL_HELPER = "__sc_fnv1a_hash_global";
constexpr std::uint64_t FNV1A64_OFFSET_BASIS = 14695981039346656037ULL;
constexpr std::uint64_t FNV1A64_PRIME = 1099511628211ULL;

llvm::Function *getOrCreateFNV1aGlobalHelper(llvm::Module &M) {
    if (llvm::Function *Existing = M.getFunction(SC_FNV1A_GLOBAL_HELPER))
        return Existing;

    llvm::LLVMContext &Context = M.getContext();
    llvm::Type *I8 = llvm::Type::getInt8Ty(Context);
    llvm::Type *I32 = llvm::Type::getInt32Ty(Context);
    llvm::Type *I64 = llvm::Type::getInt64Ty(Context);
    llvm::PointerType *GlobalPtr = llvm::PointerType::get(
        Context,
        SC_GLOBAL_ADDRESS_SPACE);

    llvm::FunctionType *FunctionType = llvm::FunctionType::get(
        I64,
        {GlobalPtr, I32},
        false);

    llvm::Function *F = llvm::Function::Create(
        FunctionType,
        llvm::GlobalValue::InternalLinkage,
        SC_FNV1A_GLOBAL_HELPER,
        &M);

    auto Argument = F->arg_begin();
    llvm::Argument *Base = &*Argument++;
    llvm::Argument *ByteCount = &*Argument;
    Base->setName("base");
    ByteCount->setName("byte_count");

    llvm::BasicBlock *Entry = llvm::BasicBlock::Create(
        Context,
        "entry",
        F);
    llvm::BasicBlock *Loop = llvm::BasicBlock::Create(
        Context,
        "loop",
        F);
    llvm::BasicBlock *Body = llvm::BasicBlock::Create(
        Context,
        "body",
        F);
    llvm::BasicBlock *Exit = llvm::BasicBlock::Create(
        Context,
        "exit",
        F);

    llvm::IRBuilder<> EntryBuilder(Entry);
    EntryBuilder.CreateBr(Loop);

    llvm::IRBuilder<> LoopBuilder(Loop);
    llvm::PHINode *Index = LoopBuilder.CreatePHI(I32, 2, "sc.fnv1a.index");
    llvm::PHINode *Hash = LoopBuilder.CreatePHI(I64, 2, "sc.fnv1a.hash");
    Index->addIncoming(LoopBuilder.getInt32(0), Entry);
    Hash->addIncoming(LoopBuilder.getInt64(FNV1A64_OFFSET_BASIS), Entry);

    llvm::Value *Continue = LoopBuilder.CreateICmpULT(
        Index,
        ByteCount,
        "sc.fnv1a.continue");
    LoopBuilder.CreateCondBr(Continue, Body, Exit);

    llvm::IRBuilder<> BodyBuilder(Body);
    llvm::Value *BytePointer = BodyBuilder.CreateGEP(
        I8,
        Base,
        Index,
        "sc.fnv1a.byte.ptr");

    llvm::LoadInst *Byte = BodyBuilder.CreateAlignedLoad(
        I8,
        BytePointer,
        llvm::Align(1),
        false,
        "sc.fnv1a.byte");
    Byte->setAtomic(
        llvm::AtomicOrdering::Monotonic,
        llvm::SyncScope::System);

    llvm::Value *Byte64 = BodyBuilder.CreateZExt(
        Byte,
        I64,
        "sc.fnv1a.byte64");

    // FNV-1a ordering is XOR first, then multiply.
    llvm::Value *Xored = BodyBuilder.CreateXor(
        Hash,
        Byte64,
        "sc.fnv1a.xor");
    llvm::Value *NextHash = BodyBuilder.CreateMul(
        Xored,
        BodyBuilder.getInt64(FNV1A64_PRIME),
        "sc.fnv1a.mul");
    llvm::Value *NextIndex = BodyBuilder.CreateAdd(
        Index,
        BodyBuilder.getInt32(1),
        "sc.fnv1a.next.index");
    BodyBuilder.CreateBr(Loop);

    Index->addIncoming(NextIndex, Body);
    Hash->addIncoming(NextHash, Body);

    llvm::IRBuilder<> ExitBuilder(Exit);
    ExitBuilder.CreateRet(Hash);

    return F;
}

llvm::Value *asI32(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Value) {

    if (Value == nullptr || !Value->getType()->isIntegerTy())
        return nullptr;

    if (Value->getType()->isIntegerTy(32))
        return Value;

    return Builder.CreateZExtOrTrunc(
        Value,
        Builder.getInt32Ty(),
        "sc.fnv1a.byte.count");
}

} // namespace

llvm::CallInst *createFNV1aGlobalHash(
    llvm::Instruction &InsertBefore,
    llvm::Value *GlobalBase,
    llvm::Value *ByteCount,
    const llvm::DebugLoc &DebugLocation) {

    if (GlobalBase == nullptr || ByteCount == nullptr)
        return nullptr;

    auto *PointerType = llvm::dyn_cast<llvm::PointerType>(
        GlobalBase->getType());
    if (PointerType == nullptr ||
        PointerType->getAddressSpace() != SC_GLOBAL_ADDRESS_SPACE) {
        return nullptr;
    }

    llvm::Module *M = InsertBefore.getFunction()->getParent();
    if (M == nullptr)
        return nullptr;

    llvm::IRBuilder<> Builder(&InsertBefore);
    Builder.SetCurrentDebugLocation(DebugLocation);

    llvm::Value *Count32 = asI32(Builder, ByteCount);
    if (Count32 == nullptr)
        return nullptr;

    llvm::Function *Helper = getOrCreateFNV1aGlobalHelper(*M);
    llvm::CallInst *Call = Builder.CreateCall(
        Helper->getFunctionType(),
        Helper,
        {GlobalBase, Count32},
        "sc.fnv1a.result");
    Call->setDebugLoc(DebugLocation);
    return Call;
}

} // namespace supercollider
