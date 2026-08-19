#include "async_memops.h"

#include "delay.h"
#include "fnv1a_hash.h"
#include "internal_defns.h"
#include "race_assert.h"
#include "sc_error.h"
#include "strong_memops.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace supercollider {

namespace {

constexpr const char *SC_BULK_DIRTY_HELPER = "__sc_dirty_global_range";
constexpr std::uint8_t SC_BULK_DIRTY_SEED = 0xefu;

bool isSupportedClassicSize(std::uint64_t Bytes) {
    return Bytes == 4 || Bytes == 8 || Bytes == 16;
}

bool isCudaPipelineInternalFunction(const llvm::Function &F) {
    llvm::StringRef Name = F.getName();
    return Name.contains("__pipeline_internal") ||
           Name.contains("CpAsyncChooser");
}

bool getConstantUnsigned(
    llvm::Value *Value,
    std::uint64_t &Result) {

    auto *Constant = llvm::dyn_cast_or_null<llvm::ConstantInt>(Value);
    if (Constant == nullptr)
        return false;

    Result = Constant->getZExtValue();
    return true;
}

bool collectClassicPipelineCall(
    llvm::CallInst &Call,
    AsyncMemoryOperation &Operation) {

    llvm::Function *Callee = Call.getCalledFunction();
    if (Callee == nullptr)
        return false;

    llvm::StringRef CalleeName = Callee->getName();
    if (!CalleeName.contains("__pipeline_memcpy_async") ||
        CalleeName.contains("__pipeline_internal")) {
        return false;
    }

    llvm::Function *Parent = Call.getFunction();
    if (Parent == nullptr ||
        Parent == Callee ||
        isCudaPipelineInternalFunction(*Parent)) {
        return false;
    }

    if (Call.arg_size() < 3)
        return false;

    llvm::Value *Destination = Call.getArgOperand(0);
    llvm::Value *Source = Call.getArgOperand(1);
    if (!Destination->getType()->isPointerTy() ||
        !Source->getType()->isPointerTy()) {
        return false;
    }

    std::uint64_t Bytes = 0;
    if (!getConstantUnsigned(Call.getArgOperand(2), Bytes) ||
        !isSupportedClassicSize(Bytes)) {
        return false;
    }

    // Some CUDA header forms carry a source-size operand for zero fill. This
    // milestone intentionally handles exact 4/8/16-byte copies only.
    if (Call.arg_size() >= 4) {
        std::uint64_t SourceBytes = 0;
        if (!getConstantUnsigned(Call.getArgOperand(3), SourceBytes) ||
            SourceBytes != Bytes) {
            return false;
        }
    }

    Operation.Origin = &Call;
    Operation.Destination = Destination;
    Operation.Source = Source;
    Operation.StaticBytes = static_cast<std::uint32_t>(Bytes);
    Operation.Kind = SCAsyncMemoryOpKind::ClassicCopy;
    Operation.OriginForm = SCAsyncOriginForm::PipelineMemcpyCall;
    return true;
}

bool collectClassicInlineAsm(
    llvm::CallInst &Call,
    llvm::InlineAsm &Asm,
    AsyncMemoryOperation &Operation) {

    llvm::StringRef Assembly = Asm.getAsmString();
    bool IsClassic =
        Assembly.contains("cp.async.ca.shared.global") ||
        Assembly.contains("cp.async.cg.shared.global");
    if (!IsClassic)
        return false;

    llvm::Function *Parent = Call.getFunction();
    if (Parent == nullptr || isCudaPipelineInternalFunction(*Parent)) {
        // CUDA's __pipeline_memcpy_async wrappers place the actual inline asm
        // in shared header helpers. Those sites are collected at the caller so
        // one application call site gets one SiteID and useful source metadata.
        return false;
    }

    if (Call.arg_size() < 4)
        return false;

    llvm::Value *Destination = Call.getArgOperand(0);
    llvm::Value *Source = Call.getArgOperand(1);
    if (!Destination->getType()->isIntegerTy() ||
        !Source->getType()->isPointerTy()) {
        return false;
    }

    std::uint64_t Bytes = 0;
    std::uint64_t SourceBytes = 0;
    if (!getConstantUnsigned(Call.getArgOperand(2), Bytes) ||
        !getConstantUnsigned(Call.getArgOperand(3), SourceBytes) ||
        !isSupportedClassicSize(Bytes) ||
        SourceBytes != Bytes) {
        return false;
    }

    Operation.Origin = &Call;
    Operation.Destination = Destination;
    Operation.Source = Source;
    Operation.StaticBytes = static_cast<std::uint32_t>(Bytes);
    Operation.Kind = SCAsyncMemoryOpKind::ClassicCopy;
    Operation.OriginForm = SCAsyncOriginForm::InlineAsm;
    return true;
}

bool collectBulkInlineAsm(
    llvm::CallInst &Call,
    llvm::InlineAsm &Asm,
    AsyncMemoryOperation &Operation) {

    llvm::StringRef Assembly = Asm.getAsmString();
    if (!Assembly.contains(
            "cp.async.bulk.global.shared::cta.bulk_group")) {
        return false;
    }

    if (Call.arg_size() < 3)
        return false;

    llvm::Value *Destination = Call.getArgOperand(0);
    llvm::Value *Source = Call.getArgOperand(1);
    llvm::Value *ByteCount = Call.getArgOperand(2);

    if (!Destination->getType()->isIntegerTy() ||
        !Source->getType()->isIntegerTy() ||
        !ByteCount->getType()->isIntegerTy()) {
        return false;
    }

    Operation.Origin = &Call;
    Operation.Destination = Destination;
    Operation.Source = Source;
    Operation.ByteCount = ByteCount;
    Operation.Kind = SCAsyncMemoryOpKind::BulkSharedToGlobal;
    Operation.OriginForm = SCAsyncOriginForm::InlineAsm;
    return true;
}

llvm::IntegerType *getPointerIntegerType(
    llvm::IRBuilder<> &Builder,
    unsigned AddressSpace) {

    llvm::BasicBlock *InsertBlock = Builder.GetInsertBlock();
    llvm::Function *F = InsertBlock != nullptr ? InsertBlock->getParent() : nullptr;
    llvm::Module *M = F != nullptr ? F->getParent() : nullptr;
    if (M == nullptr)
        return nullptr;

    unsigned PointerBits = M->getDataLayout().getPointerSizeInBits(AddressSpace);
    if (PointerBits == 0)
        return nullptr;

    return llvm::IntegerType::get(Builder.getContext(), PointerBits);
}

llvm::Value *integerAddressToPointer(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Address,
    unsigned AddressSpace,
    llvm::StringRef Name) {

    if (Address == nullptr || !Address->getType()->isIntegerTy())
        return nullptr;

    llvm::IntegerType *NativeInteger = getPointerIntegerType(
        Builder,
        AddressSpace);
    if (NativeInteger == nullptr)
        return nullptr;

    llvm::Value *NativeAddress = Address;
    if (Address->getType() != NativeInteger) {
        NativeAddress = Builder.CreateZExtOrTrunc(
            Address,
            NativeInteger,
            llvm::Twine(Name) + ".native");
    }

    llvm::PointerType *PointerType = llvm::PointerType::get(
        Builder.getContext(),
        AddressSpace);
    return Builder.CreateIntToPtr(NativeAddress, PointerType, Name);
}

llvm::Value *asI32(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Value,
    llvm::StringRef Name) {

    if (Value == nullptr || !Value->getType()->isIntegerTy())
        return nullptr;

    if (Value->getType()->isIntegerTy(32))
        return Value;

    return Builder.CreateZExtOrTrunc(Value, Builder.getInt32Ty(), Name);
}

llvm::Value *asAddressI64(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Address,
    llvm::StringRef Name) {

    if (Address == nullptr)
        return nullptr;

    if (auto *PointerType = llvm::dyn_cast<llvm::PointerType>(
            Address->getType())) {
        llvm::IntegerType *NativeInteger = getPointerIntegerType(
            Builder,
            PointerType->getAddressSpace());
        if (NativeInteger == nullptr)
            return nullptr;

        llvm::Value *NativeAddress = Builder.CreatePtrToInt(
            Address,
            NativeInteger,
            llvm::Twine(Name) + ".native");
        if (NativeInteger->isIntegerTy(64))
            return NativeAddress;

        return Builder.CreateZExtOrTrunc(
            NativeAddress,
            Builder.getInt64Ty(),
            Name);
    }

    if (Address->getType()->isIntegerTy()) {
        if (Address->getType()->isIntegerTy(64))
            return Address;

        return Builder.CreateZExtOrTrunc(
            Address,
            Builder.getInt64Ty(),
            Name);
    }

    return nullptr;
}

llvm::Value *createOffsetPointer(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Base,
    std::uint64_t Offset,
    llvm::StringRef Name) {

    if (Base == nullptr || !Base->getType()->isPointerTy())
        return nullptr;

    if (Offset == 0)
        return Base;

    return Builder.CreateGEP(
        Builder.getInt8Ty(),
        Base,
        Builder.getInt64(Offset),
        Name);
}

llvm::CallInst *createInlineAsmCall(
    llvm::IRBuilder<> &Builder,
    llvm::StringRef Assembly,
    llvm::StringRef Constraints,
    llvm::ArrayRef<llvm::Value *> Arguments,
    const llvm::DebugLoc &DebugLocation) {

    llvm::SmallVector<llvm::Type *, 4> ParameterTypes;
    ParameterTypes.reserve(Arguments.size());
    for (llvm::Value *Argument : Arguments)
        ParameterTypes.push_back(Argument->getType());

    llvm::FunctionType *FunctionType = llvm::FunctionType::get(
        Builder.getVoidTy(),
        ParameterTypes,
        false);
    llvm::InlineAsm *Asm = llvm::InlineAsm::get(
        FunctionType,
        Assembly,
        Constraints,
        true);

    llvm::CallInst *Call = Builder.CreateCall(
        FunctionType,
        Asm,
        Arguments);
    Call->setDebugLoc(DebugLocation);
    return Call;
}

llvm::CallInst *createBulkProxyFence(
    llvm::Instruction &InsertBefore,
    const llvm::DebugLoc &DebugLocation) {

    llvm::IRBuilder<> Builder(&InsertBefore);
    Builder.SetCurrentDebugLocation(DebugLocation);
    return createInlineAsmCall(
        Builder,
        "fence.proxy.async.shared::cta;",
        "~{memory}",
        {},
        DebugLocation);
}

llvm::CallInst *createBulkCommitGroup(
    llvm::Instruction &InsertBefore,
    const llvm::DebugLoc &DebugLocation) {

    llvm::IRBuilder<> Builder(&InsertBefore);
    Builder.SetCurrentDebugLocation(DebugLocation);
    return createInlineAsmCall(
        Builder,
        "cp.async.bulk.commit_group;",
        "",
        {},
        DebugLocation);
}

llvm::CallInst *createBulkFullWaitGroup(
    llvm::Instruction &InsertBefore,
    const llvm::DebugLoc &DebugLocation) {

    llvm::IRBuilder<> Builder(&InsertBefore);
    Builder.SetCurrentDebugLocation(DebugLocation);
    llvm::Value *Zero = Builder.getInt32(0);
    return createInlineAsmCall(
        Builder,
        "cp.async.bulk.wait_group $0;",
        "n,~{memory}",
        {Zero},
        DebugLocation);
}

llvm::CallInst *duplicateBulkCopy(
    llvm::CallInst &Original,
    const llvm::DebugLoc &DebugLocation) {

    auto *Asm = llvm::dyn_cast<llvm::InlineAsm>(
        Original.getCalledOperand());
    if (Asm == nullptr)
        return nullptr;

    llvm::SmallVector<llvm::Value *, 4> Arguments;
    Arguments.reserve(Original.arg_size());
    for (unsigned Index = 0; Index < Original.arg_size(); ++Index)
        Arguments.push_back(Original.getArgOperand(Index));

    llvm::IRBuilder<> Builder(&Original);
    Builder.SetCurrentDebugLocation(DebugLocation);
    llvm::CallInst *Copy = Builder.CreateCall(
        Original.getFunctionType(),
        Asm,
        Arguments);
    Copy->setDebugLoc(DebugLocation);
    return Copy;
}

llvm::Function *getOrCreateBulkDirtyHelper(llvm::Module &M) {
    if (llvm::Function *Existing = M.getFunction(SC_BULK_DIRTY_HELPER))
        return Existing;

    llvm::LLVMContext &Context = M.getContext();
    llvm::Type *I8 = llvm::Type::getInt8Ty(Context);
    llvm::Type *I32 = llvm::Type::getInt32Ty(Context);
    llvm::PointerType *GlobalPtr = llvm::PointerType::get(
        Context,
        SC_GLOBAL_ADDRESS_SPACE);

    llvm::FunctionType *FunctionType = llvm::FunctionType::get(
        llvm::Type::getVoidTy(Context),
        {GlobalPtr, I32},
        false);

    llvm::Function *F = llvm::Function::Create(
        FunctionType,
        llvm::GlobalValue::InternalLinkage,
        SC_BULK_DIRTY_HELPER,
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
    llvm::PHINode *Index = LoopBuilder.CreatePHI(I32, 2, "sc.dirty.index");
    Index->addIncoming(LoopBuilder.getInt32(0), Entry);
    llvm::Value *Continue = LoopBuilder.CreateICmpULT(
        Index,
        ByteCount,
        "sc.dirty.continue");
    LoopBuilder.CreateCondBr(Continue, Body, Exit);

    llvm::IRBuilder<> BodyBuilder(Body);
    llvm::Value *BytePointer = BodyBuilder.CreateGEP(
        I8,
        Base,
        Index,
        "sc.dirty.byte.ptr");

    // Vary the sentinel by byte offset so a bulk destination is not replaced
    // by one repeated byte value.
    llvm::Value *Pattern32 = BodyBuilder.CreateXor(
        BodyBuilder.CreateMul(Index, BodyBuilder.getInt32(131)),
        BodyBuilder.getInt32(SC_BULK_DIRTY_SEED),
        "sc.dirty.pattern32");
    llvm::Value *Pattern8 = BodyBuilder.CreateTrunc(
        Pattern32,
        I8,
        "sc.dirty.pattern8");

    llvm::StoreInst *Store = BodyBuilder.CreateAlignedStore(
        Pattern8,
        BytePointer,
        llvm::Align(1));
    Store->setAtomic(
        llvm::AtomicOrdering::Monotonic,
        llvm::SyncScope::System);

    llvm::Value *NextIndex = BodyBuilder.CreateAdd(
        Index,
        BodyBuilder.getInt32(1),
        "sc.dirty.next.index");
    BodyBuilder.CreateBr(Loop);
    Index->addIncoming(NextIndex, Body);

    llvm::IRBuilder<> ExitBuilder(Exit);
    ExitBuilder.CreateRetVoid();

    return F;
}

llvm::CallInst *createBulkDirtification(
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

    llvm::Value *Count32 = asI32(
        Builder,
        ByteCount,
        "sc.dirty.byte.count");
    if (Count32 == nullptr)
        return nullptr;

    llvm::Function *Helper = getOrCreateBulkDirtyHelper(*M);
    llvm::CallInst *Call = Builder.CreateCall(
        Helper->getFunctionType(),
        Helper,
        {GlobalBase, Count32});
    Call->setDebugLoc(DebugLocation);
    return Call;
}

bool instrumentClassicCopy(
    const AsyncMemoryOperation &Operation,
    const SCConfig &Config,
    AsyncInstrumentationSummary &Summary) {

    llvm::CallInst *Origin = Operation.Origin;
    if (Origin == nullptr ||
        Operation.Source == nullptr ||
        Operation.Destination == nullptr ||
        !isSupportedClassicSize(Operation.StaticBytes)) {
        ++Summary.InvalidOperands;
        return false;
    }

    llvm::DebugLoc DebugLocation = Origin->getDebugLoc();
    llvm::IRBuilder<> Builder(Origin);
    Builder.SetCurrentDebugLocation(DebugLocation);

    llvm::Value *SourcePointer = Operation.Source;
    llvm::Value *DestinationPointer = nullptr;

    if (!SourcePointer->getType()->isPointerTy()) {
        ++Summary.InvalidOperands;
        return false;
    }

    if (Operation.OriginForm == SCAsyncOriginForm::PipelineMemcpyCall) {
        if (!Operation.Destination->getType()->isPointerTy()) {
            ++Summary.InvalidOperands;
            return false;
        }
        DestinationPointer = Operation.Destination;
    } else {
        DestinationPointer = integerAddressToPointer(
            Builder,
            Operation.Destination,
            SC_SHARED_ADDRESS_SPACE,
            "sc.async.shared.dst");
    }

    if (DestinationPointer == nullptr) {
        ++Summary.InvalidOperands;
        return false;
    }

    struct Chunk {
        llvm::Type *Type = nullptr;
        std::uint64_t Offset = 0;
        llvm::Align Alignment = llvm::Align(1);
        llvm::Value *SourcePointer = nullptr;
        llvm::Value *DestinationPointer = nullptr;
        llvm::Value *Expected = nullptr;
    };

    llvm::SmallVector<Chunk, 2> Chunks;
    if (Operation.StaticBytes == 4) {
        Chunks.push_back({
            Builder.getInt32Ty(),
            0,
            llvm::Align(4),
            nullptr,
            nullptr,
            nullptr});
    } else if (Operation.StaticBytes == 8) {
        Chunks.push_back({
            Builder.getInt64Ty(),
            0,
            llvm::Align(8),
            nullptr,
            nullptr,
            nullptr});
    } else {
        Chunks.push_back({
            Builder.getInt64Ty(),
            0,
            llvm::Align(8),
            nullptr,
            nullptr,
            nullptr});
        Chunks.push_back({
            Builder.getInt64Ty(),
            8,
            llvm::Align(8),
            nullptr,
            nullptr,
            nullptr});
    }

    // Synchronous emulation of the classic global -> shared async copy.
    for (Chunk &Current : Chunks) {
        Current.SourcePointer = createOffsetPointer(
            Builder,
            SourcePointer,
            Current.Offset,
            "sc.async.src.chunk");
        Current.DestinationPointer = createOffsetPointer(
            Builder,
            DestinationPointer,
            Current.Offset,
            "sc.async.dst.chunk");

        if (Current.SourcePointer == nullptr ||
            Current.DestinationPointer == nullptr) {
            ++Summary.InvalidOperands;
            return false;
        }

        llvm::LoadInst *WeakLoad = Builder.CreateAlignedLoad(
            Current.Type,
            Current.SourcePointer,
            Current.Alignment,
            false,
            "sc.async.weak.src");
        WeakLoad->setDebugLoc(DebugLocation);
        Current.Expected = WeakLoad;

        llvm::StoreInst *WeakStore = Builder.CreateAlignedStore(
            WeakLoad,
            Current.DestinationPointer,
            Current.Alignment);
        WeakStore->setDebugLoc(DebugLocation);
    }

    RuntimeDelay Delay = insertRuntimeDelay(
        *Origin,
        Config.ReadDelayNs,
        Operation.SiteID,
        DebugLocation);
    if (Delay.SleepCall != nullptr)
        ++Summary.DelayCalls;
    else
        ++Summary.ZeroDelaySites;

    llvm::Value *SourceMismatch = Builder.getFalse();
    llvm::Value *DestinationMismatch = Builder.getFalse();

    for (Chunk &Current : Chunks) {
        llvm::LoadInst *StrongSource = createStrongSystemLoad(
            *Origin,
            Current.Type,
            Current.SourcePointer,
            Current.Alignment,
            DebugLocation);
        llvm::LoadInst *StrongDestination = createStrongSystemLoad(
            *Origin,
            Current.Type,
            Current.DestinationPointer,
            Current.Alignment,
            DebugLocation);

        if (StrongSource == nullptr || StrongDestination == nullptr) {
            ++Summary.InvalidOperands;
            return false;
        }

        llvm::IRBuilder<> CompareBuilder(Origin);
        CompareBuilder.SetCurrentDebugLocation(DebugLocation);

        llvm::Value *SourceChunkMismatch = CompareBuilder.CreateICmpNE(
            Current.Expected,
            StrongSource,
            "sc.async.src.mismatch");
        llvm::Value *DestinationChunkMismatch = CompareBuilder.CreateICmpNE(
            Current.Expected,
            StrongDestination,
            "sc.async.dst.mismatch");

        SourceMismatch = CompareBuilder.CreateOr(
            SourceMismatch,
            SourceChunkMismatch,
            "sc.async.src.any.mismatch");
        DestinationMismatch = CompareBuilder.CreateOr(
            DestinationMismatch,
            DestinationChunkMismatch,
            "sc.async.dst.any.mismatch");
    }

    llvm::IRBuilder<> FinalBuilder(Origin);
    FinalBuilder.SetCurrentDebugLocation(DebugLocation);
    llvm::Value *AnyMismatch = FinalBuilder.CreateOr(
        SourceMismatch,
        DestinationMismatch,
        "sc.async.any.mismatch");

    llvm::Value *SourceAddress = asAddressI64(
        FinalBuilder,
        SourcePointer,
        "sc.async.src.address");
    llvm::Value *DestinationAddress = asAddressI64(
        FinalBuilder,
        DestinationPointer,
        "sc.async.dst.address");
    if (SourceAddress == nullptr || DestinationAddress == nullptr) {
        ++Summary.InvalidOperands;
        return false;
    }

    llvm::Value *ReportAddress = FinalBuilder.CreateSelect(
        SourceMismatch,
        SourceAddress,
        DestinationAddress,
        "sc.async.report.address");

    // Dirty the destination immediately before the real cp.async. A later
    // unsynchronized application read is then handled by the ordinary CR path.
    for (Chunk &Current : Chunks) {
        llvm::Constant *Sentinel = nullptr;
        if (Current.Type->isIntegerTy(32)) {
            Sentinel = FinalBuilder.getInt32(0x0badbeefu);
        } else {
            Sentinel = FinalBuilder.getInt64(0x0badbeef0badbeefULL);
        }

        llvm::StoreInst *DirtyStore = FinalBuilder.CreateAlignedStore(
            Sentinel,
            Current.DestinationPointer,
            Current.Alignment);
        DirtyStore->setDebugLoc(DebugLocation);
    }

    RaceAssertResult Assert = insertRaceAssertFromMismatch(
        *Origin,
        AnyMismatch,
        DebugLocation);
    if (Assert.Mismatch == nullptr || Assert.FailureBlock == nullptr) {
        ++Summary.InvalidComparison;
        return false;
    }

    SCErrorInsertion Report = insertSCErrorReportForSite(
        Assert,
        Operation.SiteID,
        ReportAddress,
        SCRaceType::AsyncCopy,
        DebugLocation);
    if (Report.ReportCall == nullptr || Report.ContinueBranch == nullptr) {
        ++Summary.InvalidReporting;
        return true;
    }

    ++Summary.ClassicInstrumented;
    ++Summary.ReportPaths;
    return true;
}

bool instrumentBulkCopy(
    const AsyncMemoryOperation &Operation,
    const SCConfig &Config,
    AsyncInstrumentationSummary &Summary) {

    llvm::CallInst *Origin = Operation.Origin;
    if (Origin == nullptr ||
        Operation.Source == nullptr ||
        Operation.Destination == nullptr ||
        Operation.ByteCount == nullptr ||
        Operation.OriginForm != SCAsyncOriginForm::InlineAsm) {
        ++Summary.InvalidOperands;
        return false;
    }

    llvm::DebugLoc DebugLocation = Origin->getDebugLoc();
    llvm::IRBuilder<> Builder(Origin);
    Builder.SetCurrentDebugLocation(DebugLocation);

    llvm::Value *GlobalDestination = integerAddressToPointer(
        Builder,
        Operation.Destination,
        SC_GLOBAL_ADDRESS_SPACE,
        "sc.bulk.global.dst");
    llvm::Value *ByteCount = asI32(
        Builder,
        Operation.ByteCount,
        "sc.bulk.byte.count");

    if (GlobalDestination == nullptr || ByteCount == nullptr) {
        ++Summary.InvalidOperands;
        return false;
    }

    // The first duplicate is ordered after the application's shared writes.
    // Emit our own proxy fence as well so the instrumentation remains correct
    // even when the original fence is not immediately adjacent to the copy.
    createBulkProxyFence(*Origin, DebugLocation);
    if (duplicateBulkCopy(*Origin, DebugLocation) == nullptr) {
        ++Summary.InvalidOperands;
        return false;
    }
    createBulkCommitGroup(*Origin, DebugLocation);
    createBulkFullWaitGroup(*Origin, DebugLocation);

    llvm::CallInst *Hash0 = createFNV1aGlobalHash(
        *Origin,
        GlobalDestination,
        ByteCount,
        DebugLocation);
    if (Hash0 == nullptr) {
        ++Summary.InvalidOperands;
        return false;
    }

    RuntimeDelay Delay = insertRuntimeDelay(
        *Origin,
        Config.ReadDelayNs,
        Operation.SiteID,
        DebugLocation);
    if (Delay.SleepCall != nullptr)
        ++Summary.DelayCalls;
    else
        ++Summary.ZeroDelaySites;

    createBulkProxyFence(*Origin, DebugLocation);
    if (duplicateBulkCopy(*Origin, DebugLocation) == nullptr) {
        ++Summary.InvalidOperands;
        return false;
    }
    createBulkCommitGroup(*Origin, DebugLocation);
    createBulkFullWaitGroup(*Origin, DebugLocation);

    llvm::CallInst *Hash1 = createFNV1aGlobalHash(
        *Origin,
        GlobalDestination,
        ByteCount,
        DebugLocation);
    if (Hash1 == nullptr) {
        ++Summary.InvalidOperands;
        return false;
    }

    if (createBulkDirtification(
            *Origin,
            GlobalDestination,
            ByteCount,
            DebugLocation) == nullptr) {
        ++Summary.InvalidOperands;
        return false;
    }

    llvm::IRBuilder<> CompareBuilder(Origin);
    CompareBuilder.SetCurrentDebugLocation(DebugLocation);
    llvm::Value *Mismatch = CompareBuilder.CreateICmpNE(
        Hash0,
        Hash1,
        "sc.bulk.hash.mismatch");
    llvm::Value *ReportAddress = asAddressI64(
        CompareBuilder,
        GlobalDestination,
        "sc.bulk.report.address");

    RaceAssertResult Assert = insertRaceAssertFromMismatch(
        *Origin,
        Mismatch,
        DebugLocation);
    if (Assert.Mismatch == nullptr || Assert.FailureBlock == nullptr) {
        ++Summary.InvalidComparison;
        return false;
    }

    SCErrorInsertion Report = insertSCErrorReportForSite(
        Assert,
        Operation.SiteID,
        ReportAddress,
        SCRaceType::AsyncCopy,
        DebugLocation);
    if (Report.ReportCall == nullptr || Report.ContinueBranch == nullptr) {
        ++Summary.InvalidReporting;
        return true;
    }

    ++Summary.BulkInstrumented;
    ++Summary.ReportPaths;
    return true;
}

} // namespace

void collectAsyncMemoryOperations(
    llvm::Module &M,
    std::uint64_t FirstSiteID,
    llvm::SmallVectorImpl<AsyncMemoryOperation> &Operations) {

    std::uint64_t NextSiteID = FirstSiteID;

    for (llvm::Function &F : M) {
        if (F.isDeclaration())
            continue;

        for (llvm::BasicBlock &BB : F) {
            for (llvm::Instruction &I : BB) {
                auto *Call = llvm::dyn_cast<llvm::CallInst>(&I);
                if (Call == nullptr)
                    continue;

                AsyncMemoryOperation Operation;
                bool Accepted = collectClassicPipelineCall(
                    *Call,
                    Operation);

                if (!Accepted) {
                    if (auto *Asm = llvm::dyn_cast<llvm::InlineAsm>(
                            Call->getCalledOperand())) {
                        Accepted = collectBulkInlineAsm(
                            *Call,
                            *Asm,
                            Operation);
                        if (!Accepted) {
                            Accepted = collectClassicInlineAsm(
                                *Call,
                                *Asm,
                                Operation);
                        }
                    }
                }

                if (!Accepted)
                    continue;

                Operation.SiteID = NextSiteID++;
                Operations.push_back(Operation);
            }
        }
    }
}

bool instrumentAsyncMemoryOperations(
    llvm::ArrayRef<AsyncMemoryOperation> Operations,
    const SCConfig &Config,
    AsyncInstrumentationSummary &Summary) {

    for (const AsyncMemoryOperation &Operation : Operations) {
        if (Operation.isClassic())
            ++Summary.ClassicCandidates;
        else if (Operation.isBulkSharedToGlobal())
            ++Summary.BulkCandidates;
    }

    bool Changed = false;

    for (const AsyncMemoryOperation &Operation : Operations) {
        if (Operation.isClassic()) {
            if (!Config.EnableAsyncCopy)
                continue;

            Changed |= instrumentClassicCopy(
                Operation,
                Config,
                Summary);
            continue;
        }

        if (Operation.isBulkSharedToGlobal()) {
            if (!Config.EnableBulkAsyncCopy)
                continue;

            Changed |= instrumentBulkCopy(
                Operation,
                Config,
                Summary);
        }
    }

    return Changed;
}

void printAsyncInstrumentationSummary(
    const AsyncInstrumentationSummary &Summary,
    const SCConfig &Config) {

    if (!Config.EnableAsyncCopy && !Config.EnableBulkAsyncCopy)
        return;

    llvm::errs() << "\n[SuperCollider] Asynchronous-copy instrumentation\n";
    llvm::errs() << "  classic candidates   : "
                 << Summary.ClassicCandidates << "\n";
    llvm::errs() << "  bulk candidates      : "
                 << Summary.BulkCandidates << "\n";
    llvm::errs() << "  classic instrumented : "
                 << Summary.ClassicInstrumented << "\n";
    llvm::errs() << "  bulk instrumented    : "
                 << Summary.BulkInstrumented << "\n";
    llvm::errs() << "  nanosleep calls      : "
                 << Summary.DelayCalls << "\n";
    llvm::errs() << "  zero-delay sites     : "
                 << Summary.ZeroDelaySites << "\n";
    llvm::errs() << "  report paths         : "
                 << Summary.ReportPaths << "\n";
    llvm::errs() << "  invalid operands     : "
                 << Summary.InvalidOperands << "\n";
    llvm::errs() << "  invalid comparisons  : "
                 << Summary.InvalidComparison << "\n";
    llvm::errs() << "  invalid reporting    : "
                 << Summary.InvalidReporting << "\n";
}

} // namespace supercollider
