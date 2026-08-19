#include "intra_warp_lost_update.h"

#include "internal_defns.h"
#include "sc_runtime_abi.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace supercollider {

namespace {

constexpr const char *SC_RUNTIME_STORAGE_SYMBOL =
    "__sc_runtime_state_storage";
constexpr const char *SC_ILU_ERROR_CALLBACK_SYMBOL = "sc_err_ilu";

bool hasValidIntraWarpSiteID(std::uint64_t SiteID) {
    return SiteID != 0 && (SiteID & SC_INTRA_WARP_SITE_TAG) == 0;
}

std::uint64_t getIntraWarpReportSiteID(std::uint64_t SiteID) {
    return SiteID | SC_INTRA_WARP_SITE_TAG;
}

llvm::GlobalVariable *getOrCreateRuntimeStorageDeclaration(llvm::Module &M) {
    if (llvm::GlobalVariable *Existing =
            M.getNamedGlobal(SC_RUNTIME_STORAGE_SYMBOL)) {
        return Existing;
    }

    llvm::LLVMContext &Context = M.getContext();
    llvm::ArrayType *StorageType = llvm::ArrayType::get(
        llvm::Type::getInt8Ty(Context),
        SC_RUNTIME_STATE_BYTES);

    return new llvm::GlobalVariable(
        M,
        StorageType,
        false,
        llvm::GlobalValue::ExternalLinkage,
        nullptr,
        SC_RUNTIME_STORAGE_SYMBOL,
        nullptr,
        llvm::GlobalValue::NotThreadLocal,
        SC_GLOBAL_ADDRESS_SPACE);
}

llvm::Value *createActiveMask(
    llvm::IRBuilder<> &Builder,
    const llvm::DebugLoc &DebugLocation) {

    // PTX activemask is the only safe way to obtain the current active-lane
    // set before we know a member mask. Legacy vote.ballot is not supported for
    // sm_70+ in modern PTX, while vote.ballot.sync itself already requires a
    // correct member mask.
    llvm::FunctionType *AsmType = llvm::FunctionType::get(
        Builder.getInt32Ty(),
        false);
    llvm::InlineAsm *ActiveMaskAsm = llvm::InlineAsm::get(
        AsmType,
        "activemask.b32 $0;",
        "=r",
        true);

    llvm::CallInst *Call = Builder.CreateCall(
        AsmType,
        ActiveMaskAsm,
        {},
        "sc.ilu.active.mask");
    Call->setConvergent();
    Call->setDebugLoc(DebugLocation);
    return Call;
}

llvm::Value *createAddressI64(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Pointer) {

    return Builder.CreatePtrToInt(
        Pointer,
        Builder.getInt64Ty(),
        "sc.ilu.address");
}

struct IntraWarpMatchResult {
    llvm::Value *MatchingLanes = nullptr;
    llvm::Value *HasCollision = nullptr;
};

llvm::CallInst *createSCErrorCall(
    llvm::IRBuilder<> &Builder,
    llvm::Module &M,
    const WeakMemoryOperation &Operation,
    llvm::Value *Address,
    llvm::Value *MatchingLanes,
    const llvm::DebugLoc &DebugLocation) {

    llvm::LLVMContext &Context = M.getContext();
    llvm::GlobalVariable *Storage =
        getOrCreateRuntimeStorageDeclaration(M);

    llvm::Value *RuntimePointer = Builder.CreateAddrSpaceCast(
        Storage,
        llvm::PointerType::get(Context, SC_GENERIC_ADDRESS_SPACE),
        "sc.ilu.runtime.ptr");

    llvm::FunctionType *CallbackType = llvm::FunctionType::get(
        Builder.getVoidTy(),
        {
            llvm::PointerType::get(Context, SC_GENERIC_ADDRESS_SPACE),
            Builder.getInt64Ty(),
            Builder.getInt64Ty(),
            Builder.getInt32Ty(),
            Builder.getInt32Ty(),
        },
        false);

    llvm::FunctionCallee Callback = M.getOrInsertFunction(
        SC_ILU_ERROR_CALLBACK_SYMBOL,
        CallbackType);

    llvm::CallInst *Report = Builder.CreateCall(
        Callback,
        {
            RuntimePointer,
            Builder.getInt64(
                getIntraWarpReportSiteID(Operation.SiteID)),
            Address,
            Builder.getInt32(static_cast<std::uint32_t>(
                SCRaceType::IntraWarpLostUpdate)),
            MatchingLanes,
        });
    Report->setDebugLoc(DebugLocation);
    return Report;
}

IntraWarpMatchResult createIntraWarpMatch(
    llvm::IRBuilder<> &Builder,
    llvm::Module &M,
    llvm::Value *MemberMask,
    llvm::Value *Address,
    const llvm::DebugLoc &DebugLocation) {

    IntraWarpMatchResult Result;

    llvm::Function *MatchAny = llvm::Intrinsic::getOrInsertDeclaration(
        &M,
        llvm::Intrinsic::nvvm_match_any_sync_i64);
    llvm::CallInst *MatchingLanes = Builder.CreateCall(
        MatchAny,
        {MemberMask, Address},
        "sc.ilu.match.mask");
    MatchingLanes->setDebugLoc(DebugLocation);
    Result.MatchingLanes = MatchingLanes;

    llvm::Function *CtPop = llvm::Intrinsic::getOrInsertDeclaration(
        &M,
        llvm::Intrinsic::ctpop,
        {Builder.getInt32Ty()});
    llvm::CallInst *MatchingCount = Builder.CreateCall(
        CtPop,
        {MatchingLanes},
        "sc.ilu.match.count");
    MatchingCount->setDebugLoc(DebugLocation);

    Result.HasCollision = Builder.CreateICmpUGT(
        MatchingCount,
        Builder.getInt32(1),
        "sc.ilu.race");

    return Result;
}

llvm::Value *createLeaderPredicate(
    llvm::IRBuilder<> &Builder,
    llvm::Module &M,
    llvm::Value *MatchingLanes,
    const llvm::DebugLoc &DebugLocation) {

    llvm::Function *Cttz = llvm::Intrinsic::getOrInsertDeclaration(
        &M,
        llvm::Intrinsic::cttz,
        {Builder.getInt32Ty()});
    llvm::CallInst *LeaderLane = Builder.CreateCall(
        Cttz,
        {MatchingLanes, Builder.getInt1(true)},
        "sc.ilu.leader.lane");
    LeaderLane->setDebugLoc(DebugLocation);

    llvm::Function *ReadLaneID = llvm::Intrinsic::getOrInsertDeclaration(
        &M,
        llvm::Intrinsic::nvvm_read_ptx_sreg_laneid);
    llvm::CallInst *LaneID = Builder.CreateCall(
        ReadLaneID,
        {},
        "sc.ilu.laneid");
    LaneID->setDebugLoc(DebugLocation);

    return Builder.CreateICmpEQ(
        LaneID,
        LeaderLane,
        "sc.ilu.is.leader");
}

bool instrumentGlobalOrSharedStore(
    const WeakMemoryOperation &Operation,
    llvm::StoreInst &Store,
    IntraWarpLostUpdateSummary &Summary) {

    llvm::BasicBlock *SourceBlock = Store.getParent();
    llvm::Function *F = SourceBlock != nullptr
                            ? SourceBlock->getParent()
                            : nullptr;
    llvm::Module *M = F != nullptr ? F->getParent() : nullptr;
    if (SourceBlock == nullptr || F == nullptr || M == nullptr)
        return false;

    llvm::DebugLoc DebugLocation = Store.getDebugLoc();

    llvm::BasicBlock *ContinuationBlock = SourceBlock->splitBasicBlock(
        &Store,
        "sc.ilu.continue");
    llvm::BasicBlock *ElectionBlock = llvm::BasicBlock::Create(
        F->getContext(),
        "sc.ilu.elect",
        F,
        ContinuationBlock);
    llvm::BasicBlock *ReportBlock = llvm::BasicBlock::Create(
        F->getContext(),
        "sc.ilu.report",
        F,
        ContinuationBlock);

    llvm::Instruction *OldTerminator = SourceBlock->getTerminator();
    llvm::IRBuilder<> DetectBuilder(OldTerminator);
    DetectBuilder.SetCurrentDebugLocation(DebugLocation);

    llvm::Value *ActiveMask = createActiveMask(
        DetectBuilder,
        DebugLocation);
    llvm::Value *Address = createAddressI64(
        DetectBuilder,
        Operation.Pointer);

    IntraWarpMatchResult Match = createIntraWarpMatch(
        DetectBuilder,
        *M,
        ActiveMask,
        Address,
        DebugLocation);

    OldTerminator->eraseFromParent();

    llvm::IRBuilder<> BranchBuilder(SourceBlock);
    BranchBuilder.SetCurrentDebugLocation(DebugLocation);
    BranchBuilder.CreateCondBr(
        Match.HasCollision,
        ElectionBlock,
        ContinuationBlock);

    // Every lane named by Match.MatchingLanes reaches this block. Pick one
    // reporting lane for the same-address group.
    llvm::IRBuilder<> ElectionBuilder(ElectionBlock);
    ElectionBuilder.SetCurrentDebugLocation(DebugLocation);
    llvm::Value *IsLeader = createLeaderPredicate(
        ElectionBuilder,
        *M,
        Match.MatchingLanes,
        DebugLocation);
    ElectionBuilder.CreateCondBr(
        IsLeader,
        ReportBlock,
        ContinuationBlock);

    llvm::IRBuilder<> ReportBuilder(ReportBlock);
    ReportBuilder.SetCurrentDebugLocation(DebugLocation);
    createSCErrorCall(
        ReportBuilder,
        *M,
        Operation,
        Address,
        Match.MatchingLanes,
        DebugLocation);
    ReportBuilder.CreateBr(ContinuationBlock);

    ++Summary.ReportPaths;
    return true;
}

bool instrumentGenericStore(
    const WeakMemoryOperation &Operation,
    llvm::StoreInst &Store,
    IntraWarpLostUpdateSummary &Summary) {

    llvm::BasicBlock *SourceBlock = Store.getParent();
    llvm::Function *F = SourceBlock != nullptr
                            ? SourceBlock->getParent()
                            : nullptr;
    llvm::Module *M = F != nullptr ? F->getParent() : nullptr;
    if (SourceBlock == nullptr || F == nullptr || M == nullptr)
        return false;

    auto *PointerType = llvm::dyn_cast<llvm::PointerType>(
        Operation.Pointer->getType());
    if (PointerType == nullptr ||
        PointerType->getAddressSpace() != SC_GENERIC_ADDRESS_SPACE) {
        return false;
    }

    llvm::DebugLoc DebugLocation = Store.getDebugLoc();

    llvm::BasicBlock *ContinuationBlock = SourceBlock->splitBasicBlock(
        &Store,
        "sc.ilu.continue");
    llvm::BasicBlock *MatchBlock = llvm::BasicBlock::Create(
        F->getContext(),
        "sc.ilu.generic.match",
        F,
        ContinuationBlock);
    llvm::BasicBlock *ElectionBlock = llvm::BasicBlock::Create(
        F->getContext(),
        "sc.ilu.elect",
        F,
        ContinuationBlock);
    llvm::BasicBlock *ReportBlock = llvm::BasicBlock::Create(
        F->getContext(),
        "sc.ilu.report",
        F,
        ContinuationBlock);

    llvm::Instruction *OldTerminator = SourceBlock->getTerminator();
    llvm::IRBuilder<> GuardBuilder(OldTerminator);
    GuardBuilder.SetCurrentDebugLocation(DebugLocation);

    llvm::Value *ActiveMask = createActiveMask(
        GuardBuilder,
        DebugLocation);

    llvm::Function *IsLocalIntrinsic =
        llvm::Intrinsic::getOrInsertDeclaration(
            M,
            llvm::Intrinsic::nvvm_isspacep_local);
    llvm::CallInst *IsLocal = GuardBuilder.CreateCall(
        IsLocalIntrinsic,
        {Operation.Pointer},
        "sc.ilu.is.local");
    IsLocal->setDebugLoc(DebugLocation);

    llvm::Value *Participates = GuardBuilder.CreateNot(
        IsLocal,
        "sc.ilu.participates");

    // Every currently active lane executes this ballot with the same active
    // member mask. Its predicate removes generic pointers that resolve to
    // thread-private local memory. Only the resulting participant lanes are
    // permitted to execute match.any.sync below.
    llvm::Function *BallotSync = llvm::Intrinsic::getOrInsertDeclaration(
        M,
        llvm::Intrinsic::nvvm_vote_ballot_sync);
    llvm::CallInst *ParticipantMask = GuardBuilder.CreateCall(
        BallotSync,
        {ActiveMask, Participates},
        "sc.ilu.participant.mask");
    ParticipantMask->setDebugLoc(DebugLocation);

    llvm::Value *Address = createAddressI64(
        GuardBuilder,
        Operation.Pointer);

    OldTerminator->eraseFromParent();

    llvm::IRBuilder<> GuardBranchBuilder(SourceBlock);
    GuardBranchBuilder.SetCurrentDebugLocation(DebugLocation);
    GuardBranchBuilder.CreateCondBr(
        Participates,
        MatchBlock,
        ContinuationBlock);

    llvm::IRBuilder<> MatchBuilder(MatchBlock);
    MatchBuilder.SetCurrentDebugLocation(DebugLocation);
    IntraWarpMatchResult Match = createIntraWarpMatch(
        MatchBuilder,
        *M,
        ParticipantMask,
        Address,
        DebugLocation);
    MatchBuilder.CreateCondBr(
        Match.HasCollision,
        ElectionBlock,
        ContinuationBlock);

    // Every lane in this same-address group computes the same lowest-lane
    // leader from the exact matching-lane mask returned by match.any.sync.
    llvm::IRBuilder<> ElectionBuilder(ElectionBlock);
    ElectionBuilder.SetCurrentDebugLocation(DebugLocation);
    llvm::Value *IsLeader = createLeaderPredicate(
        ElectionBuilder,
        *M,
        Match.MatchingLanes,
        DebugLocation);
    ElectionBuilder.CreateCondBr(
        IsLeader,
        ReportBlock,
        ContinuationBlock);

    llvm::IRBuilder<> ReportBuilder(ReportBlock);
    ReportBuilder.SetCurrentDebugLocation(DebugLocation);
    createSCErrorCall(
        ReportBuilder,
        *M,
        Operation,
        Address,
        Match.MatchingLanes,
        DebugLocation);
    ReportBuilder.CreateBr(ContinuationBlock);

    ++Summary.GenericLocalGuards;
    ++Summary.ReportPaths;
    return true;
}

} // namespace

bool instrumentIntraWarpLostUpdates(
    llvm::ArrayRef<WeakMemoryOperation> Operations,
    IntraWarpLostUpdateSummary &Summary) {

    bool Changed = false;

    for (const WeakMemoryOperation &Operation : Operations) {
        if (!Operation.isStore())
            continue;

        ++Summary.StoresConsidered;

        if (Operation.Origin == nullptr || Operation.Pointer == nullptr) {
            ++Summary.InvalidInsertionPoint;
            continue;
        }

        auto *Store = llvm::dyn_cast<llvm::StoreInst>(Operation.Origin);
        if (Store == nullptr || Store->getParent() == nullptr) {
            ++Summary.InvalidInsertionPoint;
            continue;
        }

        if (!hasValidIntraWarpSiteID(Operation.SiteID)) {
            ++Summary.InvalidSiteID;
            continue;
        }

        bool Instrumented = false;
        switch (Operation.Space) {
        case SCMemorySpace::Global:
            Instrumented = instrumentGlobalOrSharedStore(
                Operation,
                *Store,
                Summary);
            if (Instrumented)
                ++Summary.Global;
            break;

        case SCMemorySpace::Shared:
            Instrumented = instrumentGlobalOrSharedStore(
                Operation,
                *Store,
                Summary);
            if (Instrumented)
                ++Summary.Shared;
            break;

        case SCMemorySpace::Generic:
            Instrumented = instrumentGenericStore(
                Operation,
                *Store,
                Summary);
            if (Instrumented)
                ++Summary.Generic;
            break;

        default:
            break;
        }

        if (!Instrumented) {
            ++Summary.InvalidInsertionPoint;
            continue;
        }

        ++Summary.Instrumented;
        Changed = true;
    }

    return Changed;
}

void printIntraWarpLostUpdateSummary(
    const IntraWarpLostUpdateSummary &Summary) {

    llvm::errs() << "\n[SuperCollider] Intra-warp lost-update instrumentation\n";
    llvm::errs() << "  stores considered    : "
                 << Summary.StoresConsidered << "\n";
    llvm::errs() << "  instrumented         : "
                 << Summary.Instrumented << "\n";
    llvm::errs() << "  global               : "
                 << Summary.Global << "\n";
    llvm::errs() << "  shared               : "
                 << Summary.Shared << "\n";
    llvm::errs() << "  generic              : "
                 << Summary.Generic << "\n";
    llvm::errs() << "  generic local guards : "
                 << Summary.GenericLocalGuards << "\n";
    llvm::errs() << "  report paths         : "
                 << Summary.ReportPaths << "\n";
    llvm::errs() << "  invalid insert point : "
                 << Summary.InvalidInsertionPoint << "\n";
    llvm::errs() << "  invalid site id      : "
                 << Summary.InvalidSiteID << "\n";
}

} // namespace supercollider
