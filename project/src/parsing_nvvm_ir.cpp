#include "parsing_nvvm_ir.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace supercollider {

SCMemorySpace classifyAddressSpace(unsigned AddressSpace) {
    switch (AddressSpace) {
    case SC_GENERIC_ADDRESS_SPACE:
        return SCMemorySpace::Generic;
    case SC_GLOBAL_ADDRESS_SPACE:
        return SCMemorySpace::Global;
    case SC_SHARED_ADDRESS_SPACE:
        return SCMemorySpace::Shared;
    case SC_CONSTANT_ADDRESS_SPACE:
        return SCMemorySpace::Constant;
    case SC_LOCAL_ADDRESS_SPACE:
        return SCMemorySpace::Local;
    default:
        return SCMemorySpace::Unknown;
    }
}

unsigned getPointerAddressSpace(const llvm::Value *V) {
    if (V == nullptr || !V->getType()->isPointerTy())
        return SC_INVALID_ADDRESS_SPACE;

    return llvm::cast<llvm::PointerType>(V->getType())->getAddressSpace();
}

const llvm::Value *stripPointerDerivations(const llvm::Value *V) {
    llvm::SmallPtrSet<const llvm::Value *, 16> Visited;

    while (V != nullptr && Visited.insert(V).second) {
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(V)) {
            V = GEP->getPointerOperand();
            continue;
        }

        if (auto *BC = llvm::dyn_cast<llvm::BitCastInst>(V)) {
            V = BC->getOperand(0);
            continue;
        }

        if (auto *ASC = llvm::dyn_cast<llvm::AddrSpaceCastInst>(V)) {
            V = ASC->getOperand(0);
            continue;
        }

        if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
            unsigned Opcode = CE->getOpcode();

            if (Opcode == llvm::Instruction::GetElementPtr ||
                Opcode == llvm::Instruction::BitCast ||
                Opcode == llvm::Instruction::AddrSpaceCast) {
                V = CE->getOperand(0);
                continue;
            }
        }

        break;
    }

    return V;
}

namespace {

// Base object plus a fixed constant GEP-index path. Field sensitivity is
// required to recover pointer provenance from local CUDA/libcudacxx helper
// objects without relying on function-name heuristics.
struct PointerAccessPath {
    const llvm::Value *Base = nullptr;
    llvm::SmallVector<std::int64_t, 8> Indices;
    bool Valid = false;
};

bool getPointerAccessPathImpl(
    const llvm::Value *V,
    PointerAccessPath &Path,
    unsigned Depth) {

    if (V == nullptr || Depth > SC_MAX_PROVENANCE_DEPTH)
        return false;

    if (auto *GEP = llvm::dyn_cast<llvm::GEPOperator>(V)) {
        PointerAccessPath Parent;

        if (!getPointerAccessPathImpl(
                GEP->getPointerOperand(), Parent, Depth + 1)) {
            return false;
        }

        for (auto It = GEP->idx_begin(), End = GEP->idx_end(); It != End; ++It) {
            const auto *CI = llvm::dyn_cast<llvm::ConstantInt>(*It);
            if (CI == nullptr)
                return false;

            Parent.Indices.push_back(CI->getSExtValue());
        }

        Path = std::move(Parent);
        Path.Valid = true;
        return true;
    }

    if (auto *BC = llvm::dyn_cast<llvm::BitCastInst>(V))
        return getPointerAccessPathImpl(BC->getOperand(0), Path, Depth + 1);

    if (auto *ASC = llvm::dyn_cast<llvm::AddrSpaceCastInst>(V))
        return getPointerAccessPathImpl(ASC->getOperand(0), Path, Depth + 1);

    if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
        unsigned Opcode = CE->getOpcode();

        if (Opcode == llvm::Instruction::BitCast ||
            Opcode == llvm::Instruction::AddrSpaceCast) {
            return getPointerAccessPathImpl(CE->getOperand(0), Path, Depth + 1);
        }
    }

    Path.Base = V;
    Path.Valid = true;
    return true;
}

bool getPointerAccessPath(
    const llvm::Value *V,
    PointerAccessPath &Path) {

    Path = PointerAccessPath();
    return getPointerAccessPathImpl(V, Path, 0);
}

bool sameIndices(
    llvm::ArrayRef<std::int64_t> A,
    llvm::ArrayRef<std::int64_t> B) {

    if (A.size() != B.size())
        return false;

    for (std::size_t I = 0; I < A.size(); ++I) {
        if (A[I] != B[I])
            return false;
    }

    return true;
}

SCMemorySpace classifyPointerSpaceImpl(
    const llvm::Value *Pointer,
    llvm::SmallPtrSetImpl<const llvm::Value *> &Visiting,
    unsigned Depth);

SCMemorySpace classifyLoadedPointerFromLocalSlot(
    const llvm::LoadInst &LI,
    llvm::SmallPtrSetImpl<const llvm::Value *> &Visiting,
    unsigned Depth);

SCMemorySpace mergePointerSpaces(
    SCMemorySpace A,
    SCMemorySpace B) {

    if (A == B)
        return A;

    if (A == SCMemorySpace::Unknown || B == SCMemorySpace::Unknown)
        return SCMemorySpace::Unknown;

    // Multiple different known spaces are conservatively represented as generic.
    return SCMemorySpace::Generic;
}

// Refine an AS0 argument to Local only when every direct call site proves that
// the corresponding actual argument reaches thread-private storage. Other AS0
// arguments intentionally remain Generic.
bool argumentIsDefinitelyLocal(
    const llvm::Argument &Arg,
    llvm::SmallPtrSetImpl<const llvm::Value *> &Visiting,
    unsigned Depth) {

    if (Depth > SC_MAX_PROVENANCE_DEPTH)
        return false;

    const llvm::Function *F = Arg.getParent();
    if (F == nullptr || F->isDeclaration())
        return false;

    unsigned ArgNo = Arg.getArgNo();
    bool SawDirectCall = false;

    for (const llvm::User *U : F->users()) {
        auto *CB = llvm::dyn_cast<llvm::CallBase>(U);
        if (CB == nullptr)
            return false;

        const llvm::Value *Called = CB->getCalledOperand();
        if (Called == nullptr || Called->stripPointerCasts() != F)
            return false;

        if (ArgNo >= CB->arg_size())
            return false;

        SawDirectCall = true;

        const llvm::Value *Actual = CB->getArgOperand(ArgNo);
        SCMemorySpace ActualSpace =
            classifyPointerSpaceImpl(Actual, Visiting, Depth + 1);

        if (ActualSpace != SCMemorySpace::Local)
            return false;
    }

    return SawDirectCall;
}

// Resolve a fixed field path through direct callers until every path reaches a
// concrete alloca. This supports pointer values staged in fields of local helper
// objects across multiple direct-call layers.
bool collectConcreteLocalSlots(
    const PointerAccessPath &Input,
    llvm::SmallVectorImpl<PointerAccessPath> &Slots,
    llvm::SmallPtrSetImpl<const llvm::Value *> &Visiting,
    unsigned Depth) {

    if (!Input.Valid || Input.Base == nullptr ||
        Depth > SC_MAX_PROVENANCE_DEPTH) {
        return false;
    }

    const llvm::Value *Base = Input.Base;

    if (llvm::isa<llvm::AllocaInst>(Base)) {
        Slots.push_back(Input);
        return true;
    }

    if (!Visiting.insert(Base).second)
        return false;

    if (auto *Arg = llvm::dyn_cast<llvm::Argument>(Base)) {
        const llvm::Function *F = Arg->getParent();

        if (F == nullptr || F->isDeclaration()) {
            Visiting.erase(Base);
            return false;
        }

        bool SawCall = false;

        for (const llvm::User *U : F->users()) {
            auto *CB = llvm::dyn_cast<llvm::CallBase>(U);

            if (CB == nullptr) {
                Visiting.erase(Base);
                return false;
            }

            const llvm::Value *Called = CB->getCalledOperand();
            if (Called == nullptr || Called->stripPointerCasts() != F) {
                Visiting.erase(Base);
                return false;
            }

            if (Arg->getArgNo() >= CB->arg_size()) {
                Visiting.erase(Base);
                return false;
            }

            SawCall = true;

            const llvm::Value *Actual = CB->getArgOperand(Arg->getArgNo());
            PointerAccessPath ActualPath;

            if (!getPointerAccessPath(Actual, ActualPath)) {
                Visiting.erase(Base);
                return false;
            }

            PointerAccessPath Combined;
            Combined.Base = ActualPath.Base;
            Combined.Valid = true;
            Combined.Indices.append(
                ActualPath.Indices.begin(), ActualPath.Indices.end());
            Combined.Indices.append(
                Input.Indices.begin(), Input.Indices.end());

            if (!collectConcreteLocalSlots(
                    Combined, Slots, Visiting, Depth + 1)) {
                Visiting.erase(Base);
                return false;
            }
        }

        Visiting.erase(Base);
        return SawCall;
    }

    if (auto *PN = llvm::dyn_cast<llvm::PHINode>(Base)) {
        bool SawIncoming = false;

        for (unsigned I = 0; I < PN->getNumIncomingValues(); ++I) {
            PointerAccessPath IncomingPath;

            if (!getPointerAccessPath(PN->getIncomingValue(I), IncomingPath)) {
                Visiting.erase(Base);
                return false;
            }

            IncomingPath.Indices.append(
                Input.Indices.begin(), Input.Indices.end());

            if (!collectConcreteLocalSlots(
                    IncomingPath, Slots, Visiting, Depth + 1)) {
                Visiting.erase(Base);
                return false;
            }

            SawIncoming = true;
        }

        Visiting.erase(Base);
        return SawIncoming;
    }

    if (auto *SI = llvm::dyn_cast<llvm::SelectInst>(Base)) {
        const llvm::Value *Choices[] = {
            SI->getTrueValue(),
            SI->getFalseValue(),
        };

        for (const llvm::Value *Choice : Choices) {
            PointerAccessPath ChoicePath;

            if (!getPointerAccessPath(Choice, ChoicePath)) {
                Visiting.erase(Base);
                return false;
            }

            ChoicePath.Indices.append(
                Input.Indices.begin(), Input.Indices.end());

            if (!collectConcreteLocalSlots(
                    ChoicePath, Slots, Visiting, Depth + 1)) {
                Visiting.erase(Base);
                return false;
            }
        }

        Visiting.erase(Base);
        return true;
    }

    Visiting.erase(Base);
    return false;
}

// Recover the provenance of a pointer value loaded from a provably local field.
// Stores to the exact field define the possible pointee spaces; unrelated fields
// are ignored.
SCMemorySpace classifyLoadedPointerFromLocalSlot(
    const llvm::LoadInst &LI,
    llvm::SmallPtrSetImpl<const llvm::Value *> &Visiting,
    unsigned Depth) {

    if (Depth > SC_MAX_PROVENANCE_DEPTH)
        return SCMemorySpace::Unknown;

    if (!LI.getType()->isPointerTy())
        return SCMemorySpace::Unknown;

    PointerAccessPath LoadedSlot;
    if (!getPointerAccessPath(LI.getPointerOperand(), LoadedSlot))
        return SCMemorySpace::Unknown;

    llvm::SmallVector<PointerAccessPath, 8> ConcreteSlots;
    llvm::SmallPtrSet<const llvm::Value *, 32> SlotVisiting;

    if (!collectConcreteLocalSlots(
            LoadedSlot, ConcreteSlots, SlotVisiting, Depth + 1)) {
        return SCMemorySpace::Unknown;
    }

    SCMemorySpace Result = SCMemorySpace::Unknown;
    bool HaveResult = false;
    bool SawPointerStore = false;

    for (const PointerAccessPath &Slot : ConcreteSlots) {
        auto *AI = llvm::dyn_cast<llvm::AllocaInst>(Slot.Base);
        if (AI == nullptr)
            return SCMemorySpace::Unknown;

        const llvm::Function *F = AI->getFunction();
        if (F == nullptr)
            return SCMemorySpace::Unknown;

        bool SawStoreForThisSlot = false;

        for (const llvm::BasicBlock &BB : *F) {
            for (const llvm::Instruction &I : BB) {
                auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I);
                if (SI == nullptr)
                    continue;

                if (!SI->getValueOperand()->getType()->isPointerTy())
                    continue;

                PointerAccessPath StorePath;
                if (!getPointerAccessPath(SI->getPointerOperand(), StorePath))
                    continue;

                if (StorePath.Base != AI)
                    continue;

                if (!sameIndices(StorePath.Indices, Slot.Indices))
                    continue;

                SawStoreForThisSlot = true;
                SawPointerStore = true;

                SCMemorySpace StoredSpace =
                    classifyPointerSpaceImpl(
                        SI->getValueOperand(), Visiting, Depth + 1);

                if (!HaveResult) {
                    Result = StoredSpace;
                    HaveResult = true;
                } else {
                    Result = mergePointerSpaces(Result, StoredSpace);
                }
            }
        }

        if (!SawStoreForThisSlot)
            return SCMemorySpace::Unknown;
    }

    if (!SawPointerStore || !HaveResult)
        return SCMemorySpace::Unknown;

    return Result;
}

SCMemorySpace classifyPointerSpaceImpl(
    const llvm::Value *Pointer,
    llvm::SmallPtrSetImpl<const llvm::Value *> &Visiting,
    unsigned Depth) {

    if (Pointer == nullptr || Depth > SC_MAX_PROVENANCE_DEPTH)
        return SCMemorySpace::Unknown;

    if (!Pointer->getType()->isPointerTy())
        return SCMemorySpace::Unknown;

    if (!Visiting.insert(Pointer).second) {
        unsigned AddressSpace = getPointerAddressSpace(Pointer);
        return AddressSpace == SC_INVALID_ADDRESS_SPACE
                   ? SCMemorySpace::Unknown
                   : classifyAddressSpace(AddressSpace);
    }

    const llvm::Value *Stripped = stripPointerDerivations(Pointer);

    if (Stripped != Pointer) {
        SCMemorySpace Result =
            classifyPointerSpaceImpl(Stripped, Visiting, Depth + 1);
        Visiting.erase(Pointer);
        return Result;
    }

    if (llvm::isa<llvm::AllocaInst>(Pointer)) {
        Visiting.erase(Pointer);
        return SCMemorySpace::Local;
    }

    if (llvm::isa<llvm::GlobalValue>(Pointer)) {
        unsigned AddressSpace = getPointerAddressSpace(Pointer);
        SCMemorySpace Result = AddressSpace == SC_INVALID_ADDRESS_SPACE
                                   ? SCMemorySpace::Unknown
                                   : classifyAddressSpace(AddressSpace);
        Visiting.erase(Pointer);
        return Result;
    }

    if (auto *Arg = llvm::dyn_cast<llvm::Argument>(Pointer)) {
        unsigned AddressSpace = getPointerAddressSpace(Arg);

        if (AddressSpace != SC_GENERIC_ADDRESS_SPACE &&
            AddressSpace != SC_INVALID_ADDRESS_SPACE) {
            SCMemorySpace Result = classifyAddressSpace(AddressSpace);
            Visiting.erase(Pointer);
            return Result;
        }

        // Do not specialize a generic argument to global/shared based only on
        // current callers. The only interprocedural refinement here is proving
        // that every direct caller passes thread-local storage.
        bool DefinitelyLocal =
            argumentIsDefinitelyLocal(*Arg, Visiting, Depth + 1);
        Visiting.erase(Pointer);

        return DefinitelyLocal ? SCMemorySpace::Local
                               : SCMemorySpace::Generic;
    }

    // Opaque AS0 does not retain the original pointee space when a pointer is
    // staged through a local helper object, so recover it from stores to the
    // exact local field.
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(Pointer)) {
        if (LI->getType()->isPointerTy()) {
            SCMemorySpace LoadedSpace =
                classifyLoadedPointerFromLocalSlot(*LI, Visiting, Depth + 1);

            if (LoadedSpace != SCMemorySpace::Unknown) {
                Visiting.erase(Pointer);
                return LoadedSpace;
            }
        }
    }

    if (auto *PN = llvm::dyn_cast<llvm::PHINode>(Pointer)) {
        SCMemorySpace Result = SCMemorySpace::Unknown;
        bool HaveResult = false;

        for (unsigned I = 0; I < PN->getNumIncomingValues(); ++I) {
            SCMemorySpace Incoming = classifyPointerSpaceImpl(
                PN->getIncomingValue(I), Visiting, Depth + 1);

            if (!HaveResult) {
                Result = Incoming;
                HaveResult = true;
            } else {
                Result = mergePointerSpaces(Result, Incoming);
            }
        }

        Visiting.erase(Pointer);
        return HaveResult ? Result : SCMemorySpace::Unknown;
    }

    if (auto *SI = llvm::dyn_cast<llvm::SelectInst>(Pointer)) {
        SCMemorySpace TrueSpace = classifyPointerSpaceImpl(
            SI->getTrueValue(), Visiting, Depth + 1);
        SCMemorySpace FalseSpace = classifyPointerSpaceImpl(
            SI->getFalseValue(), Visiting, Depth + 1);
        SCMemorySpace Result = mergePointerSpaces(TrueSpace, FalseSpace);
        Visiting.erase(Pointer);
        return Result;
    }

    unsigned AddressSpace = getPointerAddressSpace(Pointer);
    Visiting.erase(Pointer);

    if (AddressSpace == SC_INVALID_ADDRESS_SPACE)
        return SCMemorySpace::Unknown;

    return classifyAddressSpace(AddressSpace);
}

} // namespace

SCMemorySpace classifyPointerSpace(const llvm::Value *Pointer) {
    llvm::SmallPtrSet<const llvm::Value *, 32> Visiting;
    return classifyPointerSpaceImpl(Pointer, Visiting, 0);
}

} // namespace supercollider
