#pragma once

#include "llvm/IR/DebugLoc.h"

namespace llvm {
class BasicBlock;
class BranchInst;
class CallInst;
class Instruction;
class Type;
class Value;
}

namespace supercollider {

// IR objects created for one race assertion. The failure block is initially
// terminated by trap+unreachable as a valid fallback. sc_error.cpp replaces
// that fallback with sc_err()+continue once reporting insertion succeeds.
struct RaceAssertResult {
    llvm::Value *Mismatch = nullptr;
    llvm::BranchInst *Branch = nullptr;
    llvm::CallInst *TrapCall = nullptr;
    llvm::BasicBlock *FailureBlock = nullptr;
    llvm::BasicBlock *ContinuationBlock = nullptr;
};

// Return true for scalar types that can be compared directly without changing
// their representation. Integer values use icmp ne. float/double values use
// ordered fcmp ne so an unordered NaN comparison does not create a race report
// by itself.
bool isSupportedRaceComparisonType(const llvm::Type *Type);

// Insert a race assertion immediately before ContinueAt.
//
// ExpectedValue is the value produced by the original weak load, or the value
// written by the original weak store. ObservedValue is the delayed strong load.
// A mismatch branches to a dedicated failure block containing a temporary
// llvm.trap fallback; sc_error.cpp replaces it with the reporting callback.
RaceAssertResult insertRaceAssert(
    llvm::Instruction &ContinueAt,
    llvm::Value *ExpectedValue,
    llvm::Value *ObservedValue,
    const llvm::DebugLoc &DebugLocation);

// Insert a race assertion from an already-computed i1 mismatch predicate.
// This is used by detectors that combine several comparisons into one report
// condition, such as classic cp.async and bulk FNV verification.
RaceAssertResult insertRaceAssertFromMismatch(
    llvm::Instruction &ContinueAt,
    llvm::Value *Mismatch,
    const llvm::DebugLoc &DebugLocation);

} // namespace supercollider
