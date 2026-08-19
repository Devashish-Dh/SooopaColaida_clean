#pragma once

#include "llvm/IR/DebugLoc.h"

namespace llvm {
class CallInst;
class Instruction;
class Value;
}

namespace supercollider {

// Insert a call to an internal GPU helper that computes a 64-bit FNV-1a hash
// over ByteCount bytes beginning at GlobalBase. GlobalBase must be a global
// address-space pointer. The helper performs system-scoped strong byte loads so
// the hash observes the completed bulk-copy destination rather than stale cache
// contents.
llvm::CallInst *createFNV1aGlobalHash(
    llvm::Instruction &InsertBefore,
    llvm::Value *GlobalBase,
    llvm::Value *ByteCount,
    const llvm::DebugLoc &DebugLocation);

} // namespace supercollider
