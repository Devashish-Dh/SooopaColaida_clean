#pragma once

#include "llvm/Support/Alignment.h"

namespace llvm {
class DebugLoc;
class Instruction;
class LoadInst;
class Type;
class Value;
}

namespace supercollider {

// Return true for scalar value types currently represented by a single strong
// duplicate read in the NVPTX backend: 8/16/32/64-bit integers, float, and
// double.
bool isSupportedStrongLoadType(const llvm::Type *Type);

// NVPTX cannot lower an atomic duplicate load when the access alignment is
// weaker than the scalar width. Return false for those under-aligned accesses
// so callers can skip ordinary CR/LU instrumentation instead of producing IR
// that later fails in llc.
bool isSupportedStrongLoadAlignment(
    const llvm::Type *Type,
    llvm::Align Alignment);

// Insert a strong, system-scoped load immediately before InsertBefore.
//
// The load uses LLVM Monotonic ordering with SyncScope::System. For NVPTX this
// requests a strong system-scoped relaxed read, emitted as the appropriate
// ld.relaxed.sys form for generic, global, or shared addressing.
//
// Pointer and ValueType describe the original weak access. Alignment is kept
// identical to the original operation, and DebugLocation is propagated so the
// inserted instruction remains attributable to the source access.
llvm::LoadInst *createStrongSystemLoad(
    llvm::Instruction &InsertBefore,
    llvm::Type *ValueType,
    llvm::Value *Pointer,
    llvm::Align Alignment,
    const llvm::DebugLoc &DebugLocation);

} // namespace supercollider
