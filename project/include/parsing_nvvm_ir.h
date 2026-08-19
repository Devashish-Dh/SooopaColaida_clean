#pragma once

#include "internal_defns.h"

namespace llvm {
class Value;
}

namespace supercollider {

// Map an LLVM/NVPTX address-space id to SuperCollider's memory-space model.
SCMemorySpace classifyAddressSpace(unsigned AddressSpace);

// Return the address space encoded in a pointer type. Non-pointer values return
// SC_INVALID_ADDRESS_SPACE.
unsigned getPointerAddressSpace(const llvm::Value *V);

// Strip pointer-preserving GEP, bitcast, and addrspacecast chains to expose the
// underlying pointer-producing value.
const llvm::Value *stripPointerDerivations(const llvm::Value *V);

// Determine the effective NVPTX memory space reached by Pointer. The analysis
// follows pointer derivations, selected interprocedural argument flows, and
// pointer values staged through exact fields of local helper objects.
//
// Generic is retained when the concrete runtime state space cannot be proven.
// Local is returned only when the provenance walk can prove thread-private
// storage.
SCMemorySpace classifyPointerSpace(const llvm::Value *Pointer);

} // namespace supercollider
