#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace supercollider {

// Compile-time configuration for one SuperCollider pass invocation.
//
// ReadDelayNs bounds the nanosleep request inserted after weak loads before
// their strong duplicate reads. WriteDelayNs bounds the corresponding request
// inserted after weak stores. EnableIntraWarpLostUpdate controls the independent
// pre-store address-match detector. EnableAsyncCopy controls classic 4/8/16-byte
// cp.async instrumentation, while EnableBulkAsyncCopy controls the paper-supported
// cp.async.bulk shared->global detector. EnableBlockShuffle controls the affine
// blockIdx.x permutation, and BlockShuffleSeed salts the per-grid mapping. The
// pass accepts these values through its textual pipeline parameters so target
// build systems can select the policy without rebuilding the plugin.
struct SCConfig {
    std::uint32_t ReadDelayNs = 1;
    std::uint32_t WriteDelayNs = 1;
    bool EnableIntraWarpLostUpdate = false;
    bool EnableAsyncCopy = false;
    bool EnableBulkAsyncCopy = false;
    bool EnableBlockShuffle = false;
    std::uint64_t BlockShuffleSeed = 0x9e3779b97f4a7c15ULL;
};

// Parse the parameter payload from:
//
//   supercollider<rdelay=N;wdelay=M;ilu=0|1;async=0|1;bulk=0|1;shuffle=0|1;shuffle_seed=S>
//
// An empty payload keeps the default configuration. Unknown, duplicate, or
// malformed parameters are rejected rather than silently ignored.
llvm::Expected<SCConfig> parseSuperColliderConfig(llvm::StringRef Parameters);

} // namespace supercollider
