#pragma once

#include <cstddef>
#include <cstdint>

namespace supercollider {

constexpr std::uint32_t SC_RUNTIME_MAGIC = 0x53435254u;
constexpr std::uint32_t SC_RUNTIME_VERSION = 1u;
constexpr std::size_t SC_MAX_ERROR_SITES = 256;

struct alignas(8) SCRuntimeErrorRecord {
    std::uint64_t SiteID = 0;
    std::uint64_t Occurrences = 0;
    std::uint64_t FirstAddress = 0;

    std::uint32_t RaceType = 0;

    std::uint32_t ThreadX = 0;
    std::uint32_t ThreadY = 0;
    std::uint32_t ThreadZ = 0;

    std::uint32_t BlockX = 0;
    std::uint32_t BlockY = 0;
    std::uint32_t BlockZ = 0;

    // For ILU records, the matching warp-lane mask from match.any.sync.
    // Zero for ordinary CR/LU records.
    std::uint32_t LaneMask = 0;
};

struct alignas(8) SCRuntimeState {
    std::uint32_t Magic = 0;
    std::uint32_t Version = 0;
    std::uint32_t Overflow = 0;
    std::uint32_t Reserved = 0;
    std::uint64_t DroppedOccurrences = 0;
    SCRuntimeErrorRecord Errors[SC_MAX_ERROR_SITES] = {};
};

constexpr std::size_t SC_RUNTIME_STATE_BYTES = sizeof(SCRuntimeState);

static_assert(sizeof(SCRuntimeErrorRecord) == 56,
              "unexpected SuperCollider error-record layout");
static_assert(SC_RUNTIME_STATE_BYTES == 14360,
              "unexpected SuperCollider runtime-state layout");

} // namespace supercollider
