#include "config.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <string>

namespace supercollider {

namespace {

llvm::Error makeConfigError(const std::string &Message) {
    return llvm::make_error<llvm::StringError>(
        Message,
        llvm::inconvertibleErrorCode());
}

llvm::Expected<std::uint32_t> parseDelayValue(
    llvm::StringRef Key,
    llvm::StringRef ValueText) {

    if (ValueText.empty())
        return makeConfigError(
            "missing value for SuperCollider parameter '" + Key.str() + "'");

    std::uint64_t Parsed = 0;
    if (ValueText.getAsInteger(10, Parsed))
        return makeConfigError(
            "invalid unsigned integer for SuperCollider parameter '" +
            Key.str() + "': '" + ValueText.str() + "'");

    if (Parsed > std::numeric_limits<std::uint32_t>::max())
        return makeConfigError(
            "SuperCollider parameter '" + Key.str() +
            "' exceeds the 32-bit nanosleep argument range");

    return static_cast<std::uint32_t>(Parsed);
}

llvm::Expected<std::uint64_t> parseUnsigned64Value(
    llvm::StringRef Key,
    llvm::StringRef ValueText) {

    if (ValueText.empty())
        return makeConfigError(
            "missing value for SuperCollider parameter '" + Key.str() + "'");

    std::uint64_t Parsed = 0;
    if (ValueText.getAsInteger(10, Parsed))
        return makeConfigError(
            "invalid unsigned integer for SuperCollider parameter '" +
            Key.str() + "': '" + ValueText.str() + "'");

    return Parsed;
}

llvm::Expected<bool> parseBooleanSwitch(
    llvm::StringRef Key,
    llvm::StringRef ValueText) {

    if (ValueText == "0")
        return false;
    if (ValueText == "1")
        return true;

    return makeConfigError(
        "SuperCollider parameter '" + Key.str() +
        "' expects 0 or 1, got '" + ValueText.str() + "'");
}

} // namespace

llvm::Expected<SCConfig> parseSuperColliderConfig(llvm::StringRef Parameters) {
    SCConfig Config;

    Parameters = Parameters.trim();
    if (Parameters.empty())
        return Config;

    llvm::SmallVector<llvm::StringRef, 8> Fields;
    llvm::SplitString(Parameters, Fields, ";,");

    bool SawReadDelay = false;
    bool SawWriteDelay = false;
    bool SawIntraWarpLostUpdate = false;
    bool SawAsyncCopy = false;
    bool SawBulkAsyncCopy = false;
    bool SawBlockShuffle = false;
    bool SawBlockShuffleSeed = false;

    for (llvm::StringRef Field : Fields) {
        Field = Field.trim();
        if (Field.empty())
            continue;

        auto [Key, ValueText] = Field.split('=');
        Key = Key.trim();
        ValueText = ValueText.trim();

        if (Key == "rdelay") {
            if (SawReadDelay)
                return makeConfigError(
                    "duplicate SuperCollider parameter 'rdelay'");

            llvm::Expected<std::uint32_t> Value =
                parseDelayValue(Key, ValueText);
            if (!Value)
                return Value.takeError();

            Config.ReadDelayNs = *Value;
            SawReadDelay = true;
            continue;
        }

        if (Key == "wdelay") {
            if (SawWriteDelay)
                return makeConfigError(
                    "duplicate SuperCollider parameter 'wdelay'");

            llvm::Expected<std::uint32_t> Value =
                parseDelayValue(Key, ValueText);
            if (!Value)
                return Value.takeError();

            Config.WriteDelayNs = *Value;
            SawWriteDelay = true;
            continue;
        }

        if (Key == "ilu") {
            if (SawIntraWarpLostUpdate)
                return makeConfigError(
                    "duplicate SuperCollider parameter 'ilu'");

            llvm::Expected<bool> Value =
                parseBooleanSwitch(Key, ValueText);
            if (!Value)
                return Value.takeError();

            Config.EnableIntraWarpLostUpdate = *Value;
            SawIntraWarpLostUpdate = true;
            continue;
        }

        if (Key == "async") {
            if (SawAsyncCopy)
                return makeConfigError(
                    "duplicate SuperCollider parameter 'async'");

            llvm::Expected<bool> Value =
                parseBooleanSwitch(Key, ValueText);
            if (!Value)
                return Value.takeError();

            Config.EnableAsyncCopy = *Value;
            SawAsyncCopy = true;
            continue;
        }

        if (Key == "bulk") {
            if (SawBulkAsyncCopy)
                return makeConfigError(
                    "duplicate SuperCollider parameter 'bulk'");

            llvm::Expected<bool> Value =
                parseBooleanSwitch(Key, ValueText);
            if (!Value)
                return Value.takeError();

            Config.EnableBulkAsyncCopy = *Value;
            SawBulkAsyncCopy = true;
            continue;
        }

        if (Key == "shuffle") {
            if (SawBlockShuffle)
                return makeConfigError(
                    "duplicate SuperCollider parameter 'shuffle'");

            llvm::Expected<bool> Value =
                parseBooleanSwitch(Key, ValueText);
            if (!Value)
                return Value.takeError();

            Config.EnableBlockShuffle = *Value;
            SawBlockShuffle = true;
            continue;
        }

        if (Key == "shuffle_seed") {
            if (SawBlockShuffleSeed)
                return makeConfigError(
                    "duplicate SuperCollider parameter 'shuffle_seed'");

            llvm::Expected<std::uint64_t> Value =
                parseUnsigned64Value(Key, ValueText);
            if (!Value)
                return Value.takeError();

            Config.BlockShuffleSeed = *Value;
            SawBlockShuffleSeed = true;
            continue;
        }

        return makeConfigError(
            "unknown SuperCollider parameter '" + Key.str() +
            "'; expected rdelay, wdelay, ilu, async, bulk, shuffle, or shuffle_seed");
    }

    return Config;
}

} // namespace supercollider
