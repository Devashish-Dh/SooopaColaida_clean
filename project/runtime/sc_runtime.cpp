#include "sc_runtime.h"
#include "sc_runtime_abi.h"
#include "internal_defns.h"

#include <cuda.h>
#include <cupti.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr const char *SC_RUNTIME_STORAGE_SYMBOL =
    "__sc_runtime_state_storage";
constexpr const char *SC_SITE_METADATA_SYMBOL =
    "__sc_site_metadata";

struct TrackedModule {
    CUcontext Context = nullptr;
    CUmodule Module = nullptr;
    CUlibrary Library = nullptr;
    std::uint64_t Sequence = 0;
};

struct SiteMetadata {
    std::uint64_t SiteID = 0;
    std::uint32_t RaceType = 0;
    unsigned Line = 0;
    unsigned Column = 0;
    std::string Function;
    std::string File;
};

struct HostRaceRecord {
    std::uint64_t ModuleSequence = 0;
    supercollider::SCRuntimeErrorRecord DeviceRecord;
    SiteMetadata Metadata;
};

struct RuntimeStateIdentity {
    CUcontext Context = nullptr;
    CUdeviceptr Address = 0;

    bool operator==(const RuntimeStateIdentity &Other) const {
        return Context == Other.Context && Address == Other.Address;
    }
};

struct RuntimeStateIdentityHash {
    std::size_t operator()(const RuntimeStateIdentity &Identity) const {
        std::size_t ContextHash = std::hash<std::uintptr_t>{}(
            reinterpret_cast<std::uintptr_t>(Identity.Context));
        std::size_t AddressHash = std::hash<unsigned long long>{}(
            static_cast<unsigned long long>(Identity.Address));
        return ContextHash ^ (AddressHash + 0x9e3779b97f4a7c15ULL +
                              (ContextHash << 6) + (ContextHash >> 2));
    }
};

std::mutex ModulesMutex;
std::vector<TrackedModule> Modules;
std::unordered_set<std::uintptr_t> SeenModules;
std::atomic<std::uint64_t> NextModuleSequence{1};
std::atomic<bool> ExitHandlerRegistered{false};
std::atomic<bool> Reporting{false};
CUpti_SubscriberHandle Subscriber = nullptr;

std::uint64_t metadataSiteID(std::uint64_t ReportSiteID) {
    return ReportSiteID & ~supercollider::SC_INTRA_WARP_SITE_TAG;
}

const char *raceTypeDescription(std::uint32_t RaceType) {
    switch (static_cast<supercollider::SCRaceType>(RaceType)) {
    case supercollider::SCRaceType::ClobberedRead:
        return "This thread READ from an address that another thread clobbered";
    case supercollider::SCRaceType::LostUpdate:
        return "This thread WROTE a value that another thread clobbered";
    case supercollider::SCRaceType::IntraWarpLostUpdate:
        return "This warp WROTE more than one value to the same address";
    case supercollider::SCRaceType::AsyncCopy:
        return "An asynchronous copy observed conflicting source or destination data";
    default:
        return "Unknown SuperCollider race type";
    }
}

void printLaneMask(std::uint32_t LaneMask) {
    std::fprintf(stderr, "Lanes in warp: {");

    for (unsigned Lane = 0; Lane < 32; ++Lane) {
        if ((LaneMask & (std::uint32_t{1} << Lane)) != 0)
            std::fprintf(stderr, " l%u", Lane);
    }

    std::fprintf(stderr, " }\n");
}

std::string demangle(const std::string &Name) {
    int Status = 0;
    char *Result = abi::__cxa_demangle(Name.c_str(), nullptr, nullptr, &Status);
    if (Status != 0 || Result == nullptr)
        return Name;

    std::string Demangled(Result);
    std::free(Result);
    return Demangled;
}

std::vector<std::string> splitTabs(const std::string &Line) {
    std::vector<std::string> Fields;
    std::size_t Start = 0;

    while (true) {
        std::size_t End = Line.find('\t', Start);
        if (End == std::string::npos) {
            Fields.emplace_back(Line.substr(Start));
            break;
        }

        Fields.emplace_back(Line.substr(Start, End - Start));
        Start = End + 1;
    }

    return Fields;
}

std::unordered_map<std::uint64_t, SiteMetadata> parseSiteMetadata(
    const std::string &Blob) {

    std::unordered_map<std::uint64_t, SiteMetadata> Result;
    std::istringstream Input(Blob);
    std::string Line;

    while (std::getline(Input, Line)) {
        if (Line.empty())
            continue;

        std::vector<std::string> Fields = splitTabs(Line);
        if (Fields.size() != 6)
            continue;

        try {
            SiteMetadata Metadata;
            Metadata.SiteID = std::stoull(Fields[0]);
            Metadata.RaceType = static_cast<std::uint32_t>(
                std::stoul(Fields[1]));
            Metadata.Line = static_cast<unsigned>(std::stoul(Fields[2]));
            Metadata.Column = static_cast<unsigned>(std::stoul(Fields[3]));
            Metadata.Function = Fields[4];
            Metadata.File = Fields[5];
            Result.emplace(Metadata.SiteID, std::move(Metadata));
        } catch (...) {
            continue;
        }
    }

    return Result;
}

bool copyModuleGlobal(
    CUmodule Module,
    const char *Name,
    void *Destination,
    std::size_t DestinationBytes,
    std::size_t *ActualBytes = nullptr) {

    CUdeviceptr DeviceAddress = 0;
    std::size_t Bytes = 0;

    CUresult Result = cuModuleGetGlobal(
        &DeviceAddress,
        &Bytes,
        Module,
        Name);
    if (Result != CUDA_SUCCESS)
        return false;

    if (ActualBytes != nullptr)
        *ActualBytes = Bytes;

    if (Destination == nullptr)
        return true;

    if (Bytes > DestinationBytes)
        return false;

    return cuMemcpyDtoH(Destination, DeviceAddress, Bytes) == CUDA_SUCCESS;
}

bool getModuleGlobalInfo(
    CUmodule Module,
    const char *Name,
    CUdeviceptr &DeviceAddress,
    std::size_t &Bytes) {

    return cuModuleGetGlobal(
               &DeviceAddress,
               &Bytes,
               Module,
               Name) == CUDA_SUCCESS;
}

bool copyLibraryGlobal(
    CUlibrary Library,
    const char *Name,
    void *Destination,
    std::size_t DestinationBytes,
    std::size_t *ActualBytes = nullptr) {

    CUdeviceptr DeviceAddress = 0;
    std::size_t Bytes = 0;

    CUresult Result = cuLibraryGetGlobal(
        &DeviceAddress,
        &Bytes,
        Library,
        Name);
    if (Result != CUDA_SUCCESS)
        return false;

    if (ActualBytes != nullptr)
        *ActualBytes = Bytes;

    if (Destination == nullptr)
        return true;

    if (Bytes > DestinationBytes)
        return false;

    return cuMemcpyDtoH(Destination, DeviceAddress, Bytes) == CUDA_SUCCESS;
}

bool getLibraryGlobalInfo(
    CUlibrary Library,
    const char *Name,
    CUdeviceptr &DeviceAddress,
    std::size_t &Bytes) {

    return cuLibraryGetGlobal(
               &DeviceAddress,
               &Bytes,
               Library,
               Name) == CUDA_SUCCESS;
}

bool getRuntimeStateIdentity(
    const TrackedModule &Tracked,
    RuntimeStateIdentity &Identity,
    std::size_t &Bytes) {

    CUdeviceptr DeviceAddress = 0;
    bool Found = false;

    if (Tracked.Module != nullptr) {
        Found = getModuleGlobalInfo(
            Tracked.Module,
            SC_RUNTIME_STORAGE_SYMBOL,
            DeviceAddress,
            Bytes);
    } else if (Tracked.Library != nullptr) {
        Found = getLibraryGlobalInfo(
            Tracked.Library,
            SC_RUNTIME_STORAGE_SYMBOL,
            DeviceAddress,
            Bytes);
    }

    if (!Found)
        return false;

    Identity.Context = Tracked.Context;
    Identity.Address = DeviceAddress;
    return true;
}

bool readSiteMetadata(
    const TrackedModule &Tracked,
    std::unordered_map<std::uint64_t, SiteMetadata> &Metadata) {

    std::size_t Bytes = 0;
    if (Tracked.Module != nullptr) {
        if (!copyModuleGlobal(
                Tracked.Module,
                SC_SITE_METADATA_SYMBOL,
                nullptr,
                0,
                &Bytes) ||
            Bytes == 0) {
            return false;
        }
    } else if (Tracked.Library != nullptr) {
        if (!copyLibraryGlobal(
                Tracked.Library,
                SC_SITE_METADATA_SYMBOL,
                nullptr,
                0,
                &Bytes) ||
            Bytes == 0) {
            return false;
        }
    } else {
        return false;
    }

    std::string Blob(Bytes, '\0');
    if (Tracked.Module != nullptr) {
        if (!copyModuleGlobal(
                Tracked.Module,
                SC_SITE_METADATA_SYMBOL,
                Blob.data(),
                Blob.size())) {
            return false;
        }
    } else if (!copyLibraryGlobal(
                   Tracked.Library,
                   SC_SITE_METADATA_SYMBOL,
                   Blob.data(),
                   Blob.size())) {
        return false;
    }

    Metadata = parseSiteMetadata(Blob);
    return true;
}

bool readRuntimeState(
    const TrackedModule &Tracked,
    supercollider::SCRuntimeState &State) {

    std::size_t Bytes = 0;
    if (Tracked.Module != nullptr) {
        if (!copyModuleGlobal(
                Tracked.Module,
                SC_RUNTIME_STORAGE_SYMBOL,
                nullptr,
                0,
                &Bytes)) {
            return false;
        }
    } else if (Tracked.Library != nullptr) {
        if (!copyLibraryGlobal(
                Tracked.Library,
                SC_RUNTIME_STORAGE_SYMBOL,
                nullptr,
                0,
                &Bytes)) {
            return false;
        }
    } else {
        return false;
    }

    if (Bytes != sizeof(State)) {
        std::fprintf(
            stderr,
            "[SuperCollider] ignoring runtime state with unexpected size "
            "%zu (expected %zu)\n",
            Bytes,
            sizeof(State));
        return false;
    }

    std::memset(&State, 0, sizeof(State));
    if (Tracked.Module != nullptr) {
        return copyModuleGlobal(
            Tracked.Module,
            SC_RUNTIME_STORAGE_SYMBOL,
            &State,
            sizeof(State));
    }

    return copyLibraryGlobal(
        Tracked.Library,
        SC_RUNTIME_STORAGE_SYMBOL,
        &State,
        sizeof(State));
}

void collectFromModule(
    const TrackedModule &Tracked,
    std::unordered_set<RuntimeStateIdentity, RuntimeStateIdentityHash>
        &SeenRuntimeStates,
    std::vector<HostRaceRecord> &Races,
    std::uint64_t &DroppedOccurrences) {

    if (Tracked.Context == nullptr ||
        (Tracked.Module == nullptr && Tracked.Library == nullptr))
        return;

    CUresult PushResult = cuCtxPushCurrent(Tracked.Context);
    if (PushResult != CUDA_SUCCESS)
        return;

    cuCtxSynchronize();

    RuntimeStateIdentity Identity;
    std::size_t RuntimeStateBytes = 0;
    if (!getRuntimeStateIdentity(Tracked, Identity, RuntimeStateBytes) ||
        !SeenRuntimeStates.insert(Identity).second) {
        CUcontext Popped = nullptr;
        cuCtxPopCurrent(&Popped);
        return;
    }

    if (RuntimeStateBytes != sizeof(supercollider::SCRuntimeState)) {
        std::fprintf(
            stderr,
            "[SuperCollider] ignoring runtime state with unexpected size "
            "%zu (expected %zu)\n",
            RuntimeStateBytes,
            sizeof(supercollider::SCRuntimeState));
        CUcontext Popped = nullptr;
        cuCtxPopCurrent(&Popped);
        return;
    }

    supercollider::SCRuntimeState State{};
    if (!readRuntimeState(Tracked, State)) {
        CUcontext Popped = nullptr;
        cuCtxPopCurrent(&Popped);
        return;
    }

    std::unordered_map<std::uint64_t, SiteMetadata> Metadata;
    readSiteMetadata(Tracked, Metadata);

    if (State.Magic != 0 && State.Magic != supercollider::SC_RUNTIME_MAGIC) {
        std::fprintf(
            stderr,
            "[SuperCollider] module %llu has an incompatible runtime magic\n",
            static_cast<unsigned long long>(Tracked.Sequence));
    } else if (State.Version != 0 &&
               State.Version != supercollider::SC_RUNTIME_VERSION) {
        std::fprintf(
            stderr,
            "[SuperCollider] module %llu has runtime ABI version %u; expected %u\n",
            static_cast<unsigned long long>(Tracked.Sequence),
            State.Version,
            supercollider::SC_RUNTIME_VERSION);
    }

    DroppedOccurrences += State.DroppedOccurrences;

    for (const supercollider::SCRuntimeErrorRecord &Record : State.Errors) {
        if (Record.SiteID == 0 || Record.Occurrences == 0)
            continue;

        HostRaceRecord HostRecord;
        HostRecord.ModuleSequence = Tracked.Sequence;
        HostRecord.DeviceRecord = Record;

        std::uint64_t SourceSiteID = metadataSiteID(Record.SiteID);
        auto It = Metadata.find(SourceSiteID);
        if (It != Metadata.end())
            HostRecord.Metadata = It->second;
        else
            HostRecord.Metadata.SiteID = SourceSiteID;

        Races.push_back(std::move(HostRecord));
    }

    CUcontext Popped = nullptr;
    cuCtxPopCurrent(&Popped);
}

void printReport(const std::vector<HostRaceRecord> &Races,
                 std::uint64_t DroppedOccurrences) {
    if (Races.empty()) {
        std::fprintf(
            stderr,
            "+======================== SUPERCOLLIDER ========================+\n"
            "0 RACE CONDITIONS DETECTED\n"
            "+===============================================================+\n");
        return;
    }

    std::fprintf(
        stderr,
        "+======================== SUPERCOLLIDER ========================+\n"
        "+=========================== WARNING ===========================+\n"
        "%zu RACE CONDITIONS DETECTED!\n"
        "+===============================================================+\n",
        Races.size());

    for (const HostRaceRecord &Race : Races) {
        const auto &Record = Race.DeviceRecord;
        const auto &Metadata = Race.Metadata;

        std::string Function = Metadata.Function.empty()
                                   ? "<unknown>"
                                   : demangle(Metadata.Function);
        const char *File = Metadata.File.empty()
                               ? "<unknown>"
                               : Metadata.File.c_str();

        if (Metadata.Line != 0) {
            std::fprintf(
                stderr,
                "! In function %s at %s:%u\n",
                Function.c_str(),
                File,
                Metadata.Line);
        } else {
            std::fprintf(
                stderr,
                "! In function %s\n",
                Function.c_str());
        }

        std::fprintf(
            stderr,
            "Type : %s\n",
            raceTypeDescription(Record.RaceType));

        if (static_cast<supercollider::SCRaceType>(Record.RaceType) ==
            supercollider::SCRaceType::IntraWarpLostUpdate) {
            printLaneMask(Record.LaneMask);
        }

        std::fprintf(
            stderr,
            "Site : module %llu, site %llu\n"
            "First noticed: Thread (%u,%u,%u), Block (%u,%u,%u), "
            "Address 0x%llx\n"
            "Occurrences : %llu times (possibly by other threads at other addresses)\n",
            static_cast<unsigned long long>(Race.ModuleSequence),
            static_cast<unsigned long long>(metadataSiteID(Record.SiteID)),
            Record.ThreadX,
            Record.ThreadY,
            Record.ThreadZ,
            Record.BlockX,
            Record.BlockY,
            Record.BlockZ,
            static_cast<unsigned long long>(Record.FirstAddress),
            static_cast<unsigned long long>(Record.Occurrences));
    }

    if (DroppedOccurrences != 0) {
        std::fprintf(
            stderr,
            "[SuperCollider] report buffer overflow: %llu race occurrences "
            "could not be assigned to a distinct site slot\n",
            static_cast<unsigned long long>(DroppedOccurrences));
    }
}

void reportAtExit() {
    sc_report_now();
}

void registerExitHandlerOnce() {
    bool Expected = false;
    if (ExitHandlerRegistered.compare_exchange_strong(Expected, true))
        std::atexit(reportAtExit);
}

void trackLoadedImage(CUcontext Context, CUmodule Module, CUlibrary Library) {
    if (Context == nullptr || (Module == nullptr && Library == nullptr))
        return;

    std::uintptr_t Key =
        (reinterpret_cast<std::uintptr_t>(
             Module != nullptr ? static_cast<void *>(Module)
                               : static_cast<void *>(Library))
         << 1) |
        (Module != nullptr ? 0u : 1u);

    {
        std::lock_guard<std::mutex> Lock(ModulesMutex);
        if (!SeenModules.insert(Key).second)
            return;

        TrackedModule Entry;
        Entry.Context = Context;
        Entry.Module = Module;
        Entry.Library = Library;
        Entry.Sequence = NextModuleSequence.fetch_add(1);
        Modules.push_back(Entry);
    }

    // Register after CUDA has loaded a module. In normal CUDA-runtime use this
    // places our atexit callback after runtime initialization, allowing the
    // report to run before the CUDA runtime tears down its contexts.
    registerExitHandlerOnce();
}

void trackModuleFromCallback(const CUpti_CallbackData *Data) {
    if (Data == nullptr || Data->functionParams == nullptr ||
        Data->functionReturnValue == nullptr) {
        return;
    }

    CUresult Result = *static_cast<const CUresult *>(Data->functionReturnValue);
    if (Result != CUDA_SUCCESS)
        return;

    // Every cuModuleLoad* Driver API entry has CUmodule* as its first argument.
    // CUPTI's generated parameter structure stores arguments in declaration
    // order, so reading the first field covers cuModuleLoad, cuModuleLoadData,
    // cuModuleLoadDataEx, and cuModuleLoadFatBinary.
    CUmodule *ModuleOutput =
        *reinterpret_cast<CUmodule *const *>(Data->functionParams);
    if (ModuleOutput == nullptr || *ModuleOutput == nullptr)
        return;

    trackLoadedImage(Data->context, *ModuleOutput, nullptr);
}

void trackLibraryLoadFromCallback(const CUpti_CallbackData *Data) {
    if (Data == nullptr || Data->functionParams == nullptr ||
        Data->functionReturnValue == nullptr) {
        return;
    }

    CUresult Result = *static_cast<const CUresult *>(Data->functionReturnValue);
    if (Result != CUDA_SUCCESS)
        return;

    const auto *Params =
        static_cast<const cuLibraryLoadData_params *>(Data->functionParams);
    if (Params == nullptr || Params->library == nullptr ||
        *Params->library == nullptr) {
        return;
    }

    trackLoadedImage(Data->context, nullptr, *Params->library);
}

void trackLibraryModuleFromCallback(const CUpti_CallbackData *Data) {
    if (Data == nullptr || Data->functionParams == nullptr ||
        Data->functionReturnValue == nullptr) {
        return;
    }

    CUresult Result = *static_cast<const CUresult *>(Data->functionReturnValue);
    if (Result != CUDA_SUCCESS)
        return;

    const auto *Params =
        static_cast<const cuLibraryGetModule_params *>(Data->functionParams);
    if (Params == nullptr || Params->pMod == nullptr ||
        *Params->pMod == nullptr) {
        return;
    }

    trackLoadedImage(Data->context, *Params->pMod, nullptr);
}

void CUPTIAPI callback(
    void *,
    CUpti_CallbackDomain Domain,
    CUpti_CallbackId CallbackID,
    const void *CallbackData) {

    if (Domain != CUPTI_CB_DOMAIN_DRIVER_API)
        return;

    const auto *Data = static_cast<const CUpti_CallbackData *>(CallbackData);
    if (Data == nullptr || Data->callbackSite != CUPTI_API_EXIT)
        return;

    switch (CallbackID) {
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoad:
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData:
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx:
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadFatBinary:
        trackModuleFromCallback(Data);
        break;
    case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadData:
    case CUPTI_DRIVER_TRACE_CBID_cuLibraryLoadFromFile:
        trackLibraryLoadFromCallback(Data);
        break;
    case CUPTI_DRIVER_TRACE_CBID_cuLibraryGetModule:
        trackLibraryModuleFromCallback(Data);
        break;
    default:
        break;
    }
}

void initializeRuntime() {
    CUresult InitResult = cuInit(0);
    if (InitResult != CUDA_SUCCESS && InitResult != CUDA_ERROR_NO_DEVICE) {
        const char *Message = nullptr;
        cuGetErrorString(InitResult, &Message);
        std::fprintf(
            stderr,
            "[SuperCollider] CUDA driver initialization failed: %s\n",
            Message != nullptr ? Message : "unknown error");
    }

    CUptiResult Result = cuptiSubscribe(&Subscriber, callback, nullptr);
    if (Result != CUPTI_SUCCESS) {
        const char *Message = nullptr;
        cuptiGetResultString(Result, &Message);
        std::fprintf(
            stderr,
            "[SuperCollider] CUPTI subscription failed: %s\n",
            Message != nullptr ? Message : "unknown error");
        return;
    }

    Result = cuptiEnableDomain(
        1,
        Subscriber,
        CUPTI_CB_DOMAIN_DRIVER_API);
    if (Result != CUPTI_SUCCESS) {
        const char *Message = nullptr;
        cuptiGetResultString(Result, &Message);
        std::fprintf(
            stderr,
            "[SuperCollider] failed to enable CUPTI driver callbacks: %s\n",
            Message != nullptr ? Message : "unknown error");
        return;
    }

    std::fprintf(
        stderr,
        "+======================== SUPERCOLLIDER ENGAGED! =======================+\n"
        "MAX_ERRORS : %zu\n"
        "+=======================================================================+\n",
        supercollider::SC_MAX_ERROR_SITES);
}

struct RuntimeInitializer {
    RuntimeInitializer() {
        initializeRuntime();
    }
};

RuntimeInitializer Initializer;

} // namespace

extern "C" void sc_report_now(void) {
    bool Expected = false;
    if (!Reporting.compare_exchange_strong(Expected, true))
        return;

    std::vector<TrackedModule> Snapshot;
    {
        std::lock_guard<std::mutex> Lock(ModulesMutex);
        Snapshot = Modules;
    }

    std::vector<HostRaceRecord> Races;
    std::uint64_t DroppedOccurrences = 0;
    std::unordered_set<RuntimeStateIdentity, RuntimeStateIdentityHash>
        SeenRuntimeStates;

    for (const TrackedModule &Module : Snapshot)
        collectFromModule(Module, SeenRuntimeStates, Races, DroppedOccurrences);

    printReport(Races, DroppedOccurrences);
    Reporting.store(false);
}
