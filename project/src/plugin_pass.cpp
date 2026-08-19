#include "async_memops.h"
#include "block_shuffle.h"
#include "config.h"
#include "debugging_prints.h"
#include "filter_memops.h"
#include "instrument_memops.h"
#include "intra_warp_lost_update.h"
#include "sc_error.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

using namespace llvm;
using namespace supercollider;

namespace {

class SuperColliderPass : public PassInfoMixin<SuperColliderPass> {
public:
    explicit SuperColliderPass(SCConfig Config = {})
        : Config(Config) {}

    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
        (void)MAM;

        errs() << "[SuperCollider] Processing module: "
               << M.getName() << "\n";
        printConfiguration(Config);

        // Freeze the application's original weak-memory, async-copy, and
        // blockIdx.x candidates before instrumentation adds synthetic memory
        // or special-register operations.
        SmallVector<WeakMemoryOperation, 64> Candidates;
        collectInstrumentableWeakMemoryOperations(M, Candidates);

        std::uint64_t NextSiteID = Candidates.empty()
                                       ? 1
                                       : Candidates.back().SiteID + 1;
        SmallVector<AsyncMemoryOperation, 16> AsyncCandidates;
        collectAsyncMemoryOperations(M, NextSiteID, AsyncCandidates);

        SmallVector<CallInst *, 32> BlockShuffleCandidates;
        BlockShuffleSummary BlockShuffleStats;
        if (Config.EnableBlockShuffle) {
            std::string BlockShuffleError;
            if (!collectBlockShuffleCandidates(
                    M,
                    BlockShuffleCandidates,
                    BlockShuffleStats,
                    BlockShuffleError)) {
                report_fatal_error(
                    Twine("SuperCollider block shuffling rejected module: ") +
                    BlockShuffleError);
            }
        }

        CandidateSummary CandidateStats;
        for (const WeakMemoryOperation &Operation : Candidates) {
            updateCandidateSummary(CandidateStats, Operation);

            const Function *F = Operation.Origin != nullptr
                                    ? Operation.Origin->getFunction()
                                    : nullptr;

            if (F != nullptr)
                printInstrumentableCandidate(*F, Operation);
        }

        printCandidateSummary(CandidateStats);

        IntraWarpLostUpdateSummary IntraWarpStats;
        bool IntraWarpChanged = false;
        if (Config.EnableIntraWarpLostUpdate) {
            // Insert the independent pre-store address-match detector before
            // the existing CR/LU expansion touches the original stores.
            IntraWarpChanged = instrumentIntraWarpLostUpdates(
                Candidates,
                IntraWarpStats);
        }

        AsyncInstrumentationSummary AsyncStats;
        bool AsyncChanged = instrumentAsyncMemoryOperations(
            AsyncCandidates,
            Config,
            AsyncStats);

        SmallVector<InstrumentedMemoryOperation, 64> Instrumented;
        InstrumentationSummary InstrumentationStats;

        bool OrdinaryChanged = instrumentWeakMemoryOperations(
            Candidates,
            Config,
            Instrumented,
            InstrumentationStats);

        bool DetectorChanged =
            IntraWarpChanged || AsyncChanged || OrdinaryChanged;

        if (DetectorChanged) {
            SmallVector<SCReportingSite, 96> ReportingSites;
            ReportingSites.reserve(Candidates.size() + AsyncCandidates.size());

            for (const WeakMemoryOperation &Operation : Candidates) {
                ReportingSites.push_back({
                    Operation.SiteID,
                    Operation.isLoad()
                        ? SCRaceType::ClobberedRead
                        : SCRaceType::LostUpdate,
                    Operation.Origin});
            }

            for (const AsyncMemoryOperation &Operation : AsyncCandidates) {
                ReportingSites.push_back({
                    Operation.SiteID,
                    SCRaceType::AsyncCopy,
                    Operation.Origin});
            }

            emitSCReportingMetadata(M, ReportingSites);

            // Link the CUDA-C++ sc_err implementation only after application
            // instrumentation is complete so runtime memory operations never
            // become detector candidates.
            std::string RuntimeError;
            if (!linkSCDeviceRuntime(M, RuntimeError)) {
                report_fatal_error(
                    Twine("SuperCollider device runtime link failed: ") +
                    RuntimeError);
            }
        }

        // Apply block shuffling only to the application's frozen blockIdx.x
        // reads. Detector-generated ctaid.x reads remain physical and cannot
        // recursively become shuffle candidates.
        bool BlockShuffleChanged = instrumentBlockShuffling(
            BlockShuffleCandidates,
            Config,
            BlockShuffleStats);

        bool Changed = DetectorChanged || BlockShuffleChanged;

        if (Config.EnableIntraWarpLostUpdate)
            printIntraWarpLostUpdateSummary(IntraWarpStats);

        printAsyncInstrumentationSummary(AsyncStats, Config);
        printBlockShuffleSummary(BlockShuffleStats, Config);

        for (const InstrumentedMemoryOperation &Operation : Instrumented)
            printInstrumentedMemoryOperation(Operation, Config);

        printInstrumentationSummary(InstrumentationStats);

        return Changed ? PreservedAnalyses::none()
                       : PreservedAnalyses::all();
    }

private:
    SCConfig Config;
};

SuperColliderPass parseConfiguredPass(StringRef Name) {
    if (Name == "supercollider")
        return SuperColliderPass();

    Expected<SCConfig> Config = PassBuilder::parsePassParameters(
        parseSuperColliderConfig,
        Name,
        "supercollider");

    if (!Config) {
        std::string Message = toString(Config.takeError());
        report_fatal_error(
            Twine("invalid SuperCollider pass configuration: ") + Message);
    }

    return SuperColliderPass(*Config);
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK
PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION,
        "SuperColliderPass",
        LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name,
                   ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "supercollider") {
                        MPM.addPass(SuperColliderPass());
                        return true;
                    }

                    if (!PassBuilder::checkParametrizedPassName(
                            Name,
                            "supercollider")) {
                        return false;
                    }

                    MPM.addPass(parseConfiguredPass(Name));
                    return true;
                });
        }
    };
}
