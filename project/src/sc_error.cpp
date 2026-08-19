#include "sc_error.h"

#include "sc_runtime_abi.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <string>

#ifndef SC_DEVICE_RUNTIME_LL_PATH
#define SC_DEVICE_RUNTIME_LL_PATH ""
#endif

namespace supercollider {

namespace {

constexpr const char *SC_RUNTIME_STORAGE_SYMBOL =
    "__sc_runtime_state_storage";
constexpr const char *SC_SITE_METADATA_SYMBOL =
    "__sc_site_metadata";
constexpr const char *SC_ERROR_CALLBACK_SYMBOL = "sc_err";

llvm::GlobalVariable *getOrCreateRuntimeStorageDeclaration(llvm::Module &M) {
    if (llvm::GlobalVariable *Existing =
            M.getNamedGlobal(SC_RUNTIME_STORAGE_SYMBOL)) {
        return Existing;
    }

    llvm::LLVMContext &Context = M.getContext();
    llvm::ArrayType *StorageType = llvm::ArrayType::get(
        llvm::Type::getInt8Ty(Context),
        SC_RUNTIME_STATE_BYTES);

    return new llvm::GlobalVariable(
        M,
        StorageType,
        false,
        llvm::GlobalValue::ExternalLinkage,
        nullptr,
        SC_RUNTIME_STORAGE_SYMBOL,
        nullptr,
        llvm::GlobalValue::NotThreadLocal,
        SC_GLOBAL_ADDRESS_SPACE);
}

std::string sanitizeMetadataField(llvm::StringRef Input) {
    std::string Output;
    Output.reserve(Input.size());

    for (char C : Input) {
        if (C == '\t' || C == '\n' || C == '\r')
            Output.push_back(' ');
        else
            Output.push_back(C);
    }

    return Output;
}

const llvm::DILocation *getMetadataLocation(
    const llvm::DebugLoc &DebugLocation,
    SCRaceType RaceType) {

    const llvm::DILocation *Location = DebugLocation.get();

    // CUDA's low-level async helpers are commonly force-inlined. For async
    // sites, walk to the outermost call site so metadata points back to the
    // application source rather than a CUDA header implementation detail.
    if (RaceType == SCRaceType::AsyncCopy) {
        while (Location != nullptr && Location->getInlinedAt() != nullptr)
            Location = Location->getInlinedAt();
    }

    return Location;
}

std::string getSourcePath(const llvm::DILocation *Location) {
    if (Location == nullptr)
        return "<unknown>";

    const llvm::DILocalScope *Scope = Location->getScope();
    if (Scope == nullptr || Scope->getFile() == nullptr)
        return "<unknown>";

    llvm::StringRef Filename = Scope->getFilename();
    llvm::StringRef Directory = Scope->getDirectory();

    if (Directory.empty())
        return Filename.empty() ? "<unknown>" : Filename.str();

    if (Filename.empty())
        return Directory.str();

    std::string Path = Directory.str();
    if (!Path.empty() && Path.back() != '/')
        Path.push_back('/');
    Path += Filename.str();
    return Path;
}

llvm::Value *normalizeReportAddress(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Address) {

    if (Address == nullptr)
        return nullptr;

    if (Address->getType()->isPointerTy()) {
        return Builder.CreatePtrToInt(
            Address,
            Builder.getInt64Ty(),
            "sc.race.address");
    }

    if (Address->getType()->isIntegerTy()) {
        if (Address->getType()->isIntegerTy(64))
            return Address;

        return Builder.CreateZExtOrTrunc(
            Address,
            Builder.getInt64Ty(),
            "sc.race.address");
    }

    return nullptr;
}

} // namespace

SCErrorInsertion insertSCErrorReportForSite(
    RaceAssertResult &Assert,
    std::uint64_t SiteID,
    llvm::Value *Address,
    SCRaceType RaceType,
    const llvm::DebugLoc &DebugLocation) {

    SCErrorInsertion Result;

    if (Assert.FailureBlock == nullptr ||
        Assert.ContinuationBlock == nullptr ||
        Assert.TrapCall == nullptr ||
        Address == nullptr ||
        SiteID == 0) {
        return Result;
    }

    llvm::Function *ParentFunction = Assert.FailureBlock->getParent();
    if (ParentFunction == nullptr)
        return Result;

    llvm::Module *M = ParentFunction->getParent();
    if (M == nullptr)
        return Result;

    llvm::LLVMContext &Context = M->getContext();
    llvm::IRBuilder<> Builder(Assert.TrapCall);
    Builder.SetCurrentDebugLocation(DebugLocation);

    llvm::GlobalVariable *Storage =
        getOrCreateRuntimeStorageDeclaration(*M);

    llvm::Value *RuntimePointer = Builder.CreateAddrSpaceCast(
        Storage,
        llvm::PointerType::get(Context, SC_GENERIC_ADDRESS_SPACE),
        "sc.runtime.ptr");

    llvm::Value *AddressI64 = normalizeReportAddress(Builder, Address);
    if (AddressI64 == nullptr)
        return Result;

    llvm::FunctionType *CallbackType = llvm::FunctionType::get(
        Builder.getVoidTy(),
        {
            llvm::PointerType::get(Context, SC_GENERIC_ADDRESS_SPACE),
            Builder.getInt64Ty(),
            Builder.getInt64Ty(),
            Builder.getInt32Ty(),
        },
        false);

    llvm::FunctionCallee Callback = M->getOrInsertFunction(
        SC_ERROR_CALLBACK_SYMBOL,
        CallbackType);

    Result.ReportCall = Builder.CreateCall(
        Callback,
        {
            RuntimePointer,
            Builder.getInt64(SiteID),
            AddressI64,
            Builder.getInt32(static_cast<std::uint32_t>(RaceType)),
        });
    Result.ReportCall->setDebugLoc(DebugLocation);

    // race_assert.cpp deliberately creates trap+unreachable as a valid fallback.
    // Once sc_err has been inserted successfully, replace that fallback with a
    // normal edge back to the continuation so the kernel can report additional
    // race sites in the same execution.
    llvm::Instruction *Terminator = Assert.FailureBlock->getTerminator();
    Assert.TrapCall->eraseFromParent();
    Assert.TrapCall = nullptr;

    if (Terminator != nullptr)
        Terminator->eraseFromParent();

    llvm::IRBuilder<> ContinueBuilder(Assert.FailureBlock);
    ContinueBuilder.SetCurrentDebugLocation(DebugLocation);
    Result.ContinueBranch = ContinueBuilder.CreateBr(
        Assert.ContinuationBlock);

    return Result;
}

SCErrorInsertion insertSCErrorReport(
    RaceAssertResult &Assert,
    const WeakMemoryOperation &Operation,
    SCRaceType RaceType,
    const llvm::DebugLoc &DebugLocation) {

    if (Operation.Pointer == nullptr)
        return SCErrorInsertion{};

    return insertSCErrorReportForSite(
        Assert,
        Operation.SiteID,
        Operation.Pointer,
        RaceType,
        DebugLocation);
}

void emitSCReportingMetadata(
    llvm::Module &M,
    llvm::ArrayRef<SCReportingSite> Sites) {

    if (Sites.empty() || M.getNamedGlobal(SC_SITE_METADATA_SYMBOL) != nullptr)
        return;

    std::string Blob;
    llvm::raw_string_ostream Stream(Blob);

    // Text format:
    // site-id<TAB>race-type<TAB>line<TAB>column<TAB>function<TAB>file<NL>
    // A textual blob keeps the device/host ABI independent from C++ structure
    // padding and lets the host runtime copy the complete table in one transfer.
    for (const SCReportingSite &Site : Sites) {
        if (Site.SiteID == 0 || Site.Origin == nullptr)
            continue;

        llvm::DebugLoc DebugLocation = Site.Origin->getDebugLoc();
        const llvm::DILocation *Location = getMetadataLocation(
            DebugLocation,
            Site.RaceType);

        unsigned Line = Location != nullptr ? Location->getLine() : 0;
        unsigned Column = Location != nullptr ? Location->getColumn() : 0;

        std::string FunctionName = sanitizeMetadataField(
            Site.Origin->getFunction()->getName());
        std::string FileName = sanitizeMetadataField(
            getSourcePath(Location));

        Stream << Site.SiteID << '\t'
               << static_cast<std::uint32_t>(Site.RaceType) << '\t'
               << Line << '\t'
               << Column << '\t'
               << FunctionName << '\t'
               << FileName << '\n';
    }

    Stream.flush();

    llvm::LLVMContext &Context = M.getContext();
    llvm::Constant *Initializer = llvm::ConstantDataArray::getString(
        Context,
        Blob,
        false);

    auto *Metadata = new llvm::GlobalVariable(
        M,
        Initializer->getType(),
        true,
        llvm::GlobalValue::ExternalLinkage,
        Initializer,
        SC_SITE_METADATA_SYMBOL,
        nullptr,
        llvm::GlobalValue::NotThreadLocal,
        SC_GLOBAL_ADDRESS_SPACE);
    Metadata->setAlignment(llvm::Align(1));
}

bool linkSCDeviceRuntime(llvm::Module &M, std::string &ErrorMessage) {
    llvm::StringRef RuntimePath = SC_DEVICE_RUNTIME_LL_PATH;
    if (RuntimePath.empty()) {
        ErrorMessage = "device runtime path was not configured at plugin build time";
        return false;
    }

    llvm::SMDiagnostic Diagnostic;
    std::unique_ptr<llvm::Module> RuntimeModule = llvm::parseIRFile(
        RuntimePath,
        Diagnostic,
        M.getContext());

    if (!RuntimeModule) {
        std::string DiagnosticText;
        llvm::raw_string_ostream Stream(DiagnosticText);
        Diagnostic.print("SuperCollider", Stream);
        Stream.flush();
        ErrorMessage = "failed to read device reporting runtime '" +
                       RuntimePath.str() + "': " + DiagnosticText;
        return false;
    }

    if (llvm::Linker::linkModules(
            M,
            std::move(RuntimeModule),
            llvm::Linker::Flags::LinkOnlyNeeded)) {
        ErrorMessage = "failed to link the SuperCollider device reporting runtime";
        return false;
    }

    return true;
}

} // namespace supercollider
