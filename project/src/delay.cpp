#include "delay.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include <cstdint>

namespace supercollider {

namespace {

llvm::Value *readNVVMSpecialRegister(
    llvm::IRBuilder<> &Builder,
    llvm::Module &M,
    llvm::Intrinsic::ID IntrinsicID,
    llvm::StringRef Name) {

    llvm::Function *Declaration =
        llvm::Intrinsic::getOrInsertDeclaration(&M, IntrinsicID);
    return Builder.CreateCall(Declaration, {}, Name);
}

llvm::Value *asI64(llvm::IRBuilder<> &Builder, llvm::Value *Value) {
    llvm::Type *I64 = Builder.getInt64Ty();

    if (Value->getType() == I64)
        return Value;

    return Builder.CreateZExtOrTrunc(Value, I64);
}

// Linearize a three-dimensional coordinate with row-major X-fastest layout:
//
//   x + extentX * (y + extentY * z)
//
// All arithmetic is performed in i64 so large grids cannot wrap in 32 bits.
llvm::Value *linearize3D(
    llvm::IRBuilder<> &Builder,
    llvm::Value *X,
    llvm::Value *Y,
    llvm::Value *Z,
    llvm::Value *ExtentX,
    llvm::Value *ExtentY,
    llvm::StringRef Name) {

    llvm::Value *X64 = asI64(Builder, X);
    llvm::Value *Y64 = asI64(Builder, Y);
    llvm::Value *Z64 = asI64(Builder, Z);
    llvm::Value *ExtentX64 = asI64(Builder, ExtentX);
    llvm::Value *ExtentY64 = asI64(Builder, ExtentY);

    llvm::Value *YZ = Builder.CreateAdd(
        Y64,
        Builder.CreateMul(ExtentY64, Z64));

    return Builder.CreateAdd(
        X64,
        Builder.CreateMul(ExtentX64, YZ),
        Name);
}

llvm::Value *multiply3(
    llvm::IRBuilder<> &Builder,
    llvm::Value *X,
    llvm::Value *Y,
    llvm::Value *Z,
    llvm::StringRef Name) {

    llvm::Value *XY = Builder.CreateMul(
        asI64(Builder, X),
        asI64(Builder, Y));

    return Builder.CreateMul(
        XY,
        asI64(Builder, Z),
        Name);
}

// Fold one runtime integer into the seed. This stage combines structurally
// different execution properties cheaply; the SplitMix64 finalizer below then
// supplies the strong avalanche behavior.
llvm::Value *combineSeed(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Seed,
    llvm::Value *Input) {

    constexpr std::uint64_t GoldenRatio64 = 0x9e3779b97f4a7c15ULL;

    llvm::Value *Term = Builder.CreateAdd(
        asI64(Builder, Input),
        Builder.getInt64(GoldenRatio64));

    Term = Builder.CreateAdd(
        Term,
        Builder.CreateShl(Seed, Builder.getInt64(6)));
    Term = Builder.CreateAdd(
        Term,
        Builder.CreateLShr(Seed, Builder.getInt64(2)));

    return Builder.CreateXor(Seed, Term);
}

// SplitMix64 finalizer. Arithmetic intentionally wraps modulo 2^64; no nsw/nuw
// flags are attached. The finalizer gives strong avalanche behavior for nearby
// thread IDs and adjacent instrumentation-site IDs without maintaining PRNG
// state.
llvm::Value *finalizeSplitMix64(
    llvm::IRBuilder<> &Builder,
    llvm::Value *Input) {

    llvm::Value *X = Input;

    X = Builder.CreateXor(
        X,
        Builder.CreateLShr(X, Builder.getInt64(30)));
    X = Builder.CreateMul(
        X,
        Builder.getInt64(0xbf58476d1ce4e5b9ULL));

    X = Builder.CreateXor(
        X,
        Builder.CreateLShr(X, Builder.getInt64(27)));
    X = Builder.CreateMul(
        X,
        Builder.getInt64(0x94d049bb133111ebULL));

    X = Builder.CreateXor(
        X,
        Builder.CreateLShr(X, Builder.getInt64(31)),
        "sc.delay.hash");

    return X;
}

} // namespace

RuntimeDelay insertRuntimeDelay(
    llvm::Instruction &InsertBefore,
    std::uint32_t MaximumDelayNs,
    std::uint64_t SiteID,
    const llvm::DebugLoc &DebugLocation) {

    llvm::Module *M = InsertBefore.getFunction()->getParent();
    llvm::IRBuilder<> Builder(&InsertBefore);
    Builder.SetCurrentDebugLocation(DebugLocation);

    RuntimeDelay Result;

    if (MaximumDelayNs == 0) {
        Result.DelayNs = Builder.getInt32(0);
        return Result;
    }

    // Read the complete XYZ CUDA execution coordinates and dimensions needed
    // to derive a unique linear thread identity for the current grid.
    llvm::Value *TidX = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x, "sc.tid.x");
    llvm::Value *TidY = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y, "sc.tid.y");
    llvm::Value *TidZ = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z, "sc.tid.z");

    llvm::Value *BlockDimX = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x, "sc.ntid.x");
    llvm::Value *BlockDimY = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y, "sc.ntid.y");
    llvm::Value *BlockDimZ = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z, "sc.ntid.z");

    llvm::Value *BlockX = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x, "sc.ctaid.x");
    llvm::Value *BlockY = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y, "sc.ctaid.y");
    llvm::Value *BlockZ = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z, "sc.ctaid.z");

    llvm::Value *GridDimX = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x, "sc.nctaid.x");
    llvm::Value *GridDimY = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_y, "sc.nctaid.y");
    llvm::Value *GridDimZ = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_z, "sc.nctaid.z");

    llvm::Value *LocalThread = linearize3D(
        Builder,
        TidX,
        TidY,
        TidZ,
        BlockDimX,
        BlockDimY,
        "sc.thread.local");

    llvm::Value *LinearBlock = linearize3D(
        Builder,
        BlockX,
        BlockY,
        BlockZ,
        GridDimX,
        GridDimY,
        "sc.block.linear");

    llvm::Value *ThreadsPerBlock = multiply3(
        Builder,
        BlockDimX,
        BlockDimY,
        BlockDimZ,
        "sc.threads.per.block");

    llvm::Value *BlocksInGrid = multiply3(
        Builder,
        GridDimX,
        GridDimY,
        GridDimZ,
        "sc.blocks.in.grid");

    llvm::Value *GlobalThread = Builder.CreateAdd(
        LocalThread,
        Builder.CreateMul(ThreadsPerBlock, LinearBlock),
        "sc.thread.global");

    // Add runtime placement and timing information. clock64/globaltimer vary
    // across dynamic executions, while lane/warp/SM and grid ID capture the
    // execution placement and launch in which this site is currently running.
    llvm::Value *LaneID = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_laneid, "sc.laneid");
    llvm::Value *WarpID = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_warpid, "sc.warpid");
    llvm::Value *SMID = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_smid, "sc.smid");
    llvm::Value *GridID = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_gridid, "sc.gridid");
    llvm::Value *Clock64 = readNVVMSpecialRegister(
        Builder, *M, llvm::Intrinsic::nvvm_read_ptx_sreg_clock64, "sc.clock64");
    llvm::Value *GlobalTimer = readNVVMSpecialRegister(
        Builder,
        *M,
        llvm::Intrinsic::nvvm_read_ptx_sreg_globaltimer,
        "sc.globaltimer");

    llvm::Value *Seed = Builder.getInt64(
        SiteID ^ 0x243f6a8885a308d3ULL);

    Seed = combineSeed(Builder, Seed, GlobalThread);
    Seed = combineSeed(Builder, Seed, ThreadsPerBlock);
    Seed = combineSeed(Builder, Seed, BlocksInGrid);
    Seed = combineSeed(Builder, Seed, LaneID);
    Seed = combineSeed(Builder, Seed, WarpID);
    Seed = combineSeed(Builder, Seed, SMID);
    Seed = combineSeed(Builder, Seed, GridID);
    Seed = combineSeed(Builder, Seed, Clock64);
    Seed = combineSeed(Builder, Seed, GlobalTimer);

    llvm::Value *Hash = finalizeSplitMix64(Builder, Seed);

    // Compute the modulus in i64 so MaximumDelayNs == UINT32_MAX still has the
    // valid bound 2^32 rather than wrapping back to zero in 32-bit arithmetic.
    std::uint64_t BoundValue =
        static_cast<std::uint64_t>(MaximumDelayNs) + 1ULL;
    llvm::Value *Delay64 = Builder.CreateURem(
        Hash,
        Builder.getInt64(BoundValue),
        "sc.delay.bounded");

    Result.DelayNs = Builder.CreateTrunc(
        Delay64,
        Builder.getInt32Ty(),
        "sc.delay.ns");

    llvm::Function *NanoSleep = llvm::Intrinsic::getOrInsertDeclaration(
        M,
        llvm::Intrinsic::nvvm_nanosleep);

    Result.SleepCall = Builder.CreateCall(
        NanoSleep,
        {Result.DelayNs});
    Result.SleepCall->setDebugLoc(DebugLocation);

    return Result;
}

} // namespace supercollider
