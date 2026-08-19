#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#ifndef SC_EXPECT_SHUFFLE
#define SC_EXPECT_SHUFFLE 1
#endif

#ifndef SC_TEST_SHUFFLE_SEED
#define SC_TEST_SHUFFLE_SEED 12345ULL
#endif

extern "C" __device__ unsigned sc_test_physical_block_x();
extern "C" __device__ std::uint64_t sc_test_gridid();
extern "C" __device__ unsigned sc_test_prepass_helper_block_x();
extern "C" __device__ std::uint64_t
sc_test_prepass_helper_linear_block(std::uint64_t scale, std::uint64_t bias);

namespace {

constexpr std::uint64_t kPrimeTable[8] = {
    3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL,
};

constexpr std::uint64_t kAHashSalt = 0x243f6a8885a308d3ULL;
constexpr std::uint64_t kBHashSalt = 0x13198a2e03707344ULL;

void failCuda(cudaError_t error, const char *expr, const char *file, int line) {
    if (error == cudaSuccess)
        return;

    std::fprintf(
        stderr,
        "CUDA ERROR at %s:%d: %s -> %s\n",
        file,
        line,
        expr,
        cudaGetErrorString(error));
    std::exit(2);
}

#define CUDA_CHECK(expr) failCuda((expr), #expr, __FILE__, __LINE__)

struct ExpectedAffine {
    std::uint64_t a = 1;
    std::uint64_t b = 0;
    std::uint64_t selectedPrime = 0;
    bool usedFallback = false;
};

std::uint64_t splitmix64Finalizer(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

ExpectedAffine expectedAffine(
    std::uint64_t gridId,
    std::uint64_t gridX,
    std::uint64_t shuffleSeed) {

    ExpectedAffine result;
    if (gridX <= 1)
        return result;

    const std::uint64_t seed = gridId ^ shuffleSeed;
    const std::uint64_t aHash = splitmix64Finalizer(seed ^ kAHashSalt);
    const std::uint64_t bHash = splitmix64Finalizer(seed ^ kBHashSalt);

    result.selectedPrime = kPrimeTable[aHash & 7ULL];
    result.usedFallback = (gridX % result.selectedPrime) == 0;
    result.a = result.usedFallback ? gridX - 1ULL : result.selectedPrime;
    result.b = bHash % gridX;
    return result;
}

std::uint32_t expectedLogical(
    std::uint32_t physical,
    std::uint32_t gridX,
    std::uint64_t gridId) {

#if SC_EXPECT_SHUFFLE
    if (gridX <= 1)
        return physical;

    const ExpectedAffine affine = expectedAffine(
        gridId,
        gridX,
        static_cast<std::uint64_t>(SC_TEST_SHUFFLE_SEED));
    return static_cast<std::uint32_t>(
        (affine.a * static_cast<std::uint64_t>(physical) + affine.b) % gridX);
#else
    (void)gridId;
    return physical;
#endif
}

std::uint64_t mapSignature(const std::vector<unsigned> &mapping) {
    std::uint64_t h = 14695981039346656037ULL;
    for (unsigned value : mapping) {
        h ^= static_cast<std::uint64_t>(value);
        h *= 1099511628211ULL;
    }
    return h;
}

} // namespace

// These kernels intentionally have external C linkage. The test compiles the
// host and device halves in separate Clang invocations while device code uses
// relocatable-device-code mode for the post-pass helper. Keeping the kernels
// out of the anonymous namespace makes the host registration names identical
// to the final PTX entry names regardless of per-invocation CUDA CUID values.
extern "C" __global__ void mappingKernel(
    unsigned *logicalByPhysical,
    std::uint64_t *gridIdByPhysical) {

    const unsigned physical = sc_test_physical_block_x();
    const unsigned logical = blockIdx.x;
    const std::uint64_t gridId = sc_test_gridid();

    if (threadIdx.x == 0) {
        logicalByPhysical[physical] = logical;
        gridIdByPhysical[physical] = gridId;
    }
}

extern "C" __global__ void perThreadConsistencyKernel(
    unsigned *logicalByPhysicalThread,
    unsigned *threadIdxObserved,
    unsigned *blockDimObserved,
    unsigned *gridDimObserved,
    std::uint64_t *gridIdObserved) {

    const unsigned physical = sc_test_physical_block_x();
    const unsigned slot = physical * blockDim.x + threadIdx.x;

    logicalByPhysicalThread[slot] = blockIdx.x;
    threadIdxObserved[slot] = threadIdx.x;
    blockDimObserved[slot] = blockDim.x;
    gridDimObserved[slot] = gridDim.x;
    gridIdObserved[slot] = sc_test_gridid();
}

extern "C" __global__ void directIndexKernel(int *out, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count)
        out[i] = i * 7 + 3;
}

extern "C" __global__ void partialBoundsKernel(unsigned *hits, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count)
        atomicAdd(&hits[i], 1u);
}

extern "C" __global__ void gridStrideKernel(unsigned *hits, int count) {
    const int start = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(blockDim.x * gridDim.x);
    for (int i = start; i < count; i += stride)
        atomicAdd(&hits[i], 1u);
}

extern "C" __global__ void branchSwitchKernel(
    unsigned *branchByPhysical,
    unsigned *switchByPhysical) {

    const unsigned physical = sc_test_physical_block_x();
    const unsigned logical = blockIdx.x;

    unsigned branchValue = 0;
    if ((logical & 1u) == 0u)
        branchValue = 100u + logical;
    else
        branchValue = 200u + logical;

    unsigned switchValue = 0;
    switch (logical % 4u) {
    case 0:
        switchValue = 11u;
        break;
    case 1:
        switchValue = 22u;
        break;
    case 2:
        switchValue = 33u;
        break;
    default:
        switchValue = 44u;
        break;
    }

    if (threadIdx.x == 0) {
        branchByPhysical[physical] = branchValue;
        switchByPhysical[physical] = switchValue;
    }
}

extern "C" __global__ void sharedBarrierKernel(unsigned *mismatchByPhysical) {
    __shared__ unsigned logicalShared;

    const unsigned physical = sc_test_physical_block_x();
    if (threadIdx.x == 0)
        logicalShared = blockIdx.x;

    __syncthreads();

    if (logicalShared != blockIdx.x)
        atomicExch(&mismatchByPhysical[physical], 1u);
}

extern "C" __global__ void helperKernel(
    unsigned *helperValueByPhysical,
    std::uint64_t *linearValueByPhysical,
    std::uint64_t *gridIdByPhysical) {

    const unsigned physical = sc_test_physical_block_x();
    if (threadIdx.x == 0) {
        helperValueByPhysical[physical] = sc_test_prepass_helper_block_x();
        linearValueByPhysical[physical] =
            sc_test_prepass_helper_linear_block(97ULL, 13ULL);
        gridIdByPhysical[physical] = sc_test_gridid();
    }
}

extern "C" __global__ void threeDimensionalKernel(
    unsigned *logicalXByPhysical,
    unsigned *logicalYByPhysical,
    unsigned *logicalZByPhysical,
    std::uint64_t *gridIdByPhysical) {

    if (threadIdx.x != 0)
        return;

    const unsigned physicalX = sc_test_physical_block_x();
    const unsigned physicalSlot =
        physicalX + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);

    const dim3 observed = blockIdx;
    logicalXByPhysical[physicalSlot] = observed.x;
    logicalYByPhysical[physicalSlot] = observed.y;
    logicalZByPhysical[physicalSlot] = observed.z;
    gridIdByPhysical[physicalSlot] = sc_test_gridid();
}

namespace {

bool validateMapping(
    unsigned gridX,
    const std::vector<unsigned> &mapping,
    const std::vector<std::uint64_t> &gridIds,
    bool verbose,
    bool *fallbackObserved = nullptr) {

    if (mapping.size() != gridX || gridIds.size() != gridX)
        return false;

    const std::uint64_t gridId = gridIds[0];
    for (std::uint64_t value : gridIds) {
        if (value != gridId) {
            std::fprintf(stderr, "FAIL: grid %u observed inconsistent %%gridid values\n", gridX);
            return false;
        }
    }

    std::vector<unsigned> seen(gridX, 0u);
    for (unsigned physical = 0; physical < gridX; ++physical) {
        const unsigned logical = mapping[physical];
        if (logical >= gridX) {
            std::fprintf(
                stderr,
                "FAIL: grid %u physical %u produced out-of-range logical %u\n",
                gridX,
                physical,
                logical);
            return false;
        }

        ++seen[logical];
        const unsigned expected = expectedLogical(physical, gridX, gridId);
        if (logical != expected) {
            std::fprintf(
                stderr,
                "FAIL: grid %u physical %u -> logical %u, expected %u, gridid=%llu\n",
                gridX,
                physical,
                logical,
                expected,
                static_cast<unsigned long long>(gridId));
            return false;
        }
    }

    for (unsigned logical = 0; logical < gridX; ++logical) {
        if (seen[logical] != 1u) {
            std::fprintf(
                stderr,
                "FAIL: grid %u logical block %u appeared %u times\n",
                gridX,
                logical,
                seen[logical]);
            return false;
        }
    }

#if SC_EXPECT_SHUFFLE
    const ExpectedAffine affine = expectedAffine(
        gridId,
        gridX,
        static_cast<std::uint64_t>(SC_TEST_SHUFFLE_SEED));
    if (fallbackObserved != nullptr)
        *fallbackObserved = *fallbackObserved || affine.usedFallback;

    if (gridX > 1 && std::gcd(affine.a, static_cast<std::uint64_t>(gridX)) != 1ULL) {
        std::fprintf(stderr, "FAIL: gcd(a,N) != 1 for grid %u\n", gridX);
        return false;
    }
#endif

    if (verbose) {
        std::printf(
            "  N=%-4u gridid=%-8llu sig=0x%016llx",
            gridX,
            static_cast<unsigned long long>(gridId),
            static_cast<unsigned long long>(mapSignature(mapping)));
#if SC_EXPECT_SHUFFLE
        const ExpectedAffine affine = expectedAffine(
            gridId,
            gridX,
            static_cast<std::uint64_t>(SC_TEST_SHUFFLE_SEED));
        if (gridX > 1) {
            std::printf(
                " a=%llu b=%llu prime=%llu%s",
                static_cast<unsigned long long>(affine.a),
                static_cast<unsigned long long>(affine.b),
                static_cast<unsigned long long>(affine.selectedPrime),
                affine.usedFallback ? " fallback" : "");
        }
#endif
        std::printf("\n");

        if (gridX <= 16) {
            std::printf("    physical -> logical: ");
            for (unsigned i = 0; i < gridX; ++i)
                std::printf("%u->%u%s", i, mapping[i], i + 1 == gridX ? "" : " ");
            std::printf("\n");
        }
    }

    return true;
}

bool runMappingSweep() {
    const std::vector<unsigned> gridSizes = {
        1u, 2u, 3u, 4u, 5u, 7u, 8u, 11u, 13u, 16u, 17u, 19u,
        23u, 31u, 32u, 33u, 37u, 63u, 64u, 65u, 127u, 128u, 255u,
        256u, 257u, 1155u,
    };

    const unsigned maxGrid = *std::max_element(gridSizes.begin(), gridSizes.end());
    unsigned *dMap = nullptr;
    std::uint64_t *dGridIds = nullptr;
    CUDA_CHECK(cudaMalloc(&dMap, maxGrid * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dGridIds, maxGrid * sizeof(std::uint64_t)));

    bool fallbackObserved = false;
    bool ok = true;

    std::printf("[1] Affine/bijection sweep\n");
    for (unsigned gridX : gridSizes) {
        CUDA_CHECK(cudaMemset(dMap, 0xff, gridX * sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(dGridIds, 0, gridX * sizeof(std::uint64_t)));

        mappingKernel<<<gridX, 1>>>(dMap, dGridIds);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<unsigned> mapping(gridX);
        std::vector<std::uint64_t> gridIds(gridX);
        CUDA_CHECK(cudaMemcpy(
            mapping.data(), dMap, gridX * sizeof(unsigned), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            gridIds.data(), dGridIds, gridX * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));

        ok = validateMapping(
                 gridX,
                 mapping,
                 gridIds,
                 gridX <= 16 || gridX == 37 || gridX == 1155,
                 &fallbackObserved) &&
             ok;
    }

#if SC_EXPECT_SHUFFLE
    std::printf(
        "  fallback a=N-1 branch observed during sweep: %s\n",
        fallbackObserved ? "YES" : "NO (not required for correctness in this run)");
#endif

    CUDA_CHECK(cudaFree(dMap));
    CUDA_CHECK(cudaFree(dGridIds));
    return ok;
}

bool runSuccessiveLaunchTest() {
    constexpr unsigned gridX = 37;
    constexpr int launches = 6;

    unsigned *dMap = nullptr;
    std::uint64_t *dGridIds = nullptr;
    CUDA_CHECK(cudaMalloc(&dMap, gridX * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dGridIds, gridX * sizeof(std::uint64_t)));

    std::set<std::uint64_t> gridIdSet;
    std::set<std::uint64_t> signatureSet;
    bool ok = true;

    std::printf("[2] Successive launches / per-grid consistency\n");
    for (int launch = 0; launch < launches; ++launch) {
        mappingKernel<<<gridX, 1>>>(dMap, dGridIds);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<unsigned> mapping(gridX);
        std::vector<std::uint64_t> gridIds(gridX);
        CUDA_CHECK(cudaMemcpy(
            mapping.data(), dMap, gridX * sizeof(unsigned), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            gridIds.data(), dGridIds, gridX * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));

        ok = validateMapping(gridX, mapping, gridIds, false) && ok;
        gridIdSet.insert(gridIds[0]);
        signatureSet.insert(mapSignature(mapping));

        std::printf(
            "  launch %d: gridid=%llu sig=0x%016llx\n",
            launch,
            static_cast<unsigned long long>(gridIds[0]),
            static_cast<unsigned long long>(mapSignature(mapping)));
    }

    if (gridIdSet.size() != launches) {
        std::fprintf(
            stderr,
            "FAIL: expected distinct %%gridid values for %d launches, observed %zu\n",
            launches,
            gridIdSet.size());
        ok = false;
    }

#if SC_EXPECT_SHUFFLE
    std::printf("  distinct mappings observed: %zu/%d\n", signatureSet.size(), launches);
#else
    if (signatureSet.size() != 1) {
        std::fprintf(stderr, "FAIL: shuffle=0 changed identity mapping between launches\n");
        ok = false;
    }
#endif

    CUDA_CHECK(cudaFree(dMap));
    CUDA_CHECK(cudaFree(dGridIds));
    return ok;
}

bool runConcurrentStreamTest() {
    constexpr unsigned n0 = 37;
    constexpr unsigned n1 = 41;

    cudaStream_t s0 = nullptr;
    cudaStream_t s1 = nullptr;
    CUDA_CHECK(cudaStreamCreate(&s0));
    CUDA_CHECK(cudaStreamCreate(&s1));

    unsigned *dMap0 = nullptr;
    unsigned *dMap1 = nullptr;
    std::uint64_t *dGrid0 = nullptr;
    std::uint64_t *dGrid1 = nullptr;
    CUDA_CHECK(cudaMalloc(&dMap0, n0 * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dMap1, n1 * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dGrid0, n0 * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&dGrid1, n1 * sizeof(std::uint64_t)));

    std::printf("[3] Concurrent streams\n");
    mappingKernel<<<n0, 1, 0, s0>>>(dMap0, dGrid0);
    CUDA_CHECK(cudaGetLastError());
    mappingKernel<<<n1, 1, 0, s1>>>(dMap1, dGrid1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s0));
    CUDA_CHECK(cudaStreamSynchronize(s1));

    std::vector<unsigned> map0(n0), map1(n1);
    std::vector<std::uint64_t> grid0(n0), grid1(n1);
    CUDA_CHECK(cudaMemcpy(map0.data(), dMap0, n0 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(map1.data(), dMap1, n1 * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid0.data(), dGrid0, n0 * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(grid1.data(), dGrid1, n1 * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));

    bool ok = validateMapping(n0, map0, grid0, false) &&
              validateMapping(n1, map1, grid1, false);

    if (grid0[0] == grid1[0]) {
        std::fprintf(stderr, "FAIL: concurrent grids observed identical %%gridid\n");
        ok = false;
    }

    std::printf(
        "  stream0 gridid=%llu, stream1 gridid=%llu\n",
        static_cast<unsigned long long>(grid0[0]),
        static_cast<unsigned long long>(grid1[0]));

    CUDA_CHECK(cudaFree(dMap0));
    CUDA_CHECK(cudaFree(dMap1));
    CUDA_CHECK(cudaFree(dGrid0));
    CUDA_CHECK(cudaFree(dGrid1));
    CUDA_CHECK(cudaStreamDestroy(s0));
    CUDA_CHECK(cudaStreamDestroy(s1));
    return ok;
}

bool runPerThreadConsistencyTest() {
    constexpr unsigned gridX = 17;
    constexpr unsigned blockX = 64;
    constexpr unsigned count = gridX * blockX;

    unsigned *dLogical = nullptr;
    unsigned *dThread = nullptr;
    unsigned *dBlockDim = nullptr;
    unsigned *dGridDim = nullptr;
    std::uint64_t *dGridId = nullptr;

    CUDA_CHECK(cudaMalloc(&dLogical, count * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dThread, count * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dBlockDim, count * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dGridDim, count * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dGridId, count * sizeof(std::uint64_t)));

    perThreadConsistencyKernel<<<gridX, blockX>>>(
        dLogical, dThread, dBlockDim, dGridDim, dGridId);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned> logical(count), thread(count), blockDim(count), gridDim(count);
    std::vector<std::uint64_t> gridId(count);
    CUDA_CHECK(cudaMemcpy(logical.data(), dLogical, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(thread.data(), dThread, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(blockDim.data(), dBlockDim, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gridDim.data(), dGridDim, count * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gridId.data(), dGridId, count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));

    bool ok = true;
    const std::uint64_t launchGridId = gridId[0];
    for (unsigned physical = 0; physical < gridX; ++physical) {
        const unsigned expected = expectedLogical(physical, gridX, launchGridId);
        for (unsigned t = 0; t < blockX; ++t) {
            const unsigned slot = physical * blockX + t;
            if (logical[slot] != expected || thread[slot] != t ||
                blockDim[slot] != blockX || gridDim[slot] != gridX ||
                gridId[slot] != launchGridId) {
                std::fprintf(
                    stderr,
                    "FAIL: per-thread consistency at physical=%u thread=%u\n",
                    physical,
                    t);
                ok = false;
                goto done;
            }
        }
    }

done:
    std::printf("[4] CTA-wide consistency / unchanged threadIdx, blockDim, gridDim: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dLogical));
    CUDA_CHECK(cudaFree(dThread));
    CUDA_CHECK(cudaFree(dBlockDim));
    CUDA_CHECK(cudaFree(dGridDim));
    CUDA_CHECK(cudaFree(dGridId));
    return ok;
}

bool runIndexingTests() {
    constexpr int count = 1003;
    constexpr unsigned blockX = 64;
    constexpr unsigned gridX = (count + blockX - 1) / blockX;

    int *dOut = nullptr;
    unsigned *dHits = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dHits, count * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(dOut, 0xff, count * sizeof(int)));
    CUDA_CHECK(cudaMemset(dHits, 0, count * sizeof(unsigned)));

    directIndexKernel<<<gridX, blockX>>>(dOut, count);
    CUDA_CHECK(cudaGetLastError());
    partialBoundsKernel<<<gridX, blockX>>>(dHits, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> out(count);
    std::vector<unsigned> hits(count);
    CUDA_CHECK(cudaMemcpy(out.data(), dOut, count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hits.data(), dHits, count * sizeof(unsigned), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < count; ++i) {
        if (out[i] != i * 7 + 3 || hits[i] != 1u) {
            std::fprintf(
                stderr,
                "FAIL: direct/partial indexing at i=%d out=%d hits=%u\n",
                i,
                out[i],
                hits[i]);
            ok = false;
            break;
        }
    }

    CUDA_CHECK(cudaMemset(dHits, 0, count * sizeof(unsigned)));
    gridStrideKernel<<<7, 53>>>(dHits, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hits.data(), dHits, count * sizeof(unsigned), cudaMemcpyDeviceToHost));

    for (int i = 0; i < count && ok; ++i) {
        if (hits[i] != 1u) {
            std::fprintf(stderr, "FAIL: grid-stride indexing at i=%d hits=%u\n", i, hits[i]);
            ok = false;
        }
    }

    std::printf("[5] Direct indexing / partial block / grid-stride loop: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dOut));
    CUDA_CHECK(cudaFree(dHits));
    return ok;
}

bool runControlFlowAndSharedTest() {
    constexpr unsigned gridX = 29;
    constexpr unsigned blockX = 32;

    unsigned *dBranch = nullptr;
    unsigned *dSwitch = nullptr;
    unsigned *dMismatch = nullptr;
    CUDA_CHECK(cudaMalloc(&dBranch, gridX * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dSwitch, gridX * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dMismatch, gridX * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(dMismatch, 0, gridX * sizeof(unsigned)));

    branchSwitchKernel<<<gridX, blockX>>>(dBranch, dSwitch);
    CUDA_CHECK(cudaGetLastError());
    sharedBarrierKernel<<<gridX, blockX>>>(dMismatch);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned> branch(gridX), sw(gridX), mismatch(gridX);
    CUDA_CHECK(cudaMemcpy(branch.data(), dBranch, gridX * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sw.data(), dSwitch, gridX * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mismatch.data(), dMismatch, gridX * sizeof(unsigned), cudaMemcpyDeviceToHost));

    unsigned *dMap = nullptr;
    std::uint64_t *dGridIds = nullptr;
    CUDA_CHECK(cudaMalloc(&dMap, gridX * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dGridIds, gridX * sizeof(std::uint64_t)));
    mappingKernel<<<gridX, 1>>>(dMap, dGridIds);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned> map(gridX);
    std::vector<std::uint64_t> gridIds(gridX);
    CUDA_CHECK(cudaMemcpy(map.data(), dMap, gridX * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gridIds.data(), dGridIds, gridX * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));

    // branchSwitchKernel and mappingKernel are separate launches, so their
    // affine mappings can differ. Validate branch/switch values internally by
    // their encoded logical value rather than against mappingKernel's launch.
    bool ok = true;
    std::vector<unsigned> branchLogicalSeen(gridX, 0u);
    for (unsigned physical = 0; physical < gridX; ++physical) {
        const unsigned encoded = branch[physical];
        const unsigned logical = encoded >= 200u ? encoded - 200u : encoded - 100u;
        if (logical >= gridX ||
            ((logical & 1u) == 0u ? encoded != 100u + logical : encoded != 200u + logical)) {
            std::fprintf(stderr, "FAIL: branch control flow physical=%u encoded=%u\n", physical, encoded);
            ok = false;
            break;
        }
        ++branchLogicalSeen[logical];

        const unsigned expectedSwitch[4] = {11u, 22u, 33u, 44u};
        if (sw[physical] != expectedSwitch[logical % 4u]) {
            std::fprintf(stderr, "FAIL: switch control flow physical=%u\n", physical);
            ok = false;
            break;
        }
    }

    for (unsigned logical = 0; logical < gridX && ok; ++logical) {
        if (branchLogicalSeen[logical] != 1u) {
            std::fprintf(stderr, "FAIL: control-flow logical block %u appeared %u times\n", logical, branchLogicalSeen[logical]);
            ok = false;
        }
    }

    for (unsigned value : mismatch) {
        if (value != 0u) {
            std::fprintf(stderr, "FAIL: __syncthreads/shared CTA logical identity mismatch\n");
            ok = false;
            break;
        }
    }

    std::printf("[6] Branch/switch uses + shared memory/__syncthreads: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dBranch));
    CUDA_CHECK(cudaFree(dSwitch));
    CUDA_CHECK(cudaFree(dMismatch));
    CUDA_CHECK(cudaFree(dMap));
    CUDA_CHECK(cudaFree(dGridIds));
    return ok;
}

bool runPrepassDeviceHelperTest() {
    constexpr unsigned gridX = 31;

    unsigned *dHelper = nullptr;
    std::uint64_t *dLinear = nullptr;
    std::uint64_t *dGridIds = nullptr;
    CUDA_CHECK(cudaMalloc(&dHelper, gridX * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dLinear, gridX * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&dGridIds, gridX * sizeof(std::uint64_t)));

    helperKernel<<<gridX, 1>>>(dHelper, dLinear, dGridIds);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned> helper(gridX);
    std::vector<std::uint64_t> linear(gridX);
    std::vector<std::uint64_t> gridIds(gridX);
    CUDA_CHECK(cudaMemcpy(helper.data(), dHelper, gridX * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(linear.data(), dLinear, gridX * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gridIds.data(), dGridIds, gridX * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));

    bool ok = true;
    std::vector<unsigned> seen(gridX, 0u);
    const std::uint64_t launchGridId = gridIds[0];
    for (unsigned physical = 0; physical < gridX; ++physical) {
        const unsigned expected = expectedLogical(physical, gridX, launchGridId);
        if (gridIds[physical] != launchGridId || helper[physical] != expected) {
            std::fprintf(stderr, "FAIL: cross-TU pre-pass helper physical=%u got=%u expected=%u\n", physical, helper[physical], expected);
            ok = false;
            break;
        }
        ++seen[helper[physical]];
        if (linear[physical] != static_cast<std::uint64_t>(helper[physical]) * 97ULL + 13ULL) {
            std::fprintf(stderr, "FAIL: cross-TU pre-pass helper linear computation\n");
            ok = false;
            break;
        }
    }
    for (unsigned logical = 0; logical < gridX && ok; ++logical) {
        if (seen[logical] != 1u) {
            std::fprintf(stderr, "FAIL: cross-TU helper did not see logical block permutation\n");
            ok = false;
        }
    }

    std::printf("[7] noinline device helper in separate TU linked BEFORE pass: %s\n", ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dHelper));
    CUDA_CHECK(cudaFree(dLinear));
    CUDA_CHECK(cudaFree(dGridIds));
    return ok;
}

bool runThreeDimensionalTest(dim3 grid) {
    const unsigned totalBlocks = grid.x * grid.y * grid.z;

    unsigned *dX = nullptr;
    unsigned *dY = nullptr;
    unsigned *dZ = nullptr;
    std::uint64_t *dGridId = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, totalBlocks * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dY, totalBlocks * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dZ, totalBlocks * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&dGridId, totalBlocks * sizeof(std::uint64_t)));

    threeDimensionalKernel<<<grid, 1>>>(dX, dY, dZ, dGridId);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned> x(totalBlocks), y(totalBlocks), z(totalBlocks);
    std::vector<std::uint64_t> gridIds(totalBlocks);
    CUDA_CHECK(cudaMemcpy(x.data(), dX, totalBlocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(), dY, totalBlocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(z.data(), dZ, totalBlocks * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gridIds.data(), dGridId, totalBlocks * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));

    const std::uint64_t launchGridId = gridIds[0];
    bool ok = true;
    for (unsigned pz = 0; pz < grid.z && ok; ++pz) {
        for (unsigned py = 0; py < grid.y && ok; ++py) {
            std::vector<unsigned> seenX(grid.x, 0u);
            for (unsigned px = 0; px < grid.x; ++px) {
                const unsigned slot = px + grid.x * (py + grid.y * pz);
                const unsigned expectedX = expectedLogical(px, grid.x, launchGridId);
                if (gridIds[slot] != launchGridId || x[slot] != expectedX ||
                    y[slot] != py || z[slot] != pz) {
                    std::fprintf(
                        stderr,
                        "FAIL: 3D grid physical=(%u,%u,%u) observed=(%u,%u,%u) expectedX=%u\n",
                        px,
                        py,
                        pz,
                        x[slot],
                        y[slot],
                        z[slot],
                        expectedX);
                    ok = false;
                    break;
                }
                ++seenX[x[slot]];
            }
            for (unsigned lx = 0; lx < grid.x && ok; ++lx) {
                if (seenX[lx] != 1u) {
                    std::fprintf(stderr, "FAIL: 3D x permutation not bijective in y/z plane\n");
                    ok = false;
                }
            }
        }
    }

    std::printf(
        "[8] %uD-style grid (%u,%u,%u): x shuffled, y/z unchanged: %s\n",
        grid.z > 1 ? 3u : 2u,
        grid.x,
        grid.y,
        grid.z,
        ok ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dZ));
    CUDA_CHECK(cudaFree(dGridId));
    return ok;
}

} // namespace

int main() {
    std::printf("============================================================\n");
    std::printf("SuperCollider block-shuffle validation\n");
    std::printf("shuffle expected : %d\n", SC_EXPECT_SHUFFLE);
    std::printf("shuffle seed     : %llu\n", static_cast<unsigned long long>(SC_TEST_SHUFFLE_SEED));
    std::printf("============================================================\n");

    CUDA_CHECK(cudaFree(nullptr));

    bool ok = true;
    ok = runMappingSweep() && ok;
    ok = runSuccessiveLaunchTest() && ok;
    ok = runConcurrentStreamTest() && ok;
    ok = runPerThreadConsistencyTest() && ok;
    ok = runIndexingTests() && ok;
    ok = runControlFlowAndSharedTest() && ok;
    ok = runPrepassDeviceHelperTest() && ok;
    ok = runThreeDimensionalTest(dim3(7, 3, 1)) && ok;
    ok = runThreeDimensionalTest(dim3(5, 3, 2)) && ok;

    std::printf("============================================================\n");
    std::printf("FINAL: %s\n", ok ? "PASS" : "FAIL");
    std::printf("============================================================\n");

    return ok ? 0 : 1;
}
