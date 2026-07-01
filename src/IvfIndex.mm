// SPDX-License-Identifier: Apache-2.0
// MetalFlat — IvfIndex implementation.
//
// Build: k-means clusters the database into nlist cells, then every
// vector is assigned to its cell and the database is reordered into a
// CSR layout (cellStart[] + reordered vectors + reordered ids) so each
// cell is a contiguous block.
//
// Search (two stages):
//   coarse — for each query, find its nprobe nearest centroids (cells).
//            Reuses FlatIndex over the centroids (a tiny exact search).
//   fine   — scan only those cells' contiguous blocks, keep the top-k.
//            GPU: the ivf_scan kernel (one thread per query). CPU
//            fallback: searchCpu (also the recall reference).
// Exact within the probed cells, approximate overall (recall < 1,
// tunable via nprobe — higher nprobe = more cells scanned = more recall,
// less speed). This is the path past the exact-flat compute floor.
//
// k-means + CSR build are CPU (one-time). The per-query fine scan — the
// hot path — runs on the GPU.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "metalflat/IvfIndex.h"
#include "metalflat/FlatIndex.h"
#include "Internal.h"
#include "Log_internal.h"

namespace mflat {

using namespace detail;   // parallelFor, normalizeRows, sqL2, kmeansGpu, ...

namespace {

// CPU-side mirror of the kernel's `IvfParams`, field-for-field.
struct IvfParams {
    uint32_t dim;
    uint32_t k;
    uint32_t nprobe;
    uint32_t metric;       // 0 = L2, 1 = InnerProduct, 2 = Cosine
    uint32_t queryCount;
};

NSString* const kShaderSrc = @R"(
#include <metal_stdlib>
using namespace metal;

struct IvfParams { uint dim; uint k; uint nprobe; uint metric; uint queryCount; };

constant uint kMaxK = 64;

// One THREADGROUP per query (cooperative). The tg's threads split the
// query's probed-cell vectors round-robin, each keeping a register top-k;
// then a threadgroup reduction merges the per-thread top-k lists into the
// final top-k. The query is staged in threadgroup memory once. This
// replaces the old one-thread-per-query kernel (which left the GPU ~99%
// idle and serialized every cell). Cell membership ids come back via
// reorderedIds, so output ids are the caller's original indices. L2 ranks
// by -dist², so one path serves all metrics; cosine arrives as dot
// products over already-normalized vectors. Threadgroup buffers: qsh[dim],
// redScore[tgSize*k], redId[tgSize*k] (sizes set by the host).
kernel void ivf_scan(
    device const half*  reorderedDb  [[buffer(0)]],   // fp16 storage (half BW)
    device const int*   reorderedIds [[buffer(1)]],
    device const int*   cellStart    [[buffer(2)]],
    device const float* queries      [[buffer(3)]],
    device const int*   probedCells  [[buffer(4)]],  // queryCount × nprobe
    device int*         outIds       [[buffer(5)]],
    device float*       outVal       [[buffer(6)]],
    constant IvfParams& p            [[buffer(7)]],
    threadgroup float*  qsh          [[threadgroup(0)]],   // dim
    threadgroup float*  redScore     [[threadgroup(1)]],   // tgSize × k
    threadgroup int*    redId        [[threadgroup(2)]],   // tgSize × k
    uint                qi           [[threadgroup_position_in_grid]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                tgs          [[threads_per_threadgroup]])
{
    if (qi >= p.queryCount) return;
    const uint dim  = p.dim;
    const uint k    = min(p.k, kMaxK);
    const bool isL2 = (p.metric == 0u);

    // Stage the query in threadgroup memory once (read by every thread).
    device const float* qg = queries + (uint64_t)qi * dim;
    for (uint c = tid; c < dim; c += tgs) qsh[c] = qg[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bestScore[kMaxK];
    int   bestId[kMaxK];
    for (uint i = 0; i < k; ++i) { bestScore[i] = -INFINITY; bestId[i] = -1; }

    // Each thread scans a strided subset of every probed cell's block.
    device const int* myCells = probedCells + (uint64_t)qi * p.nprobe;
    for (uint pp = 0; pp < p.nprobe; ++pp) {
        const int cell = myCells[pp];
        if (cell < 0) continue;
        const int lo = cellStart[cell];
        const int hi = cellStart[cell + 1];
        for (int j = lo + (int)tid; j < hi; j += (int)tgs) {
            device const half* d = reorderedDb + (uint64_t)j * dim;
            float score;
            if ((dim & 3u) == 0u) {
                // 64-bit half4 loads (dim%4==0 => each row is 8-byte aligned);
                // db is fp16, query/accumulation stay fp32.
                const uint c4 = dim >> 2;
                threadgroup const float4* q4 = (threadgroup const float4*)qsh;
                device const half4*       d4 = (device const half4*)d;
                float4 acc = float4(0.0);
                if (isL2) { for (uint c = 0; c < c4; ++c) { float4 e = q4[c] - float4(d4[c]); acc += e * e; } }
                else      { for (uint c = 0; c < c4; ++c) acc += q4[c] * float4(d4[c]); }
                float s = acc.x + acc.y + acc.z + acc.w;
                score = isL2 ? -s : s;
            } else if (isL2) {
                float acc = 0.0;
                for (uint c = 0; c < dim; ++c) { float e = qsh[c] - float(d[c]); acc += e * e; }
                score = -acc;
            } else {
                float acc = 0.0;
                for (uint c = 0; c < dim; ++c) acc += qsh[c] * float(d[c]);
                score = acc;
            }
            if (score > bestScore[0]) {
                uint pos = 0;
                while (pos + 1u < k && score > bestScore[pos + 1u]) {
                    bestScore[pos] = bestScore[pos + 1u];
                    bestId[pos]    = bestId[pos + 1u];
                    ++pos;
                }
                bestScore[pos] = score;
                bestId[pos]    = reorderedIds[j];
            }
        }
    }

    // Publish this thread's top-k (ascending) and tree-merge across the
    // threadgroup: log2(tgs) steps, each merging two ascending top-k lists and
    // keeping the k largest. Candidate sets are disjoint, so no dedup. Requires
    // a power-of-two threadgroup size (the host guarantees it).
    for (uint i = 0; i < k; ++i) {
        redScore[tid * k + i] = bestScore[i];
        redId[tid * k + i]    = bestId[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint off = tgs >> 1; off > 0u; off >>= 1) {
        if (tid < off) {
            threadgroup float* aS = redScore + tid * k;
            threadgroup int*   aI = redId    + tid * k;
            threadgroup float* bS = redScore + (tid + off) * k;
            threadgroup int*   bI = redId    + (tid + off) * k;
            float mS[kMaxK];
            int   mI[kMaxK];
            int ia = (int)k - 1, ib = (int)k - 1;
            for (int o = (int)k - 1; o >= 0; --o) {     // fill from the largest
                const float av = (ia >= 0) ? aS[ia] : -INFINITY;
                const float bv = (ib >= 0) ? bS[ib] : -INFINITY;
                if (av >= bv) { mS[o] = av; mI[o] = (ia >= 0) ? aI[ia] : -1; --ia; }
                else          { mS[o] = bv; mI[o] = (ib >= 0) ? bI[ib] : -1; --ib; }
            }
            for (uint i = 0; i < k; ++i) { aS[i] = mS[i]; aI[i] = mI[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        device int*   oi = outIds + (uint64_t)qi * k;
        device float* ov = outVal + (uint64_t)qi * k;
        for (uint i = 0; i < k; ++i) {
            uint src = k - 1u - i;   // redScore[0..k) ascending -> output descending
            oi[i] = redId[src];
            ov[i] = isL2 ? -redScore[src] : redScore[src];
        }
    }
}
)";

// Shared helpers (normalizeRows, sqL2, parallelFor, accumulateCentroids,
// kmeansGpu, acquireMetalDevice) now live in Internal.h (mflat::detail).

// Scalar CPU k-means — the no-Metal fallback (IvfIndex-specific; the GPU path
// uses detail::kmeansGpu).
void kmeans(const float* data, int n, int dim, int nlist, int iters,
            std::vector<float>& centroids) {
    centroids.assign(static_cast<size_t>(nlist) * dim, 0.0f);
    std::mt19937 rng(12345);
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int c = 0; c < nlist; ++c)
        std::copy_n(data + static_cast<size_t>(perm[c]) * dim, dim,
                    centroids.begin() + static_cast<size_t>(c) * dim);

    std::vector<int> assign(n, 0);
    for (int it = 0; it < iters; ++it) {
        parallelFor(n, [&](int i) {
            const float* v = data + static_cast<size_t>(i) * dim;
            float best = std::numeric_limits<float>::infinity();
            int   bestC = 0;
            for (int c = 0; c < nlist; ++c) {
                float dd = sqL2(v, &centroids[static_cast<size_t>(c) * dim], dim);
                if (dd < best) { best = dd; bestC = c; }
            }
            assign[i] = bestC;
        });
        std::vector<double> sums(static_cast<size_t>(nlist) * dim, 0.0);
        std::vector<int>    counts(nlist, 0);
        for (int i = 0; i < n; ++i) {
            const float* v = data + static_cast<size_t>(i) * dim;
            const int    c = assign[i];
            double* s = &sums[static_cast<size_t>(c) * dim];
            for (int d = 0; d < dim; ++d) s[d] += v[d];
            ++counts[c];
        }
        for (int c = 0; c < nlist; ++c) {
            float* ce = &centroids[static_cast<size_t>(c) * dim];
            if (counts[c] > 0) {
                const double* s = &sums[static_cast<size_t>(c) * dim];
                for (int d = 0; d < dim; ++d)
                    ce[d] = static_cast<float>(s[d] / counts[c]);
            } else {
                const int r = static_cast<int>(rng() % static_cast<unsigned>(n));
                std::copy_n(data + static_cast<size_t>(r) * dim, dim, ce);
            }
        }
    }
}

}  // namespace

struct IvfIndex::Impl {
    int    dim     = 0;
    Metric metric  = Metric::L2;
    int    nlist   = 0;
    int    dbCount = 0;
    bool   ready   = false;

    // CPU-authoritative training output + CSR inverted lists.
    std::vector<float> centroids;     // nlist × dim
    std::vector<int>   cellStart;     // nlist + 1
    std::vector<float> reorderedDb;   // dbCount × dim
    std::vector<int>   reorderedIds;  // dbCount

    // GPU path.
    id<MTLDevice>               device  = nil;
    id<MTLCommandQueue>         queue   = nil;
    id<MTLComputePipelineState> scanPipe = nil;
    std::unique_ptr<FlatIndex>  coarse;          // exact search over centroids
    id<MTLBuffer>               dbBuf    = nil;   // reorderedDb as fp16 (half)
    id<MTLBuffer>               idBuf    = nil;   // reorderedIds
    id<MTLBuffer>               cellBuf  = nil;   // cellStart
    bool                        gpuReady = false;
};

IvfIndex::IvfIndex(int dim, Metric metric, int nlist)
    : mImpl(std::make_unique<Impl>()) {
    mImpl->dim    = dim;
    mImpl->metric = metric;
    mImpl->nlist  = nlist;

    mImpl->device = acquireMetalDevice();
    if (!mImpl->device) return;   // CPU-only path stays available
    mImpl->queue = [mImpl->device newCommandQueue];

    NSError* err = nil;
    id<MTLLibrary> lib = [mImpl->device newLibraryWithSource:kShaderSrc
                                                     options:nil
                                                       error:&err];
    if (!lib) {
        MFLAT_LOG_ERROR("ivf shader compile failed: %s",
                        err ? [[err localizedDescription] UTF8String] : "?");
        return;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"ivf_scan"];
    mImpl->scanPipe = [mImpl->device newComputePipelineStateWithFunction:fn
                                                                   error:&err];
    if (!mImpl->scanPipe) {
        MFLAT_LOG_ERROR("ivf pipeline build failed: %s",
                        err ? [[err localizedDescription] UTF8String] : "?");
    }
}

IvfIndex::~IvfIndex() = default;

int  IvfIndex::dim()   const { return mImpl->dim; }
int  IvfIndex::size()  const { return mImpl->dbCount; }
int  IvfIndex::nlist() const { return mImpl->nlist; }
bool IvfIndex::ready() const { return mImpl->ready; }

void IvfIndex::build(const float* vectors, int n) {
    if (n <= 0 || mImpl->dim <= 0) return;
    const int dim = mImpl->dim;

    std::vector<float> data(vectors, vectors + static_cast<size_t>(n) * dim);
    if (mImpl->metric == Metric::Cosine) normalizeRows(data, n, dim);

    const int nlist = std::min(mImpl->nlist, n);
    mImpl->nlist = nlist;

    constexpr int kIters = 12;
    std::vector<int> assign;
    if (mImpl->scanPipe) {
        // GPU k-means: the assignment step runs on the MPS GEMM path via a
        // reused FlatIndex; assign[] comes back consistent with the final
        // centroids, so no separate CPU assignment pass is needed.
        kmeansGpu(data.data(), n, dim, nlist, kIters, mImpl->centroids, assign);
    } else {
        // No Metal device: scalar CPU k-means + one assignment pass.
        kmeans(data.data(), n, dim, nlist, kIters, mImpl->centroids);
        assign.assign(n, 0);
        parallelFor(n, [&](int i) {
            const float* v = &data[static_cast<size_t>(i) * dim];
            float best = std::numeric_limits<float>::infinity();
            int   bestC = 0;
            for (int c = 0; c < nlist; ++c) {
                float dd = sqL2(v, &mImpl->centroids[static_cast<size_t>(c) * dim], dim);
                if (dd < best) { best = dd; bestC = c; }
            }
            assign[i] = bestC;
        });
    }

    mImpl->cellStart.assign(nlist + 1, 0);
    for (int i = 0; i < n; ++i) ++mImpl->cellStart[assign[i] + 1];
    for (int c = 0; c < nlist; ++c)
        mImpl->cellStart[c + 1] += mImpl->cellStart[c];

    mImpl->reorderedDb.assign(static_cast<size_t>(n) * dim, 0.0f);
    mImpl->reorderedIds.assign(n, 0);
    std::vector<int> cursor(mImpl->cellStart.begin(), mImpl->cellStart.end());
    for (int i = 0; i < n; ++i) {
        const int c   = assign[i];
        const int pos = cursor[c]++;
        std::copy_n(&data[static_cast<size_t>(i) * dim], dim,
                    &mImpl->reorderedDb[static_cast<size_t>(pos) * dim]);
        mImpl->reorderedIds[pos] = i;
    }

    mImpl->dbCount = n;
    mImpl->ready   = true;

    // --- GPU path setup: coarse quantizer + upload CSR ---------------
    if (mImpl->scanPipe) {
        mImpl->coarse = std::make_unique<FlatIndex>(dim, Metric::L2);
        mImpl->coarse->add(mImpl->centroids.data(), nlist);

        // Upload the database as fp16 (halves the scan's memory traffic; the
        // CPU keeps the fp32 copy for the reference path). Lossless for inputs
        // exactly representable in half (e.g. SIFT's small integers).
        std::vector<__fp16> dbHalf(mImpl->reorderedDb.size());
        for (size_t i = 0; i < dbHalf.size(); ++i)
            dbHalf[i] = static_cast<__fp16>(mImpl->reorderedDb[i]);
        mImpl->dbBuf = [mImpl->device
            newBufferWithBytes:dbHalf.data()
                        length:dbHalf.size() * sizeof(__fp16)
                       options:MTLResourceStorageModeShared];
        mImpl->idBuf = [mImpl->device
            newBufferWithBytes:mImpl->reorderedIds.data()
                        length:mImpl->reorderedIds.size() * sizeof(int32_t)
                       options:MTLResourceStorageModeShared];
        mImpl->cellBuf = [mImpl->device
            newBufferWithBytes:mImpl->cellStart.data()
                        length:mImpl->cellStart.size() * sizeof(int32_t)
                       options:MTLResourceStorageModeShared];
        mImpl->gpuReady = mImpl->coarse->ready();
    }
}

namespace {

// CPU two-stage search — the fallback (and the numerically-clean
// reference for the GPU path). Takes raw fields, not Impl, so it stays a
// free function without reaching into IvfIndex's private nested type.
void searchCpu(const std::vector<float>& centroids,
               const std::vector<int>&   cellStart,
               const std::vector<float>& reorderedDb,
               const std::vector<int>&   reorderedIds,
               int dim, int nlist, Metric metric,
               const float* qPtr, int m, int k, int nprobe,
               SearchResult& out) {
    const bool isL2 = (metric == Metric::L2);

    parallelFor(m, [&](int qi) {
        const float* q = qPtr + static_cast<size_t>(qi) * dim;
        std::vector<std::pair<float, int>> cd(nlist);
        for (int c = 0; c < nlist; ++c)
            cd[c] = {sqL2(q, &centroids[static_cast<size_t>(c) * dim], dim), c};
        std::partial_sort(cd.begin(), cd.begin() + nprobe, cd.end());

        std::vector<float> bestScore(k, -std::numeric_limits<float>::infinity());
        std::vector<int>   bestId(k, -1);
        for (int pp = 0; pp < nprobe; ++pp) {
            const int c  = cd[pp].second;
            const int lo = cellStart[c];
            const int hi = cellStart[c + 1];
            for (int j = lo; j < hi; ++j) {
                const float* d = &reorderedDb[static_cast<size_t>(j) * dim];
                float score;
                if (isL2) {
                    score = -sqL2(q, d, dim);
                } else {
                    float acc = 0.0f;
                    for (int cc = 0; cc < dim; ++cc) acc += q[cc] * d[cc];
                    score = acc;
                }
                if (score > bestScore[0]) {
                    int pos = 0;
                    while (pos + 1 < k && score > bestScore[pos + 1]) {
                        bestScore[pos] = bestScore[pos + 1];
                        bestId[pos]    = bestId[pos + 1];
                        ++pos;
                    }
                    bestScore[pos] = score;
                    bestId[pos]    = reorderedIds[j];
                }
            }
        }
        for (int i = 0; i < k; ++i) {
            const int src = k - 1 - i;
            const size_t o = static_cast<size_t>(qi) * k + i;
            out.ids[o]       = bestId[src];
            out.distances[o] = isL2 ? -bestScore[src] : bestScore[src];
        }
    });
}

}  // namespace

SearchResult IvfIndex::search(const float* queries, int m, int k, int nprobe) {
    SearchResult out;
    if (m <= 0 || !mImpl->ready) return out;
    const int dim   = mImpl->dim;
    const int nlist = mImpl->nlist;
    if (k < 1) k = 1;
    // k > kMaxK is served by the exact CPU path below (the GPU top-k uses
    // fixed kMaxK-sized per-thread buffers); not clamped, so results stay correct.
    if (nprobe < 1) nprobe = 1;
    if (nprobe > nlist) nprobe = nlist;

    const bool isL2 = (mImpl->metric == Metric::L2);
    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k,
                         isL2 ? std::numeric_limits<float>::infinity()
                              : -std::numeric_limits<float>::infinity());

    // Cosine: normalize queries once; used by both stages (the coarse
    // quantizer's centroids and the cells are built from normalized data).
    std::vector<float> qNorm;
    const float* qPtr = queries;
    if (mImpl->metric == Metric::Cosine) {
        qNorm.assign(queries, queries + static_cast<size_t>(m) * dim);
        normalizeRows(qNorm, m, dim);
        qPtr = qNorm.data();
    }

    // The GPU fine scan runs for ANY nprobe. Only the coarse cell-selection is
    // bounded by FlatIndex::kMaxK (its GPU top-k), so: for nprobe<=kMaxK use the
    // GPU coarse quantizer; for larger nprobe select the nprobe nearest
    // centroids on the CPU (partial_sort over the small nlist) and feed the same
    // nprobe-agnostic ivf_scan kernel. No more CPU-only cliff (and recall cap)
    // at nprobe=64.
    const bool gpuPath = mImpl->gpuReady && mImpl->coarse && k <= kMaxK;
    if (!gpuPath) {
        searchCpu(mImpl->centroids, mImpl->cellStart, mImpl->reorderedDb,
                  mImpl->reorderedIds, dim, nlist, mImpl->metric,
                  qPtr, m, k, nprobe, out);
        return out;
    }

    // Build the m × nprobe probed-cell list (cell indices), GPU or CPU coarse.
    SearchResult coarse;             // owns storage for the GPU coarse path
    std::vector<int32_t> probedCpu;  // owns storage for the CPU coarse path
    const int32_t* probedPtr = nullptr;
    if (nprobe <= FlatIndex::kMaxK) {
        coarse = mImpl->coarse->search(qPtr, m, nprobe);  // exact, on GPU
        probedPtr = coarse.ids.data();
    } else {
        probedCpu.assign(static_cast<size_t>(m) * nprobe, -1);
        const std::vector<float>& cents = mImpl->centroids;
        parallelFor(m, [&](int qi) {
            const float* q = qPtr + static_cast<size_t>(qi) * dim;
            std::vector<std::pair<float, int>> cd(nlist);
            for (int c = 0; c < nlist; ++c)
                cd[c] = {sqL2(q, &cents[static_cast<size_t>(c) * dim], dim), c};
            std::partial_sort(cd.begin(), cd.begin() + nprobe, cd.end());
            int32_t* row = &probedCpu[static_cast<size_t>(qi) * nprobe];
            for (int p = 0; p < nprobe; ++p) row[p] = cd[p].second;
        });
        probedPtr = probedCpu.data();
    }

    @autoreleasepool {
        id<MTLBuffer> qBuf = [mImpl->device
            newBufferWithBytes:qPtr
                        length:static_cast<size_t>(m) * dim * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> probedBuf = [mImpl->device
            newBufferWithBytes:probedPtr
                        length:static_cast<size_t>(m) * nprobe * sizeof(int32_t)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> outIdBuf = [mImpl->device
            newBufferWithLength:static_cast<size_t>(m) * k * sizeof(int32_t)
                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> outValBuf = [mImpl->device
            newBufferWithLength:static_cast<size_t>(m) * k * sizeof(float)
                        options:MTLResourceStorageModeShared];

        IvfParams p;
        p.dim        = static_cast<uint32_t>(dim);
        p.k          = static_cast<uint32_t>(k);
        p.nprobe     = static_cast<uint32_t>(nprobe);
        p.metric     = static_cast<uint32_t>(mImpl->metric);
        p.queryCount = static_cast<uint32_t>(m);

        id<MTLCommandBuffer>         cb  = [mImpl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mImpl->scanPipe];
        [enc setBuffer:mImpl->dbBuf   offset:0 atIndex:0];
        [enc setBuffer:mImpl->idBuf   offset:0 atIndex:1];
        [enc setBuffer:mImpl->cellBuf offset:0 atIndex:2];
        [enc setBuffer:qBuf           offset:0 atIndex:3];
        [enc setBuffer:probedBuf      offset:0 atIndex:4];
        [enc setBuffer:outIdBuf       offset:0 atIndex:5];
        [enc setBuffer:outValBuf      offset:0 atIndex:6];
        [enc setBytes:&p length:sizeof(p) atIndex:7];
        // One threadgroup per query. Pick the largest threadgroup whose
        // staged query (dim floats) + reduction scratch (tg × k × (float+int))
        // fits the threadgroup-memory budget; round to a SIMD-width multiple.
        const NSUInteger kk  = static_cast<NSUInteger>(k);
        const NSUInteger qsh = static_cast<NSUInteger>(dim) * sizeof(float);
        NSUInteger memTg = (qsh < 32000)
            ? (32000 - qsh) / (kk * (sizeof(float) + sizeof(int)))
            : 1;
        NSUInteger tg = std::min<NSUInteger>(
            mImpl->scanPipe.maxTotalThreadsPerThreadgroup, 256);
        tg = std::min<NSUInteger>(tg, std::max<NSUInteger>(memTg, 32));
        // Floor to a power of two — the tree-merge reduction requires it.
        NSUInteger pw = 32;
        while (pw * 2 <= tg) pw *= 2;
        tg = pw;

        [enc setThreadgroupMemoryLength:qsh atIndex:0];
        [enc setThreadgroupMemoryLength:tg * kk * sizeof(float) atIndex:1];
        [enc setThreadgroupMemoryLength:tg * kk * sizeof(int)   atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(m), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out.ids.data(), [outIdBuf contents],
                    out.ids.size() * sizeof(int32_t));
        std::memcpy(out.distances.data(), [outValBuf contents],
                    out.distances.size() * sizeof(float));
    }
    return out;
}

}  // namespace mflat
