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
#include <cstdlib>
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
#include "CoarseQuantizer.h"

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
// CPU-side mirrors of the tiled kernels' params, field-for-field.
struct TileParams  { uint32_t dim, k, nprobe, metric, Tq, Cv, workCount; };
struct MergeParams { uint32_t k, nprobe, metric, queryCount; };

// Tiled-scan tile shape: Tq queries share each staged db row; Cv rows staged
// per chunk. Bigger tiles = more work per barrier (Cv*Tq pairs/chunk) and more
// row reuse, bounded by the ~32KB threadgroup budget (validated at dispatch).
// MFLAT_IVF_TQ / MFLAT_IVF_CV override for tuning.
inline uint32_t tileTq() {
    const char* e = std::getenv("MFLAT_IVF_TQ");
    const int x = e ? atoi(e) : 16;
    return static_cast<uint32_t>(std::min(std::max(x, 1), 64));
}
inline uint32_t tileCv() {
    const char* e = std::getenv("MFLAT_IVF_CV");
    const int x = e ? atoi(e) : 32;   // measured best on M2 (bigger kills occupancy)
    return static_cast<uint32_t>(std::min(std::max(x, 8), 256));
}

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

struct TileParams  { uint dim; uint k; uint nprobe; uint metric; uint Tq; uint Cv; uint workCount; };
struct MergeParams { uint k; uint nprobe; uint metric; uint queryCount; };

// Phase 1 of the query-tiled fine scan (large batches): one threadgroup per
// (cell, tile of <=Tq queries probing it). Each fp16 db row is staged into
// threadgroup memory ONCE and scored against the whole query tile, cutting
// device reads ~Tq× vs ivf_scan (which re-reads every row per query). Thread
// tid < qCount OWNS query slot tid: it drains its score-slab column into an
// ascending top-k and writes the (query,probe) partial slot — exactly one
// writer per slot, so no pre-clearing. Distance math is verbatim ivf_scan
// (half4 loads, float4 accumulate) so scores match the legacy kernel bitwise.
kernel void ivf_scan_tiled(
    device const half*  reorderedDb  [[buffer(0)]],
    device const int*   reorderedIds [[buffer(1)]],
    device const int*   cellStart    [[buffer(2)]],
    device const float* queries      [[buffer(3)]],
    device const uint*  entries      [[buffer(4)]],   // qi*nprobe+pp, grouped by cell
    device const int*   qStart       [[buffer(5)]],   // nlist+1 CSR over entries
    device const int*   workCell     [[buffer(6)]],   // work item -> cell id
    device const int*   workOff      [[buffer(7)]],   // work item -> entries offset
    device float*       pScore       [[buffer(8)]],   // (m*nprobe) × k partials, ascending
    device int*         pId          [[buffer(9)]],
    constant TileParams& p           [[buffer(10)]],
    threadgroup float*  qsh          [[threadgroup(0)]],   // Tq × dim
    threadgroup half*   dsh          [[threadgroup(1)]],   // Cv × dim
    threadgroup float*  ssh          [[threadgroup(2)]],   // Cv × Tq
    uint wg  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    if (wg >= p.workCount) return;
    const uint dim = p.dim, k = min(p.k, kMaxK), Tq = p.Tq, Cv = p.Cv;
    const bool isL2 = (p.metric == 0u);
    const int  cell = workCell[wg];
    const int  eOff = workOff[wg];
    const uint qCount = min(Tq, (uint)(qStart[cell + 1] - eOff));
    const int  lo = cellStart[cell];
    const int  hi = cellStart[cell + 1];   // hi==lo (empty cell): loop skips,
                                           // owners still write -INF sentinels
    // Stage the query tile.
    for (uint idx = tid; idx < qCount * dim; idx += tgs) {
        const uint qq = idx / dim, c = idx % dim;
        const uint qi = entries[eOff + qq] / p.nprobe;
        qsh[qq * dim + c] = queries[(uint64_t)qi * dim + c];
    }
    float bestScore[kMaxK];
    int   bestId[kMaxK];
    float worst = -INFINITY;
    if (tid < qCount)
        for (uint i = 0; i < k; ++i) { bestScore[i] = -INFINITY; bestId[i] = -1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c4 = dim >> 2;   // host gates this path on dim % 4 == 0
    for (int base = lo; base < hi; base += (int)Cv) {
        const uint nrows = min(Cv, (uint)(hi - base));
        // Stage Cv fp16 rows (half4, coalesced).
        for (uint idx = tid; idx < nrows * c4; idx += tgs) {
            const uint r = idx / c4, cc = idx - r * c4;
            ((threadgroup half4*)dsh)[r * c4 + cc] =
                ((device const half4*)(reorderedDb + (uint64_t)(base + (int)r) * dim))[cc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Score slab: (row, query) pairs round-robin across all threads.
        for (uint pj = tid; pj < nrows * Tq; pj += tgs) {
            const uint r = pj / Tq, q = pj - r * Tq;   // Tq is a power of two
            if (q < qCount) {
                threadgroup const float4* q4 = (threadgroup const float4*)(qsh + q * dim);
                threadgroup const half4*  d4 = (threadgroup const half4*)(dsh + r * dim);
                float4 acc = float4(0.0);
                if (isL2) { for (uint c = 0; c < c4; ++c) { float4 e = q4[c] - float4(d4[c]); acc += e * e; } }
                else      { for (uint c = 0; c < c4; ++c) acc += q4[c] * float4(d4[c]); }
                const float s = acc.x + acc.y + acc.z + acc.w;
                ssh[r * Tq + q] = isL2 ? -s : s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Selection: each owner drains its column (threshold prefilter keeps
        // the hot compare on a scalar; inserts are rare after warmup).
        if (tid < qCount) {
            for (uint r = 0; r < nrows; ++r) {
                const float s = ssh[r * Tq + tid];
                if (s > worst) {
                    uint pos = 0;
                    while (pos + 1u < k && s > bestScore[pos + 1u]) {
                        bestScore[pos] = bestScore[pos + 1u];
                        bestId[pos]    = bestId[pos + 1u];
                        ++pos;
                    }
                    bestScore[pos] = s;
                    bestId[pos]    = reorderedIds[base + (int)r];
                    worst = bestScore[0];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // before restaging dsh/ssh
    }

    if (tid < qCount) {
        const uint e = entries[eOff + tid];
        device float* oS = pScore + (uint64_t)e * k;
        device int*   oI = pId    + (uint64_t)e * k;
        for (uint i = 0; i < k; ++i) { oS[i] = bestScore[i]; oI[i] = bestId[i]; }
    }
}

// Phase 2: one threadgroup per query merges its nprobe ascending k-lists into
// the final descending top-k (per-thread top-k + tree merge, as ivf_scan).
// Partial lists have disjoint ids (a db row lives in exactly one cell).
kernel void ivf_partial_merge(
    device const float* pScore [[buffer(0)]],
    device const int*   pId    [[buffer(1)]],
    device int*         outIds [[buffer(2)]],
    device float*       outVal [[buffer(3)]],
    constant MergeParams& p    [[buffer(4)]],
    threadgroup float*  redScore [[threadgroup(0)]],   // tgs × k
    threadgroup int*    redId    [[threadgroup(1)]],
    uint qi  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    if (qi >= p.queryCount) return;
    const uint k = min(p.k, kMaxK);
    const bool isL2 = (p.metric == 0u);
    const uint total = p.nprobe * k;
    device const float* qS = pScore + (uint64_t)qi * total;
    device const int*   qI = pId    + (uint64_t)qi * total;

    float bestScore[kMaxK];
    int   bestId[kMaxK];
    for (uint i = 0; i < k; ++i) { bestScore[i] = -INFINITY; bestId[i] = -1; }
    for (uint e = tid; e < total; e += tgs) {
        const float s  = qS[e];
        const int   id = qI[e];
        if (id >= 0 && s > bestScore[0]) {
            uint pos = 0;
            while (pos + 1u < k && s > bestScore[pos + 1u]) {
                bestScore[pos] = bestScore[pos + 1u];
                bestId[pos]    = bestId[pos + 1u];
                ++pos;
            }
            bestScore[pos] = s;
            bestId[pos]    = id;
        }
    }
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
            for (int o = (int)k - 1; o >= 0; --o) {
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
            uint src = k - 1u - i;
            oi[i] = redId[src];
            ov[i] = isL2 ? -redScore[src] : redScore[src];
        }
    }
}
)";

// Shared helpers (normalizeRows, sqL2, parallelFor, accumulateCentroids,
// kmeansGpu, acquireMetalDevice) now live in Internal.h (mflat::detail).

// Scalar CPU k-means (no-Metal fallback) + all coarse-layer logic now live in
// CoarseQuantizer (src/CoarseQuantizer.{h,mm}).

}  // namespace

struct IvfIndex::Impl {
    int    dim     = 0;
    Metric metric  = Metric::L2;
    int    nlist   = 0;
    int    dbCount = 0;
    bool   ready   = false;

    // Coarse layer: centroids, CSR, coarse FlatIndex, cell/id GPU buffers.
    std::unique_ptr<CoarseQuantizer> cq;
    std::vector<float> reorderedDb;   // dbCount × dim (float payload; CPU reference)

    // GPU fine-scan path.
    id<MTLDevice>               device    = nil;
    id<MTLCommandQueue>         queue     = nil;
    id<MTLComputePipelineState> scanPipe  = nil;
    id<MTLComputePipelineState> tiledPipe = nil;   // query-tiled phase 1
    id<MTLComputePipelineState> mergePipe = nil;   // query-tiled phase 2
    id<MTLBuffer>               dbBuf     = nil;   // reorderedDb as fp16 (half)

    // Query-tiled fine scan (see ivf_scan_tiled): invert probed to cell->query
    // lists on the CPU, run phase 1 (partial top-k per (query,probe)) + phase 2
    // (per-query merge) in one command buffer. Caller guarantees gpu-ready,
    // k <= kMaxK, dim % 4 == 0.
    void runTiledGpu(const float* qPtr, const std::vector<int32_t>& probed,
                     int m, int k, int nprobe, SearchResult& out);
};

IvfIndex::IvfIndex(int dim, Metric metric, int nlist)
    : mImpl(std::make_unique<Impl>()) {
    mImpl->dim    = dim;
    mImpl->metric = metric;
    mImpl->nlist  = nlist;

    mImpl->device = acquireMetalDevice();
    if (mImpl->device) {
        mImpl->queue = [mImpl->device newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [mImpl->device newLibraryWithSource:kShaderSrc
                                                         options:nil
                                                           error:&err];
        if (!lib) {
            MFLAT_LOG_ERROR("ivf shader compile failed: %s",
                            err ? [[err localizedDescription] UTF8String] : "?");
        } else {
            id<MTLFunction> fn = [lib newFunctionWithName:@"ivf_scan"];
            mImpl->scanPipe = [mImpl->device newComputePipelineStateWithFunction:fn
                                                                           error:&err];
            if (!mImpl->scanPipe)
                MFLAT_LOG_ERROR("ivf pipeline build failed: %s",
                                err ? [[err localizedDescription] UTF8String] : "?");
            // Tiled-scan pipelines (optional — legacy per-query kernel remains
            // the fallback if either fails).
            id<MTLFunction> ft = [lib newFunctionWithName:@"ivf_scan_tiled"];
            id<MTLFunction> fm = [lib newFunctionWithName:@"ivf_partial_merge"];
            if (ft) mImpl->tiledPipe = [mImpl->device newComputePipelineStateWithFunction:ft error:&err];
            if (fm) mImpl->mergePipe = [mImpl->device newComputePipelineStateWithFunction:fm error:&err];
            if (!mImpl->tiledPipe || !mImpl->mergePipe)
                MFLAT_LOG_WARN("ivf tiled pipelines unavailable: %s",
                               err ? [[err localizedDescription] UTF8String] : "?");
        }
    }
    // The coarse quantizer uses the device only when the fine-scan pipeline is
    // usable (preserves the old scanPipe-gated GPU behavior); nil => CPU coarse.
    mImpl->cq = std::make_unique<CoarseQuantizer>(mImpl->scanPipe ? mImpl->device : nil, dim);
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

    // Coarse quantizer: k-means + CSR + coarse FlatIndex + cell/id GPU buffers.
    mImpl->cq->train(data.data(), n, mImpl->nlist, 12,
                     CoarseQuantizer::KmeansBackend::Auto);
    mImpl->nlist   = mImpl->cq->nlist();
    mImpl->dbCount = n;

    // Reorder the float payload into CSR slot order; then (GPU) cast to fp16 for
    // the scan buffer. reorder-before-cast keeps the fp16 db lossless-identical.
    mImpl->reorderedDb.assign(static_cast<size_t>(n) * dim, 0.0f);
    mImpl->cq->reorderPayload(data.data(), mImpl->reorderedDb.data(),
                              static_cast<size_t>(dim) * sizeof(float));
    mImpl->ready = true;

    if (mImpl->scanPipe && mImpl->cq->gpuReady()) {
        std::vector<__fp16> dbHalf(mImpl->reorderedDb.size());
        for (size_t i = 0; i < dbHalf.size(); ++i)
            dbHalf[i] = static_cast<__fp16>(mImpl->reorderedDb[i]);
        mImpl->dbBuf = [mImpl->device
            newBufferWithBytes:dbHalf.data()
                        length:dbHalf.size() * sizeof(__fp16)
                       options:MTLResourceStorageModeShared];
    }
}

namespace {

// CPU two-stage search — the fallback (and the numerically-clean
// reference for the GPU path). Takes raw fields, not Impl, so it stays a
// free function without reaching into IvfIndex's private nested type.
void searchCpu(const CoarseQuantizer& cq, const std::vector<float>& reorderedDb,
               int dim, Metric metric, const float* qPtr, int m, int k, int nprobe,
               SearchResult& out) {
    const std::vector<int>& cellStart    = cq.cellStart();
    const std::vector<int>& reorderedIds = cq.reorderedIds();
    parallelFor(m, [&](int qi) {
        const float* q = qPtr + static_cast<size_t>(qi) * dim;
        std::vector<int> cells(nprobe);
        cq.probeCellsCpu(q, nprobe, cells.data());

        std::vector<float> bestScore(k, -std::numeric_limits<float>::infinity());
        std::vector<int>   bestId(k, -1);
        for (int pp = 0; pp < nprobe; ++pp) {
            const int c  = cells[pp];
            const int lo = cellStart[c];
            const int hi = cellStart[c + 1];
            for (int j = lo; j < hi; ++j) {
                const float s = detail::score(metric, q,
                        &reorderedDb[static_cast<size_t>(j) * dim], dim);
                if (s > bestScore[0]) {
                    int pos = 0;
                    while (pos + 1 < k && s > bestScore[pos + 1]) {
                        bestScore[pos] = bestScore[pos + 1];
                        bestId[pos]    = bestId[pos + 1];
                        ++pos;
                    }
                    bestScore[pos] = s;
                    bestId[pos]    = reorderedIds[j];
                }
            }
        }
        for (int i = 0; i < k; ++i) {
            const int src = k - 1 - i;
            const size_t o = static_cast<size_t>(qi) * k + i;
            out.ids[o]       = bestId[src];
            out.distances[o] = detail::scoreToValue(metric, bestScore[src]);
        }
    });
}

}  // namespace

void IvfIndex::Impl::runTiledGpu(const float* qPtr, const std::vector<int32_t>& probed,
                                 int m, int k, int nprobe, SearchResult& out) {
    // --- CPU inversion: probed (m × nprobe, query-major) -> per-cell query
    // lists (CSR qStart + entries), then fixed-size Tq work items. Stable
    // qi-major fill keeps each cell's list deterministic.
    const uint32_t Tq = tileTq(), Cv = tileCv();
    const uint32_t E = static_cast<uint32_t>(static_cast<uint64_t>(m) * nprobe);
    std::vector<int32_t>  qStart(nlist + 1, 0);
    std::vector<uint32_t> entries(E);
    for (uint32_t e = 0; e < E; ++e) ++qStart[probed[e] + 1];
    for (int c = 0; c < nlist; ++c) qStart[c + 1] += qStart[c];
    {
        std::vector<int32_t> cursor(qStart.begin(), qStart.end() - 1);
        for (int qi = 0; qi < m; ++qi)
            for (int pp = 0; pp < nprobe; ++pp) {
                const int c = probed[static_cast<size_t>(qi) * nprobe + pp];
                entries[cursor[c]++] = static_cast<uint32_t>(qi) * nprobe + pp;
            }
    }
    std::vector<int32_t> workCell, workOff;
    workCell.reserve(E / Tq + nlist);
    workOff.reserve(E / Tq + nlist);
    for (int c = 0; c < nlist; ++c)
        for (int off = qStart[c]; off < qStart[c + 1]; off += (int)Tq) {
            workCell.push_back(c);
            workOff.push_back(off);
        }
    const uint32_t workCount = static_cast<uint32_t>(workCell.size());

    @autoreleasepool {
        auto bufWith = [&](const void* p, size_t bytes) {
            return [device newBufferWithBytes:p length:bytes
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> qBuf     = bufWith(qPtr, static_cast<size_t>(m) * dim * sizeof(float));
        id<MTLBuffer> entBuf   = bufWith(entries.data(),  entries.size()  * sizeof(uint32_t));
        id<MTLBuffer> qsBuf    = bufWith(qStart.data(),   qStart.size()   * sizeof(int32_t));
        id<MTLBuffer> wcBuf    = bufWith(workCell.data(), workCell.size() * sizeof(int32_t));
        id<MTLBuffer> woBuf    = bufWith(workOff.data(),  workOff.size()  * sizeof(int32_t));
        id<MTLBuffer> pScore   = [device newBufferWithLength:static_cast<size_t>(E) * k * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> pId      = [device newBufferWithLength:static_cast<size_t>(E) * k * sizeof(int32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outIdBuf = [device newBufferWithLength:static_cast<size_t>(m) * k * sizeof(int32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outValBuf = [device newBufferWithLength:static_cast<size_t>(m) * k * sizeof(float)
                                                      options:MTLResourceStorageModeShared];

        TileParams tp;
        tp.dim = (uint32_t)dim; tp.k = (uint32_t)k; tp.nprobe = (uint32_t)nprobe;
        tp.metric = (uint32_t)metric; tp.Tq = Tq; tp.Cv = Cv;
        tp.workCount = workCount;
        MergeParams mp;
        mp.k = (uint32_t)k; mp.nprobe = (uint32_t)nprobe;
        mp.metric = (uint32_t)metric; mp.queryCount = (uint32_t)m;

        id<MTLCommandBuffer>         cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // Phase 1 — partial top-k per (query, probe). Metal's hazard tracking
        // serializes the pScore/pId write->read across the two dispatches.
        [enc setComputePipelineState:tiledPipe];
        [enc setBuffer:dbBuf              offset:0 atIndex:0];
        [enc setBuffer:cq->idBuffer()     offset:0 atIndex:1];
        [enc setBuffer:cq->cellBuffer()   offset:0 atIndex:2];
        [enc setBuffer:qBuf               offset:0 atIndex:3];
        [enc setBuffer:entBuf             offset:0 atIndex:4];
        [enc setBuffer:qsBuf              offset:0 atIndex:5];
        [enc setBuffer:wcBuf              offset:0 atIndex:6];
        [enc setBuffer:woBuf              offset:0 atIndex:7];
        [enc setBuffer:pScore             offset:0 atIndex:8];
        [enc setBuffer:pId                offset:0 atIndex:9];
        [enc setBytes:&tp length:sizeof(tp) atIndex:10];
        NSUInteger tg1 = std::min<NSUInteger>(tiledPipe.maxTotalThreadsPerThreadgroup, 256);
        { NSUInteger pw = 32; while (pw * 2 <= tg1) pw *= 2; tg1 = pw; }
        [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(Tq) * dim * sizeof(float)    atIndex:0];
        [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(Cv) * dim * sizeof(uint16_t) atIndex:1];
        [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(Cv) * Tq * sizeof(float) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(workCount, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg1, 1, 1)];

        // Phase 2 — per-query merge of the nprobe partial lists.
        [enc setComputePipelineState:mergePipe];
        [enc setBuffer:pScore    offset:0 atIndex:0];
        [enc setBuffer:pId       offset:0 atIndex:1];
        [enc setBuffer:outIdBuf  offset:0 atIndex:2];
        [enc setBuffer:outValBuf offset:0 atIndex:3];
        [enc setBytes:&mp length:sizeof(mp) atIndex:4];
        const NSUInteger kk = static_cast<NSUInteger>(k);
        NSUInteger memTg = 30000 / (kk * (sizeof(float) + sizeof(int)));
        NSUInteger tg2 = std::min<NSUInteger>(mergePipe.maxTotalThreadsPerThreadgroup, 256);
        tg2 = std::min<NSUInteger>(tg2, std::max<NSUInteger>(memTg, 32));
        { NSUInteger pw = 32; while (pw * 2 <= tg2) pw *= 2; tg2 = pw; }
        [enc setThreadgroupMemoryLength:tg2 * kk * sizeof(float) atIndex:0];
        [enc setThreadgroupMemoryLength:tg2 * kk * sizeof(int)   atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(m), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out.ids.data(), [outIdBuf contents],
                    out.ids.size() * sizeof(int32_t));
        std::memcpy(out.distances.data(), [outValBuf contents],
                    out.distances.size() * sizeof(float));
    }
}

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

    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k, detail::emptyValue(mImpl->metric));

    // Cosine: normalize queries once (coarse centroids + cells were built from
    // normalized data).
    std::vector<float> qNorm;
    const float* qPtr = queries;
    if (mImpl->metric == Metric::Cosine) {
        qNorm.assign(queries, queries + static_cast<size_t>(m) * dim);
        normalizeRows(qNorm, m, dim);
        qPtr = qNorm.data();
    }

    // GPU path needs the fine-scan pipeline AND the GPU coarse quantizer; k>kMaxK
    // falls to the exact CPU path (the GPU top-k uses fixed kMaxK-sized buffers).
    // MFLAT_IVF_CPU forces the CPU path regardless (benchmarking / hybrid split).
    const bool forceCpu = std::getenv("MFLAT_IVF_CPU") != nullptr;
    const bool gpuPath = !forceCpu && mImpl->scanPipe && mImpl->cq->gpuReady() && k <= kMaxK;
    if (!gpuPath) {
        searchCpu(*mImpl->cq, mImpl->reorderedDb, dim, mImpl->metric,
                  qPtr, m, k, nprobe, out);
        return out;
    }

    // Coarse: the m × nprobe nearest cells per query (GPU FlatIndex or CPU).
    std::vector<int32_t> probed;
    mImpl->cq->probeCells(qPtr, m, nprobe, probed);
    const int32_t* probedPtr = probed.data();

    // Query-tiled fine scan: when the batch is large enough that cells are
    // probed by many queries (lambda = avg queries/cell), grouping queries by
    // cell lets each staged db row serve ~Tq queries. Measured on M2 (SIFT1M,
    // m=1000): ~1.1x at lambda~31, ~1.15x at lambda~62; the scan is not purely
    // bandwidth-bound, so gains are modest — gate to where it clearly wins.
    // Gated to dim%4==0 (half4 staging) and k<=32 (owner register budget);
    // MFLAT_IVF_TILED=1/0 forces it on/off for benchmarking.
    {
        const double lambda = static_cast<double>(m) * nprobe / std::max(1, mImpl->nlist);
        const char* tEnv = std::getenv("MFLAT_IVF_TILED");
        const uint32_t Tq = tileTq(), Cv = tileCv();
        const size_t tileMem = static_cast<size_t>(Tq) * dim * sizeof(float)
                             + static_cast<size_t>(Cv) * dim * sizeof(uint16_t)
                             + static_cast<size_t>(Cv) * Tq * sizeof(float);
        bool tiled = mImpl->tiledPipe && mImpl->mergePipe
                  && (dim & 3) == 0 && k <= 32
                  && static_cast<uint64_t>(m) * nprobe < UINT32_MAX
                  && static_cast<uint64_t>(m) * nprobe * k * 8ull <= (512ull << 20)
                  && tileMem <= 30000   // partial buffers capped at 512 MB
                  && (tEnv ? tEnv[0] == '1' : (lambda >= 16.0 && m >= 256));
        if (tiled)   // negative cell ids would corrupt the inversion histogram
            for (const int32_t c : probed) if (c < 0) { tiled = false; break; }
        if (tiled) {
            mImpl->runTiledGpu(qPtr, probed, m, k, nprobe, out);
            return out;
        }
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
        [enc setBuffer:mImpl->cq->idBuffer()   offset:0 atIndex:1];
        [enc setBuffer:mImpl->cq->cellBuffer() offset:0 atIndex:2];
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
