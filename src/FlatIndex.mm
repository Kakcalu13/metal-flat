// SPDX-License-Identifier: Apache-2.0
// MetalFlat — FlatIndex implementation (Objective-C++ / Metal).
//
// Search = tiled GEMM + parallel top-k selection. The database is processed
// in column-tiles: for each block of db rows, GemmDistance (MPS GEMM,
// near-peak FP32) computes that block's dot products G_tile = Q·D_blockᵀ
// into a small reused buffer, then a selection stage folds the tile into a
// per-query running top-k held in GPU memory. The full m×N score matrix
// is never materialized, so memory is bounded (≈ m×tile) regardless of
// database size — no OOM ceiling at large N.
//
// Selection is workload-adaptive:
//   - tiny workloads      -> exact multithreaded CPU scan (GPU dispatch has a
//                            fixed ~1.5-3 ms cost that dwarfs the work).
//   - m >> tile rows      -> topk_merge_serial: one thread per query (the m
//                            threads alone saturate the GPU; e.g. k-means
//                            assignment / coarse probing, m ~ 10^6, N ~ 10^3).
//   - everything else     -> topk_partial (+ topk_merge_partials): the tile is
//                            split into segments, one THREADGROUP per
//                            (query, segment) scans cooperatively with a
//                            per-thread register top-k, then a simdgroup
//                            shuffle merge + cross-simdgroup tree merge
//                            reduce to one list. numSeg > 1 keeps the GPU
//                            full even at m = 1 (single-query latency).
//   - k > kMaxK           -> GEMM tiles + exact multithreaded CPU heap
//                            selection (double-buffered so the GPU computes
//                            tile t+1 while the CPU selects over tile t).
//                            No more silent clamp to kMaxK.
//
// L2 uses the identity ‖q−d‖² = ‖q‖² + ‖d‖² − 2·(q·d) with CPU-
// precomputed squared norms. Cosine normalizes vectors (→ InnerProduct
// on unit vectors); InnerProduct ranks on the dot directly.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "metalflat/FlatIndex.h"
#include "GemmDistance.h"
#include "Log_internal.h"
#include "Distance.h"   // mflat::detail::{dot,sqL2,rowSqNorms,normalizeRows,score,parallelFor,...}

namespace mflat {

namespace {

// Matches the `MergeParams` struct in the kernel below, field-for-field.
// segW / numSeg / fold are used by the cooperative kernels only; the serial
// kernel ignores them.
struct MergeParams {
    uint32_t tileW;        // db rows in this tile
    uint32_t tileBase;     // global index of the tile's first db row
    uint32_t k;
    uint32_t metric;       // 0 = L2, 1 = InnerProduct, 2 = Cosine
    uint32_t queryCount;
    uint32_t segW;         // rows per segment (cooperative kernels)
    uint32_t numSeg;       // segments in this tile
    uint32_t fold;         // 1 = topk_partial folds the running top-k itself
};

NSString* const kShaderSrc = @R"(
#include <metal_stdlib>
using namespace metal;

struct MergeParams {
    uint tileW; uint tileBase; uint k; uint metric; uint queryCount;
    uint segW; uint numSeg; uint fold;
};

constant uint kMaxK = 64;

// The per-thread top-k lists are kept ASCENDING over kk = nextPow2(k) slots
// (index 0 = worst kept; unfilled slots hold -INF/-1). Rounding k up to a
// power of two lets two lists be merged with the classic bitonic trick:
// c[i] = max(a[i], b[kk-1-i]) holds the kk largest of the union and is a
// bitonic sequence, so a log2(kk)-stage bitonic merge re-sorts it. The top k
// of the kk kept is exactly the true top k.

inline void insertTopk(thread float* s, thread int* id, uint kk, float sc, int gid) {
    uint pos = 0;
    while (pos + 1u < kk && sc > s[pos + 1u]) {
        s[pos] = s[pos + 1u];
        id[pos] = id[pos + 1u];
        ++pos;
    }
    s[pos] = sc;
    id[pos] = gid;
}

// Merge this lane's ascending kk-list with lane^off's (via simd shuffle):
// pairwise max against the partner's reversed list, then bitonic re-sort.
// After the butterfly over off = 1,2,...,16 every lane holds the simdgroup's
// merged top-kk.
inline void simdMergeTopk(thread float* s, thread int* id, uint kk, uint off) {
    float ns[kMaxK];
    int   ni[kMaxK];
    for (uint i = 0; i < kk; ++i) {
        const float os = simd_shuffle_xor(s[kk - 1u - i], off);
        const int   oi = simd_shuffle_xor(id[kk - 1u - i], off);
        if (os > s[i]) { ns[i] = os;   ni[i] = oi; }
        else           { ns[i] = s[i]; ni[i] = id[i]; }
    }
    for (uint i = 0; i < kk; ++i) { s[i] = ns[i]; id[i] = ni[i]; }
    for (uint st = kk >> 1; st > 0u; st >>= 1)
        for (uint i = 0; i < kk; ++i) {
            const uint j = i | st;
            if ((i & st) == 0u && j < kk && s[i] > s[j]) {
                const float ts = s[i]; s[i] = s[j]; s[j] = ts;
                const int   ti = id[i]; id[i] = id[j]; id[j] = ti;
            }
        }
}

// Two-pointer merge of two ascending kk-lists in threadgroup memory,
// keeping the kk largest in a (same as the IVF kernels' tree merge step).
inline void mergeListsTg(threadgroup float* aS, threadgroup int* aI,
                         threadgroup float* bS, threadgroup int* bI, uint kk) {
    float mS[kMaxK];
    int   mI[kMaxK];
    int ia = (int)kk - 1, ib = (int)kk - 1;
    for (int o = (int)kk - 1; o >= 0; --o) {
        const float av = (ia >= 0) ? aS[ia] : -INFINITY;
        const float bv = (ib >= 0) ? bS[ib] : -INFINITY;
        if (av >= bv) { mS[o] = av; mI[o] = (ia >= 0) ? aI[ia] : -1; --ia; }
        else          { mS[o] = bv; mI[o] = (ib >= 0) ? bI[ib] : -1; --ib; }
    }
    for (uint i = 0; i < kk; ++i) { aS[i] = mS[i]; aI[i] = mI[i]; }
}

// Reduce the calling threadgroup's per-thread lists to ONE ascending kk-list
// in redScore/redId[0..kk): simdgroup shuffle butterfly (register-level, no
// barriers), lane 0 of each simdgroup publishes, then a cross-simdgroup tree
// merge. Scratch needed: (tgs/32) * kk entries — small enough that k never
// constrains the threadgroup size. tgs is a power of two >= 32 (host-set).
inline void reduceTopkTg(thread float* s, thread int* id, uint kk,
                         threadgroup float* redScore, threadgroup int* redId,
                         uint tid, uint tgs, uint sgid, uint lane) {
    for (uint off = 1u; off < 32u; off <<= 1) simdMergeTopk(s, id, kk, off);
    if (lane == 0u)
        for (uint i = 0; i < kk; ++i) {
            redScore[sgid * kk + i] = s[i];
            redId[sgid * kk + i]    = id[i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = (tgs >> 5) >> 1; off > 0u; off >>= 1) {
        if (tid < off)
            mergeListsTg(redScore + tid * kk, redId + tid * kk,
                         redScore + (tid + off) * kk, redId + (tid + off) * kk, kk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Tile scores, shared by all kernels. L2 reconstructs squared distance from
// ‖q−d‖² = ‖q‖² + ‖d‖² − 2·dot (clamped ≥ 0); ranking is by "score"
// (larger = better) so L2 uses −dist² and one path serves all metrics.
inline float tileScore(float dot, float qn, float dn, bool isL2) {
    return isL2 ? -max(0.0f, qn + dn - 2.0f * dot) : dot;
}

// Cooperative partial top-k: one threadgroup per (segment, query). Threads
// stride the segment's dots keeping register top-kk lists, then reduce to one
// list. fold==0: write the segment's top-k (ascending) to its pScore/pId
// slot (phase 2 merges). fold==1 (numSeg==1): also fold the query's running
// top-k in as candidates and write the result straight back to runScore/runId
// — no phase 2 dispatch needed.
kernel void topk_partial(
    device const float*  Gtile    [[buffer(0)]],
    device const float*  qnorm    [[buffer(1)]],
    device const float*  dnorm    [[buffer(2)]],
    device float*        pScore   [[buffer(3)]],   // (queryCount*numSeg) × k
    device int*          pId      [[buffer(4)]],
    constant MergeParams& p       [[buffer(5)]],
    device float*        runScore [[buffer(6)]],   // queryCount × k, ascending
    device int*          runId    [[buffer(7)]],
    threadgroup float*   redScore [[threadgroup(0)]],   // (tgs/32) × kk
    threadgroup int*     redId    [[threadgroup(1)]],
    // Position attributes must be uniformly vector-typed (MSL rejects a
    // uint2 grid position next to scalar ones); simdgroup indices stay uint.
    uint2 tg   [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tgs2 [[threads_per_threadgroup]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const uint tid = tid2.x, tgs = tgs2.x;
    const uint qi = tg.x, seg = tg.y;   // m in x (can be huge), numSeg in y
    if (qi >= p.queryCount || seg >= p.numSeg) return;
    const uint k  = min(p.k, kMaxK);
    uint kk = 1; while (kk < k) kk <<= 1;
    const bool isL2 = (p.metric == 0u);
    const uint lo = seg * p.segW;
    const uint hi = min(lo + p.segW, p.tileW);
    device const float* row = Gtile + (uint64_t)qi * p.tileW;
    const float qn = isL2 ? qnorm[qi] : 0.0f;

    float bestScore[kMaxK];
    int   bestId[kMaxK];
    for (uint i = 0; i < kk; ++i) { bestScore[i] = -INFINITY; bestId[i] = -1; }

    for (uint loc = lo + tid; loc < hi; loc += tgs) {
        const uint gid = p.tileBase + loc;
        const float sc = tileScore(row[loc], qn, isL2 ? dnorm[gid] : 0.0f, isL2);
        if (sc > bestScore[0]) insertTopk(bestScore, bestId, kk, sc, (int)gid);
    }
    if (p.fold != 0u) {
        device const float* rS = runScore + (uint64_t)qi * k;
        device const int*   rI = runId    + (uint64_t)qi * k;
        for (uint e = tid; e < k; e += tgs) {
            if (rI[e] >= 0 && rS[e] > bestScore[0])
                insertTopk(bestScore, bestId, kk, rS[e], rI[e]);
        }
    }

    reduceTopkTg(bestScore, bestId, kk, redScore, redId, tid, tgs, sgid, lane);

    if (tid == 0u) {
        device float* oS = (p.fold != 0u) ? runScore + (uint64_t)qi * k
                                          : pScore + ((uint64_t)qi * p.numSeg + seg) * k;
        device int*   oI = (p.fold != 0u) ? runId + (uint64_t)qi * k
                                          : pId + ((uint64_t)qi * p.numSeg + seg) * k;
        for (uint i = 0; i < k; ++i) {
            oS[i] = redScore[kk - k + i];
            oI[i] = redId[kk - k + i];
        }
    }
}

// Phase 2 (numSeg > 1 only): one threadgroup per query merges its numSeg
// partial k-lists plus the running top-k into the new running top-k.
kernel void topk_merge_partials(
    device const float*  pScore   [[buffer(0)]],
    device const int*    pId      [[buffer(1)]],
    device float*        runScore [[buffer(2)]],
    device int*          runId    [[buffer(3)]],
    constant MergeParams& p       [[buffer(4)]],
    threadgroup float*   redScore [[threadgroup(0)]],
    threadgroup int*     redId    [[threadgroup(1)]],
    uint qi   [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    if (qi >= p.queryCount) return;
    const uint k  = min(p.k, kMaxK);
    uint kk = 1; while (kk < k) kk <<= 1;
    const uint nPart = p.numSeg * k;
    device const float* qS = pScore + (uint64_t)qi * nPart;
    device const int*   qI = pId    + (uint64_t)qi * nPart;
    device float*       rS = runScore + (uint64_t)qi * k;
    device int*         rI = runId    + (uint64_t)qi * k;

    float bestScore[kMaxK];
    int   bestId[kMaxK];
    for (uint i = 0; i < kk; ++i) { bestScore[i] = -INFINITY; bestId[i] = -1; }

    for (uint e = tid; e < nPart + k; e += tgs) {
        const float s  = (e < nPart) ? qS[e] : rS[e - nPart];
        const int   id = (e < nPart) ? qI[e] : rI[e - nPart];
        if (id >= 0 && s > bestScore[0]) insertTopk(bestScore, bestId, kk, s, id);
    }

    reduceTopkTg(bestScore, bestId, kk, redScore, redId, tid, tgs, sgid, lane);

    if (tid == 0u)
        for (uint i = 0; i < k; ++i) {
            rS[i] = redScore[kk - k + i];
            rI[i] = redId[kk - k + i];
        }
}

// One thread per query, whole tile scanned serially. Optimal when m alone
// saturates the GPU (large batches; k-means assignment; coarse probing) —
// per-thread state is touched once per tile, not once per (segment,thread),
// and each thread streams its own contiguous Gtile row (unit stride).
// The `worst` register keeps the hot compare off the (stack-backed) top-k
// array. (A transposed-tile "coalesced" variant measured strictly slower.)
kernel void topk_merge_serial(
    device const float*  Gtile    [[buffer(0)]],
    device const float*  qnorm    [[buffer(1)]],
    device const float*  dnorm    [[buffer(2)]],
    device float*        runScore [[buffer(3)]],
    device int*          runId    [[buffer(4)]],
    constant MergeParams& p       [[buffer(5)]],
    uint                 qi       [[thread_position_in_grid]])
{
    if (qi >= p.queryCount) return;
    const uint  k    = min(p.k, kMaxK);
    const bool  isL2 = (p.metric == 0u);
    device const float* row = Gtile + (uint64_t)qi * p.tileW;
    const float qn = isL2 ? qnorm[qi] : 0.0f;

    device float* rS = runScore + (uint64_t)qi * k;
    device int*   rI = runId    + (uint64_t)qi * k;

    float bestScore[kMaxK];
    int   bestId[kMaxK];
    for (uint i = 0; i < k; ++i) { bestScore[i] = rS[i]; bestId[i] = rI[i]; }
    float worst = bestScore[0];

    for (uint loc = 0; loc < p.tileW; ++loc) {
        const uint gid = p.tileBase + loc;
        const float sc = tileScore(row[loc], qn, isL2 ? dnorm[gid] : 0.0f, isL2);
        if (sc > worst) {
            insertTopk(bestScore, bestId, k, sc, (int)gid);
            worst = bestScore[0];
        }
    }

    for (uint i = 0; i < k; ++i) { rS[i] = bestScore[i]; rI[i] = bestId[i]; }
}
)";

// normalizeRows / rowSqNorms / dot / sqL2 / score / parallelFor come from
// Distance.h (mflat::detail) — shared with the IVF sources.

// Acquire a Metal device robustly. MTLCreateSystemDefaultDevice() can
// return nil in valid session contexts (certain logins / headless-ish
// setups) even when a usable GPU is present — observed on an M2 Pro
// Mac mini where MTLCopyAllDevices() lists the GPU but the "system
// default" query returns nil. Fall back to the first reported device.
id<MTLDevice> acquireMetalDevice() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev) return dev;
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    return all.count > 0 ? all[0] : nil;
}

// Tile-width budget: cap the reused score tile (m × tileW floats) at
// ~128 MB so memory stays bounded no matter how large N (or m) gets.
int chooseTileWidth(int m, int n) {
    const long kTileBytes = 128L * 1024 * 1024;
    long w = kTileBytes / (static_cast<long>(m) * static_cast<long>(sizeof(float)));
    if (w < 256)  w = 256;
    if (w > n)    w = n;
    return static_cast<int>(w);
}

uint32_t nextPow2(uint32_t x) {
    uint32_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

// CPU-vs-GPU routing: the GPU dispatch has a fixed ~1.5-3 ms floor; below
// this much scoring work the multithreaded exact CPU scan wins (measured on
// M2 Pro via temp/flat_sweep.mm). MFLAT_FLAT_CPU=1/0 forces CPU/GPU.
bool routeToCpu(int m, int n, int dim) {
    if (const char* e = std::getenv("MFLAT_FLAT_CPU")) return e[0] == '1';
    return static_cast<double>(m) * n * dim <= 32.0 * 1024.0 * 1024.0;
}

// Serial-vs-cooperative selection. Serial (one thread per query) wins once
// m alone fills the GPU; the cooperative reduction's cost grows with
// kk = nextPow2(k), so its crossover shrinks as k grows. Cuts measured on
// M2 Pro / SIFT-like shapes via temp/flat_cross.mm (k=10: coop wins <=512,
// serial >=1024; k=32: coop wins only <=128; k=64: serial from 128).
// MFLAT_FLAT_SERIAL=1/0 forces it.
bool routeToSerial(int m, int tileW, int k) {
    if (const char* e = std::getenv("MFLAT_FLAT_SERIAL")) return e[0] == '1';
    if (static_cast<uint64_t>(m) >= 32ull * static_cast<uint64_t>(tileW))
        return true;   // k-means / coarse-probe shape: tiny tile, huge m
    const int cut = (k <= 16) ? 768 : (k <= 32) ? 192 : 128;
    return m >= cut;
}

}  // namespace

struct FlatIndex::Impl {
    int    dim    = 0;
    Metric metric = Metric::L2;

    id<MTLDevice>               device      = nil;
    id<MTLCommandQueue>         queue       = nil;
    id<MTLComputePipelineState> partialPipe = nil;   // cooperative phase 1
    id<MTLComputePipelineState> mergePipe   = nil;   // cooperative phase 2
    id<MTLComputePipelineState> serialPipe  = nil;   // one-thread-per-query
    std::unique_ptr<GemmDistance> gemm;

    std::vector<float> dbCpu;          // row-major, authoritative
    int                dbCount  = 0;
    id<MTLBuffer>      dbBuf    = nil;
    id<MTLBuffer>      dnormBuf = nil;  // n squared norms (for the L2 identity)
    std::vector<float> dnormCpu;        // same values, for the CPU-selection path
    bool               dirty    = false;

    // Reused per-tile score buffer (m × tileW), grown on demand. GPU-
    // private: MPS writes it, the merge kernel reads it, CPU never does.
    id<MTLBuffer> tileBuf = nil;
    size_t        tileCap = 0;          // capacity in floats

    // Shared-storage tile pair for the k > kMaxK path (GPU writes, CPU
    // reads; two so selection over tile t overlaps the GEMM of tile t+1).
    id<MTLBuffer> tileShared[2] = {nil, nil};
    size_t        tileSharedCap = 0;    // capacity in floats (each)

    // Reused partial top-k buffers (m × numSeg × k), grown on demand.
    id<MTLBuffer> partScore = nil;
    id<MTLBuffer> partId    = nil;
    size_t        partCap   = 0;        // capacity in entries

    bool ready = false;

    void ensureBuffer() {
        if (!dirty) return;
        if (dbCount > 0) {
            dbBuf = [device newBufferWithBytes:dbCpu.data()
                                        length:dbCpu.size() * sizeof(float)
                                       options:MTLResourceStorageModeShared];
            dbBuf.label = @"mflat_db";
            dnormCpu.resize(dbCount);
            detail::rowSqNorms(dbCpu.data(), dbCount, dim, dnormCpu.data());
            dnormBuf = [device newBufferWithBytes:dnormCpu.data()
                                           length:dnormCpu.size() * sizeof(float)
                                          options:MTLResourceStorageModeShared];
            dnormBuf.label = @"mflat_dnorm";
        } else {
            dbBuf = nil;
            dnormBuf = nil;
            dnormCpu.clear();
        }
        dirty = false;
    }

    id<MTLBuffer> ensureTile(size_t floats) {
        if (floats > tileCap) {
            tileBuf = [device newBufferWithLength:floats * sizeof(float)
                                          options:MTLResourceStorageModePrivate];
            tileBuf.label = @"mflat_tile";
            tileCap = floats;
        }
        return tileBuf;
    }

    void ensureSharedTiles(size_t floats) {
        if (floats > tileSharedCap) {
            for (int i = 0; i < 2; ++i) {
                tileShared[i] = [device newBufferWithLength:floats * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
                tileShared[i].label = @"mflat_tile_shared";
            }
            tileSharedCap = floats;
        }
    }

    void ensurePartials(size_t entries) {
        if (entries > partCap) {
            partScore = [device newBufferWithLength:entries * sizeof(float)
                                            options:MTLResourceStorageModePrivate];
            partId    = [device newBufferWithLength:entries * sizeof(int32_t)
                                            options:MTLResourceStorageModePrivate];
            partScore.label = @"mflat_partial_score";
            partId.label    = @"mflat_partial_id";
            partCap = entries;
        }
    }

    // GEMM tiles + multithreaded CPU heap selection — serves k > kMaxK
    // exactly (the GPU kernels' per-thread lists cap at kMaxK).
    SearchResult searchGemmCpuTopk(const float* qPtr, const std::vector<float>& qn,
                                   int m, int k);
};

FlatIndex::FlatIndex(int dim, Metric metric)
    : mImpl(std::make_unique<Impl>()) {
    mImpl->dim    = dim;
    mImpl->metric = metric;

    mImpl->device = acquireMetalDevice();
    if (!mImpl->device) {
        MFLAT_LOG_INFO("no Metal device — using CPU fallback");
        return;
    }
    mImpl->queue = [mImpl->device newCommandQueue];

    NSError* err = nil;
    id<MTLLibrary> lib = [mImpl->device newLibraryWithSource:kShaderSrc
                                                     options:nil
                                                       error:&err];
    if (!lib) {
        MFLAT_LOG_ERROR("shader compile failed: %s",
                        err ? [[err localizedDescription] UTF8String] : "?");
        return;
    }
    auto pipe = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        id<MTLComputePipelineState> ps =
            fn ? [mImpl->device newComputePipelineStateWithFunction:fn error:&err] : nil;
        if (!ps)
            MFLAT_LOG_ERROR("pipeline %s build failed: %s", [name UTF8String],
                            err ? [[err localizedDescription] UTF8String] : "?");
        return ps;
    };
    mImpl->partialPipe = pipe(@"topk_partial");
    mImpl->mergePipe   = pipe(@"topk_merge_partials");
    mImpl->serialPipe  = pipe(@"topk_merge_serial");
    if (!mImpl->partialPipe || !mImpl->mergePipe || !mImpl->serialPipe) return;

    mImpl->gemm = std::make_unique<GemmDistance>(mImpl->device);
    if (!mImpl->gemm->ready()) {
        MFLAT_LOG_INFO("MPS GEMM unavailable — using CPU fallback");
        return;
    }
    mImpl->ready = true;
}

FlatIndex::~FlatIndex() = default;

bool   FlatIndex::ready()  const { return mImpl->ready; }
int    FlatIndex::size()   const { return mImpl->dbCount; }
int    FlatIndex::dim()    const { return mImpl->dim; }
Metric FlatIndex::metric() const { return mImpl->metric; }

void FlatIndex::add(const float* vectors, int n) {
    if (n <= 0 || mImpl->dim <= 0) return;
    const size_t base = mImpl->dbCpu.size();
    mImpl->dbCpu.insert(mImpl->dbCpu.end(), vectors,
                        vectors + static_cast<size_t>(n) * mImpl->dim);
    if (mImpl->metric == Metric::Cosine)
        detail::normalizeRows(&mImpl->dbCpu[base], n, mImpl->dim);   // in place
    mImpl->dbCount += n;
    mImpl->dirty = true;
}

void FlatIndex::reset() {
    mImpl->dbCpu.clear();
    mImpl->dbCount = 0;
    mImpl->dbBuf    = nil;
    mImpl->dnormBuf = nil;
    mImpl->dnormCpu.clear();
    mImpl->dirty    = false;
}

namespace {

// Exact CPU top-k — the tiny-workload route and the no-Metal fallback.
// Direct (q−d)² so it doubles as the numerically-cleanest reference.
// Multithreaded across queries; serves any k.
void cpuSearch(const std::vector<float>& db, int dbCount, int dim,
               Metric metric, const float* queries, int m, int k,
               SearchResult& out) {
    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k, detail::emptyValue(metric));

    detail::parallelFor(m, [&](int qi) {
        const float* q = queries + static_cast<size_t>(qi) * dim;
        std::vector<float> bestScore(k, -INFINITY);
        std::vector<int>   bestId(k, -1);
        for (int j = 0; j < dbCount; ++j) {
            const float s = detail::score(metric, q, &db[static_cast<size_t>(j) * dim], dim);
            if (s > bestScore[0]) {
                int pos = 0;
                while (pos + 1 < k && s > bestScore[pos + 1]) {
                    bestScore[pos] = bestScore[pos + 1];
                    bestId[pos]    = bestId[pos + 1];
                    ++pos;
                }
                bestScore[pos] = s;
                bestId[pos]    = j;
            }
        }
        for (int i = 0; i < k; ++i) {
            int src = k - 1 - i;
            out.ids[static_cast<size_t>(qi) * k + i] = bestId[src];
            out.distances[static_cast<size_t>(qi) * k + i] =
                detail::scoreToValue(metric, bestScore[src]);
        }
    });
}

}  // namespace

// k > kMaxK: GPU GEMM computes each tile's dot products into a shared-storage
// buffer; the CPU keeps a min-heap of the k best per query (any k). Double-
// buffered: the GEMM for tile t+1 is committed before the CPU selects over
// tile t, so compute and selection overlap.
SearchResult FlatIndex::Impl::searchGemmCpuTopk(const float* qPtr,
                                                const std::vector<float>& qn,
                                                int m, int k) {
    SearchResult out;
    const int n     = dbCount;
    const bool isL2 = (metric == Metric::L2);
    const int tileW = chooseTileWidth(2 * m, n);   // two buffers in flight

    using Cand = std::pair<float, int32_t>;        // (score, id)
    auto worse = [](const Cand& a, const Cand& b) { return a.first > b.first; };
    std::vector<Cand> heaps(static_cast<size_t>(m) * k);
    std::vector<int>  heapSize(m, 0);

    @autoreleasepool {
        id<MTLBuffer> qBuf = [device
            newBufferWithBytes:qPtr
                        length:static_cast<size_t>(m) * dim * sizeof(float)
                       options:MTLResourceStorageModeShared];
        ensureSharedTiles(static_cast<size_t>(m) * tileW);

        auto kick = [&](int base, int tw, int slot) -> id<MTLCommandBuffer> {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            gemm->encode(cb, qBuf, m, dbBuf, base, tw, dim, tileShared[slot]);
            [cb commit];
            return cb;
        };

        id<MTLCommandBuffer> pending[2] = {nil, nil};
        pending[0] = kick(0, std::min(tileW, n), 0);
        for (int base = 0, t = 0; base < n; base += tileW, ++t) {
            const int tw   = std::min(tileW, n - base);
            const int slot = t & 1;
            const int nextBase = base + tileW;
            if (nextBase < n)
                pending[slot ^ 1] = kick(nextBase, std::min(tileW, n - nextBase), slot ^ 1);

            [pending[slot] waitUntilCompleted];
            const float* G = static_cast<const float*>([tileShared[slot] contents]);
            detail::parallelFor(m, [&](int qi) {
                const float* row = G + static_cast<size_t>(qi) * tw;
                Cand* h  = &heaps[static_cast<size_t>(qi) * k];
                int&  sz = heapSize[qi];
                for (int j = 0; j < tw; ++j) {
                    const float s = isL2
                        ? detail::l2ScoreFromDot(qn[qi], dnormCpu[base + j], row[j])
                        : row[j];
                    if (sz < k) {
                        h[sz++] = {s, base + j};
                        if (sz == k) std::make_heap(h, h + k, worse);
                    } else if (s > h[0].first) {
                        std::pop_heap(h, h + k, worse);
                        h[k - 1] = {s, base + j};
                        std::push_heap(h, h + k, worse);
                    }
                }
            });
        }
    }

    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k, detail::emptyValue(metric));
    detail::parallelFor(m, [&](int qi) {
        Cand* h = &heaps[static_cast<size_t>(qi) * k];
        std::sort(h, h + heapSize[qi],
                  [](const Cand& a, const Cand& b) { return a.first > b.first; });
        for (int i = 0; i < heapSize[qi]; ++i) {
            out.ids[static_cast<size_t>(qi) * k + i]       = h[i].second;
            out.distances[static_cast<size_t>(qi) * k + i] =
                detail::scoreToValue(metric, h[i].first);
        }
    });
    return out;
}

SearchResult FlatIndex::search(const float* queries, int m, int k) {
    SearchResult out;
    if (m <= 0 || mImpl->dim <= 0) return out;
    if (k < 1) k = 1;

    const int dim = mImpl->dim;

    // Cosine: normalize queries into a local copy before scoring.
    std::vector<float> qNorm;
    const float* qPtr = queries;
    if (mImpl->metric == Metric::Cosine) {
        qNorm.assign(queries, queries + static_cast<size_t>(m) * dim);
        detail::normalizeRows(qNorm, m, dim);
        qPtr = qNorm.data();
    }

    // No Metal, or too little work to cover the GPU dispatch floor: exact CPU.
    if (!mImpl->ready || routeToCpu(m, mImpl->dbCount, dim)) {
        cpuSearch(mImpl->dbCpu, mImpl->dbCount, dim, mImpl->metric,
                  qPtr, m, k, out);
        return out;
    }

    mImpl->ensureBuffer();
    const bool isL2 = (mImpl->metric == Metric::L2);
    if (mImpl->dbCount == 0 || !mImpl->dbBuf) {
        out.ids.assign(static_cast<size_t>(m) * k, -1);
        out.distances.assign(static_cast<size_t>(m) * k,
                             isL2 ? INFINITY : -INFINITY);
        return out;
    }

    // Per-query squared norms for the L2 identity (cheap; computed
    // regardless of metric so the kernel binding stays uniform).
    std::vector<float> qn(m);
    detail::rowSqNorms(qPtr, m, dim, qn.data());

    // k above the GPU kernels' register top-k: GEMM + exact CPU selection.
    if (k > kMaxK)
        return mImpl->searchGemmCpuTopk(qPtr, qn, m, k);

    const int n     = mImpl->dbCount;
    const int tileW = chooseTileWidth(m, n);

    // Selection shape. Serial kernel when queries alone saturate the GPU;
    // otherwise cooperative threadgroups, split into numSeg segments so
    // small batches still fill the GPU.
    const bool serial = routeToSerial(m, tileW, k);
    uint32_t numSeg = 1, segW = static_cast<uint32_t>(tileW);
    NSUInteger tgW = 256;
    if (!serial) {
        const uint32_t wantSeg = static_cast<uint32_t>((512 + m - 1) / m);
        const uint32_t maxSeg  = static_cast<uint32_t>(std::max(1, tileW / 8192));
        numSeg = std::min({wantSeg, maxSeg, 256u});
        segW   = static_cast<uint32_t>((tileW + numSeg - 1) / numSeg);
        // Small per-query segments don't need 256 threads (>= ~64 rows each).
        while (tgW > 32 && segW < tgW * 64) tgW >>= 1;
        tgW = std::min({tgW, mImpl->partialPipe.maxTotalThreadsPerThreadgroup,
                        mImpl->mergePipe.maxTotalThreadsPerThreadgroup});
        NSUInteger pw = 32; while (pw * 2 <= tgW) pw *= 2; tgW = pw;
    }
    const uint32_t kk = nextPow2(static_cast<uint32_t>(k));
    const bool     fold = (numSeg == 1);

    @autoreleasepool {
        id<MTLBuffer> qBuf = [mImpl->device
            newBufferWithBytes:qPtr
                        length:static_cast<size_t>(m) * dim * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> qnormBuf = [mImpl->device
            newBufferWithBytes:qn.data()
                        length:static_cast<size_t>(m) * sizeof(float)
                       options:MTLResourceStorageModeShared];

        // Running per-query top-k, initialised to the empty state
        // (score -inf, id -1), ascending so index 0 is the worst kept.
        const size_t rkN = static_cast<size_t>(m) * k;
        id<MTLBuffer> runScore = [mImpl->device
            newBufferWithLength:rkN * sizeof(float)
                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> runId = [mImpl->device
            newBufferWithLength:rkN * sizeof(int32_t)
                        options:MTLResourceStorageModeShared];
        {
            float*   rs = static_cast<float*>([runScore contents]);
            int32_t* ri = static_cast<int32_t*>([runId contents]);
            for (size_t i = 0; i < rkN; ++i) { rs[i] = -INFINITY; ri[i] = -1; }
        }

        id<MTLBuffer> tileBuf =
            mImpl->ensureTile(static_cast<size_t>(m) * tileW);
        if (!serial && !fold)
            mImpl->ensurePartials(static_cast<size_t>(m) * numSeg * k);

        // One command buffer for the whole search: per tile, GEMM the
        // block then fold it into the running top-k. Reusing tileBuf
        // across tiles is safe — Metal hazard-tracks it, serialising
        // each tile's GEMM-write after the prior tile's read.
        id<MTLCommandBuffer> cb = [mImpl->queue commandBuffer];
        for (int base = 0; base < n; base += tileW) {
            const int tw = std::min(tileW, n - base);

            mImpl->gemm->encode(cb, qBuf, m, mImpl->dbBuf, base, tw, dim,
                                tileBuf);

            MergeParams p;
            p.tileW      = static_cast<uint32_t>(tw);
            p.tileBase   = static_cast<uint32_t>(base);
            p.k          = static_cast<uint32_t>(k);
            p.metric     = static_cast<uint32_t>(mImpl->metric);
            p.queryCount = static_cast<uint32_t>(m);
            p.segW       = segW;
            p.numSeg     = std::max(1u, (static_cast<uint32_t>(tw) + segW - 1) / segW);
            p.fold       = fold ? 1u : 0u;

            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (serial) {
                const NSUInteger tg = std::min<NSUInteger>(
                    64, mImpl->serialPipe.maxTotalThreadsPerThreadgroup);
                [enc setComputePipelineState:mImpl->serialPipe];
                [enc setBuffer:tileBuf         offset:0 atIndex:0];
                [enc setBuffer:qnormBuf        offset:0 atIndex:1];
                [enc setBuffer:mImpl->dnormBuf offset:0 atIndex:2];
                [enc setBuffer:runScore        offset:0 atIndex:3];
                [enc setBuffer:runId           offset:0 atIndex:4];
                [enc setBytes:&p length:sizeof(p) atIndex:5];
                [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(m), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            } else {
                // Scratch: (tgs/32) lists of kk entries; lengths 16B-aligned.
                const NSUInteger scratch = ((tgW / 32) * kk * 4 + 15) & ~NSUInteger(15);
                [enc setComputePipelineState:mImpl->partialPipe];
                [enc setBuffer:tileBuf         offset:0 atIndex:0];
                [enc setBuffer:qnormBuf        offset:0 atIndex:1];
                [enc setBuffer:mImpl->dnormBuf offset:0 atIndex:2];
                [enc setBuffer:(fold ? runScore : mImpl->partScore) offset:0 atIndex:3];
                [enc setBuffer:(fold ? runId    : mImpl->partId)    offset:0 atIndex:4];
                [enc setBytes:&p length:sizeof(p) atIndex:5];
                [enc setBuffer:runScore offset:0 atIndex:6];
                [enc setBuffer:runId    offset:0 atIndex:7];
                [enc setThreadgroupMemoryLength:scratch atIndex:0];
                [enc setThreadgroupMemoryLength:scratch atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(m), p.numSeg, 1)
                    threadsPerThreadgroup:MTLSizeMake(tgW, 1, 1)];
                if (!fold) {
                    [enc setComputePipelineState:mImpl->mergePipe];
                    [enc setBuffer:mImpl->partScore offset:0 atIndex:0];
                    [enc setBuffer:mImpl->partId    offset:0 atIndex:1];
                    [enc setBuffer:runScore         offset:0 atIndex:2];
                    [enc setBuffer:runId            offset:0 atIndex:3];
                    [enc setBytes:&p length:sizeof(p) atIndex:4];
                    [enc setThreadgroupMemoryLength:scratch atIndex:0];
                    [enc setThreadgroupMemoryLength:scratch atIndex:1];
                    [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(m), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tgW, 1, 1)];
                }
            }
            [enc endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];

        // Emit nearest-first. runScore/runId are ascending (index k-1 =
        // nearest); output the natural metric value (dist² for L2).
        out.ids.assign(rkN, -1);
        out.distances.assign(rkN, 0.0f);
        const float*   rs = static_cast<const float*>([runScore contents]);
        const int32_t* ri = static_cast<const int32_t*>([runId contents]);
        for (int qi = 0; qi < m; ++qi) {
            for (int i = 0; i < k; ++i) {
                const int src = k - 1 - i;
                const size_t o = static_cast<size_t>(qi) * k + i;
                const size_t s = static_cast<size_t>(qi) * k + src;
                out.ids[o]       = ri[s];
                out.distances[o] = isL2 ? -rs[s] : rs[s];
            }
        }
    }
    return out;
}

}  // namespace mflat
