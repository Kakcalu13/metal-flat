// SPDX-License-Identifier: Apache-2.0
// MetalFlat — FlatIndex implementation (Objective-C++ / Metal).
//
// Search = tiled GEMM + running top-k. The database is processed in
// column-tiles: for each block of db rows, GemmDistance (MPS GEMM,
// near-peak FP32) computes that block's dot products G_tile = Q·D_blockᵀ
// into a small reused buffer, then a merge kernel folds the tile into a
// per-query running top-k held in GPU memory. The full m×N score matrix
// is never materialized, so memory is bounded (≈ m×tile) regardless of
// database size — no OOM ceiling at large N.
//
// L2 uses the identity ‖q−d‖² = ‖q‖² + ‖d‖² − 2·(q·d) with CPU-
// precomputed squared norms. Cosine normalizes vectors (→ InnerProduct
// on unit vectors); InnerProduct ranks on the dot directly.
//
// v1 limits (roadmap): the per-tile merge is one-thread-per-query (the
// parallel warp-select that speeds top-k further is the next step); k is
// capped at 64.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "metalflat/FlatIndex.h"
#include "GemmDistance.h"

namespace mflat {

namespace {

// Matches the `MergeParams` struct in the kernel below, field-for-field.
struct MergeParams {
    uint32_t tileW;        // db rows in this tile
    uint32_t tileBase;     // global index of the tile's first db row
    uint32_t k;
    uint32_t metric;       // 0 = L2, 1 = InnerProduct, 2 = Cosine
    uint32_t queryCount;
};

NSString* const kShaderSrc = @R"(
#include <metal_stdlib>
using namespace metal;

struct MergeParams { uint tileW; uint tileBase; uint k; uint metric; uint queryCount; };

constant uint kMaxK = 64;

// One thread per query. Folds one tile of dot products into the query's
// running top-k. `run*` hold the best-so-far across previous tiles
// (ascending: index 0 = worst kept); this loads them, scans the tile's
// tileW dots, updates, and writes them back. L2 reconstructs squared
// distance from ‖q−d‖² = ‖q‖² + ‖d‖² − 2·dot (clamped ≥ 0); ranking is
// by "score" (larger = better) so L2 uses −dist² and one path serves
// all metrics. Tile-local ids are offset by tileBase to stay global.
kernel void topk_merge(
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
    const float qn = isL2 ? qnorm[qi] : 0.0;

    device float* rS = runScore + (uint64_t)qi * k;
    device int*   rI = runId    + (uint64_t)qi * k;

    float bestScore[kMaxK];
    int   bestId[kMaxK];
    for (uint i = 0; i < k; ++i) { bestScore[i] = rS[i]; bestId[i] = rI[i]; }

    for (uint loc = 0; loc < p.tileW; ++loc) {
        const float dot = row[loc];
        const uint  gid = p.tileBase + loc;
        float score;
        if (isL2) {
            score = -max(0.0, qn + dnorm[gid] - 2.0 * dot);
        } else {
            score = dot;
        }
        if (score > bestScore[0]) {
            uint pos = 0;
            while (pos + 1u < k && score > bestScore[pos + 1u]) {
                bestScore[pos] = bestScore[pos + 1u];
                bestId[pos]    = bestId[pos + 1u];
                ++pos;
            }
            bestScore[pos] = score;
            bestId[pos]    = (int)gid;
        }
    }

    for (uint i = 0; i < k; ++i) { rS[i] = bestScore[i]; rI[i] = bestId[i]; }
}
)";

void normalizeRows(std::vector<float>& v, int n, int dim) {
    for (int i = 0; i < n; ++i) {
        float* row = &v[static_cast<size_t>(i) * dim];
        double s = 0.0;
        for (int c = 0; c < dim; ++c) s += double(row[c]) * row[c];
        if (s > 0.0) {
            const float inv = static_cast<float>(1.0 / std::sqrt(s));
            for (int c = 0; c < dim; ++c) row[c] *= inv;
        }
    }
}

// Squared L2 norm per row → out[i] = ‖row_i‖². Feeds the L2 identity.
void rowSqNorms(const float* v, int n, int dim, std::vector<float>& out) {
    out.resize(n);
    for (int i = 0; i < n; ++i) {
        const float* row = v + static_cast<size_t>(i) * dim;
        double s = 0.0;
        for (int c = 0; c < dim; ++c) s += double(row[c]) * row[c];
        out[i] = static_cast<float>(s);
    }
}

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

}  // namespace

struct FlatIndex::Impl {
    int    dim    = 0;
    Metric metric = Metric::L2;

    id<MTLDevice>               device    = nil;
    id<MTLCommandQueue>         queue     = nil;
    id<MTLComputePipelineState> mergePipe = nil;
    std::unique_ptr<GemmDistance> gemm;

    std::vector<float> dbCpu;          // row-major, authoritative
    int                dbCount  = 0;
    id<MTLBuffer>      dbBuf    = nil;
    id<MTLBuffer>      dnormBuf = nil;  // n squared norms (for the L2 identity)
    bool               dirty    = false;

    // Reused per-tile score buffer (m × tileW), grown on demand. GPU-
    // private: MPS writes it, the merge kernel reads it, CPU never does.
    id<MTLBuffer> tileBuf = nil;
    size_t        tileCap = 0;          // capacity in floats

    bool ready   = false;
    bool warnedK = false;

    void ensureBuffer() {
        if (!dirty) return;
        if (dbCount > 0) {
            dbBuf = [device newBufferWithBytes:dbCpu.data()
                                        length:dbCpu.size() * sizeof(float)
                                       options:MTLResourceStorageModeShared];
            dbBuf.label = @"mflat_db";
            std::vector<float> dn;
            rowSqNorms(dbCpu.data(), dbCount, dim, dn);
            dnormBuf = [device newBufferWithBytes:dn.data()
                                           length:dn.size() * sizeof(float)
                                          options:MTLResourceStorageModeShared];
            dnormBuf.label = @"mflat_dnorm";
        } else {
            dbBuf = nil;
            dnormBuf = nil;
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
};

FlatIndex::FlatIndex(int dim, Metric metric)
    : mImpl(std::make_unique<Impl>()) {
    mImpl->dim    = dim;
    mImpl->metric = metric;

    mImpl->device = acquireMetalDevice();
    if (!mImpl->device) {
        fprintf(stderr, "[metalflat] no Metal device — using CPU fallback\n");
        return;
    }
    mImpl->queue = [mImpl->device newCommandQueue];

    NSError* err = nil;
    id<MTLLibrary> lib = [mImpl->device newLibraryWithSource:kShaderSrc
                                                     options:nil
                                                       error:&err];
    if (!lib) {
        fprintf(stderr, "[metalflat] shader compile failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
        return;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"topk_merge"];
    mImpl->mergePipe = [mImpl->device newComputePipelineStateWithFunction:fn
                                                                    error:&err];
    if (!mImpl->mergePipe) {
        fprintf(stderr, "[metalflat] pipeline build failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
        return;
    }

    mImpl->gemm = std::make_unique<GemmDistance>(mImpl->device);
    if (!mImpl->gemm->ready()) {
        fprintf(stderr, "[metalflat] MPS GEMM unavailable — using CPU fallback\n");
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
    if (mImpl->metric == Metric::Cosine) {
        std::vector<float> tmp(mImpl->dbCpu.begin() + base, mImpl->dbCpu.end());
        normalizeRows(tmp, n, mImpl->dim);
        std::copy(tmp.begin(), tmp.end(), mImpl->dbCpu.begin() + base);
    }
    mImpl->dbCount += n;
    mImpl->dirty = true;
}

void FlatIndex::reset() {
    mImpl->dbCpu.clear();
    mImpl->dbCount = 0;
    mImpl->dbBuf    = nil;
    mImpl->dnormBuf = nil;
    mImpl->dirty    = false;
}

namespace {

// Exact CPU top-k — the fallback when Metal/MPS is unavailable. Direct
// (q−d)² so it doubles as the numerically-cleanest reference.
void cpuSearch(const std::vector<float>& db, int dbCount, int dim,
               Metric metric, const float* queries, int m, int k,
               SearchResult& out) {
    const bool isL2 = (metric == Metric::L2);
    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k,
                         isL2 ? INFINITY : -INFINITY);

    std::vector<float> bestScore(k);
    std::vector<int>   bestId(k);
    for (int qi = 0; qi < m; ++qi) {
        const float* q = queries + static_cast<size_t>(qi) * dim;
        for (int i = 0; i < k; ++i) { bestScore[i] = -INFINITY; bestId[i] = -1; }
        for (int j = 0; j < dbCount; ++j) {
            const float* d = &db[static_cast<size_t>(j) * dim];
            float score;
            if (isL2) {
                float acc = 0.0f;
                for (int c = 0; c < dim; ++c) { float e = q[c] - d[c]; acc += e * e; }
                score = -acc;
            } else {
                float acc = 0.0f;
                for (int c = 0; c < dim; ++c) acc += q[c] * d[c];
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
                bestId[pos]    = j;
            }
        }
        for (int i = 0; i < k; ++i) {
            int src = k - 1 - i;
            out.ids[static_cast<size_t>(qi) * k + i] = bestId[src];
            out.distances[static_cast<size_t>(qi) * k + i] =
                isL2 ? -bestScore[src] : bestScore[src];
        }
    }
}

}  // namespace

SearchResult FlatIndex::search(const float* queries, int m, int k) {
    SearchResult out;
    if (m <= 0 || mImpl->dim <= 0) return out;
    if (k < 1) k = 1;
    if (k > kMaxK) {
        if (!mImpl->warnedK) {
            fprintf(stderr,
                "[metalflat] k=%d clamped to kMaxK=%d (v1 GPU limit)\n",
                k, kMaxK);
            mImpl->warnedK = true;
        }
        k = kMaxK;
    }

    const int dim = mImpl->dim;

    // Cosine: normalize queries into a local copy before scoring.
    std::vector<float> qNorm;
    const float* qPtr = queries;
    if (mImpl->metric == Metric::Cosine) {
        qNorm.assign(queries, queries + static_cast<size_t>(m) * dim);
        normalizeRows(qNorm, m, dim);
        qPtr = qNorm.data();
    }

    if (!mImpl->ready) {
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

    const int n      = mImpl->dbCount;
    const int tileW  = chooseTileWidth(m, n);

    // Per-query squared norms for the L2 identity (cheap; computed
    // regardless of metric so the kernel binding stays uniform).
    std::vector<float> qn;
    rowSqNorms(qPtr, m, dim, qn);

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

        const NSUInteger tg = std::min<NSUInteger>(
            64, mImpl->mergePipe.maxTotalThreadsPerThreadgroup);

        // One command buffer for the whole search: per tile, GEMM the
        // block then merge it into the running top-k. Reusing tileBuf
        // across tiles is safe — Metal hazard-tracks it, serialising
        // each tile's GEMM-write after the prior tile's merge-read.
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

            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:mImpl->mergePipe];
            [enc setBuffer:tileBuf        offset:0 atIndex:0];
            [enc setBuffer:qnormBuf       offset:0 atIndex:1];
            [enc setBuffer:mImpl->dnormBuf offset:0 atIndex:2];
            [enc setBuffer:runScore       offset:0 atIndex:3];
            [enc setBuffer:runId          offset:0 atIndex:4];
            [enc setBytes:&p length:sizeof(p) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(m), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
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
