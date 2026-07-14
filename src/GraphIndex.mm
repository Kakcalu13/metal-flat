// SPDX-License-Identifier: Apache-2.0
// GraphIndex.mm — CAGRA-style graph ANN index.
//
// Build: an R-degree approximate k-NN graph (forward nearest + reverse edges),
// constructed by REUSING IvfIndex as the batched-kNN primitive (search the
// database against itself, drop self). Search: a fixed-iteration best-first beam
// search on the GPU (one threadgroup per query, CAGRA single-CTA style), with a
// dynamic best-first CPU reference/fallback. All distances follow Distance.h.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <unordered_set>
#include <vector>

#include "metalflat/GraphIndex.h"
#include "metalflat/IvfIndex.h"
#include "Internal.h"        // detail::parallelFor, normalizeRows, sqL2, score, acquireMetalDevice
#include "PqTrainer.h"       // detail::PqTrainer — the traversal's LOSSLESS neighbour filter
#include "GpuScratch.h"      // persistent per-search buffers (no alloc per query)
#include "Log_internal.h"    // MFLAT_LOG_ERROR

namespace mflat {

using namespace detail;

// ---------------------------------------------------------------------------
// GPU kernel: fixed-iteration best-first beam search, one threadgroup / query.
//
// Each iteration expands the single best UNEXPANDED beam node (search_width=1,
// CAGRA single-CTA), computing distances to its R neighbours cooperatively, then
// merges them into the size-L beam via a cooperative threadgroup bitonic sort
// (sorted best-first, tie-broken on id so duplicate ids land adjacent and are
// dropped). A FIXED trip count keeps every thread hitting every barrier the same
// number of times — the only barrier-safe way to keep barriers inside the loop.
// ---------------------------------------------------------------------------
namespace {

NSString* const kGraphShaderSrc = @R"(
#include <metal_stdlib>
using namespace metal;

constant uint kGMaxK = 64u;

struct GraphParams {
    uint dim, k, L, R, Lc, numSeeds, maxIter, metric, queryCount, dbCount, entry, H, W;
};

inline uint gmix(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

// Per-query open-addressing "visited" hash (id+1, 0=empty; H is a power of two).
// Only thread 0 touches it (during parent selection), so no atomics are needed.
inline bool hashHas(threadgroup const int* vh, uint H, int id) {
    uint h = gmix((uint)id) & (H - 1u);
    for (uint p = 0; p < H; ++p) { int s = vh[h]; if (s == 0) return false; if (s == id + 1) return true; h = (h + 1u) & (H - 1u); }
    return true;
}
inline void hashMark(threadgroup int* vh, uint H, int id) {
    uint h = gmix((uint)id) & (H - 1u);
    for (uint p = 0; p < H; ++p) { int s = vh[h]; if (s == 0) { vh[h] = id + 1; return; } if (s == id + 1) return; h = (h + 1u) & (H - 1u); }
}

// Metric score for a node: larger = nearer. L2 => -||q-d||^2, IP/Cosine => q.d.
// fp16 db, fp32 accumulation — mirrors ivf_scan exactly.
inline float gdist(device const half* db, threadgroup const float* qsh,
                   int id, uint dim, bool isL2) {
    device const half* d = db + (uint64_t)id * dim;
    if ((dim & 3u) == 0u) {
        const uint c4 = dim >> 2;
        threadgroup const float4* q4 = (threadgroup const float4*)qsh;
        device const half4*       d4 = (device const half4*)d;
        float4 acc = float4(0.0);
        if (isL2) { for (uint c = 0; c < c4; ++c) { float4 e = q4[c] - float4(d4[c]); acc += e * e; } }
        else      { for (uint c = 0; c < c4; ++c) acc += q4[c] * float4(d4[c]); }
        float s = acc.x + acc.y + acc.z + acc.w;
        return isL2 ? -s : s;
    } else if (isL2) {
        float acc = 0.0; for (uint c = 0; c < dim; ++c) { float e = qsh[c] - float(d[c]); acc += e * e; }
        return -acc;
    } else {
        float acc = 0.0; for (uint c = 0; c < dim; ++c) acc += qsh[c] * float(d[c]);
        return acc;
    }
}

// Cooperative bitonic sort of `n` (power-of-two) entries, DESCENDING by
// (score, id). Tie-break on id keeps equal-id duplicates adjacent (so the dedup
// pass can drop them). Every thread hits every barrier the same number of times.
inline void gbitonic(threadgroup float* sc, threadgroup int* id,
                     uint n, uint tid, uint tgs) {
    for (uint size = 2u; size <= n; size <<= 1) {
        for (uint stride = size >> 1; stride > 0u; stride >>= 1) {
            for (uint i = tid; i < n; i += tgs) {
                uint j = i ^ stride;
                if (j > i) {
                    bool up    = ((i & size) == 0u);
                    bool iLess = (sc[i] <  sc[j]) || (sc[i] == sc[j] && id[i] <  id[j]);
                    bool iGrt  = (sc[i] >  sc[j]) || (sc[i] == sc[j] && id[i] >  id[j]);
                    if (up ? iLess : iGrt) {
                        float ts = sc[i]; sc[i] = sc[j]; sc[j] = ts;
                        int   ti = id[i]; id[i] = id[j]; id[j] = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void graph_search(
    device const half*    db      [[buffer(0)]],
    device const int*     graph   [[buffer(1)]],
    device const float*   queries [[buffer(2)]],
    device int*           outIds  [[buffer(3)]],
    device float*         outVal  [[buffer(4)]],
    constant GraphParams& p       [[buffer(5)]],
    threadgroup float*    qsh     [[threadgroup(0)]],   // dim
    threadgroup float*    lScore  [[threadgroup(1)]],   // L   (beam)
    threadgroup int*      lId     [[threadgroup(2)]],   // L
    threadgroup float*    mScore  [[threadgroup(3)]],   // Lc  (merge scratch)
    threadgroup int*      mId     [[threadgroup(4)]],   // Lc
    threadgroup int*      vhash   [[threadgroup(5)]],   // H   (visited set)
    uint qi  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    if (qi >= p.queryCount) return;            // uniform per threadgroup, before any barrier
    const uint dim = p.dim, L = p.L, R = p.R, Lc = p.Lc, H = p.H, W = p.W, k = min(p.k, kGMaxK);
    const bool isL2 = (p.metric == 0u);
    threadgroup int parents[16];               // up to W parents expanded per iteration
    threadgroup int haveParent;

    // Stage the query once; clear the visited hash.
    device const float* qg = queries + (uint64_t)qi * dim;
    for (uint c = tid; c < dim; c += tgs) qsh[c] = qg[c];
    for (uint i = tid; i < H;   i += tgs) vhash[i] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Empty beam.
    for (uint i = tid; i < L; i += tgs) { lScore[i] = -INFINITY; lId[i] = -1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Seeds: slot 0 = entry medoid, 1..numSeeds-1 = deterministic random restarts.
    for (uint s = tid; s < p.numSeeds && s < L; s += tgs) {
        int nid = (s == 0u) ? (int)p.entry
                            : (int)(gmix((uint)qi * 2654435761u + s) % p.dbCount);
        lScore[s] = gdist(db, qsh, nid, dim, isL2); lId[s] = nid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Order the seeds best-first so parent selection is meaningful.
    for (uint i = tid; i < Lc; i += tgs) {
        if (i < L) { mScore[i] = lScore[i]; mId[i] = lId[i]; }
        else       { mScore[i] = -INFINITY; mId[i] = -1;     }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gbitonic(mScore, mId, Lc, tid, tgs);
    for (uint i = tid; i < L; i += tgs) { lScore[i] = mScore[i]; lId[i] = mId[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint it = 0; it < p.maxIter; ++it) {
        // Pick the W best NOT-YET-VISITED beam nodes (beam is best-first). A
        // PERSISTENT visited set (not a beam-local flag) is essential: it stops
        // the search re-expanding evicted nodes and cycling on a fixed set.
        if (tid == 0u) {
            uint np = 0;
            for (uint i = 0; i < L && np < W; ++i) {
                int id = lId[i];
                if (id >= 0 && !hashHas(vhash, H, id)) { parents[np++] = id; hashMark(vhash, H, id); }
            }
            for (uint w = np; w < W; ++w) parents[w] = -1;
            haveParent = (np > 0u) ? 1 : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // haveParent is uniform across the threadgroup, so this break is taken by
        // every thread together (barrier-safe) — it ends the search the moment no
        // unvisited beam node remains, skipping the wasted no-op iterations.
        if (haveParent == 0) break;

        // Load beam into scratch [0,L); expand the W parents' neighbours into
        // [L, L+W*R). Duplicates (across parents / with the beam) are dropped below.
        for (uint i = tid; i < Lc; i += tgs) {
            if (i < L) { mScore[i] = lScore[i]; mId[i] = lId[i]; }
            else       { mScore[i] = -INFINITY; mId[i] = -1;     }
        }
        for (uint e = tid; e < W * R; e += tgs) {
            int par = parents[e / R];
            int nid = -1; float sc = -INFINITY;
            if (par >= 0) { nid = graph[(uint64_t)par * R + (e % R)];
                            if (nid >= 0) sc = gdist(db, qsh, nid, dim, isL2); }
            if (L + e < Lc) { mScore[L + e] = sc; mId[L + e] = nid; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        gbitonic(mScore, mId, Lc, tid, tgs);
        // Drop duplicate ids (equal id => equal score => adjacent after the sort).
        for (uint i = tid; i < Lc; i += tgs)
            if (i > 0u && mId[i] >= 0 && mId[i] == mId[i - 1]) mScore[i] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gbitonic(mScore, mId, Lc, tid, tgs);

        // Keep the top-L back in the beam.
        for (uint i = tid; i < L; i += tgs) { lScore[i] = mScore[i]; lId[i] = mId[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        device int*   oi = outIds + (uint64_t)qi * k;
        device float* ov = outVal + (uint64_t)qi * k;
        for (uint i = 0; i < k; ++i) { oi[i] = lId[i]; ov[i] = isL2 ? -lScore[i] : lScore[i]; }
    }
}
)";

// Host mirror of the kernel's GraphParams (identical field order/types).
struct GraphParams {
    uint32_t dim, k, L, R, Lc, numSeeds, maxIter, metric, queryCount, dbCount, entry, H, W;
};

// Coarse cells for the kNN self-search. The self-search scans nprobe * (n/nlist)
// candidates per node, so a FINER partition cuts build cost linearly: at a fixed
// nprobe, mul=4 quarters the scanned candidates. Graph quality tolerates it —
// the RNG prune keeps ~10 of the K0 candidates, and the beam search repairs the
// rest — so this is the build's cheapest lever (MFLAT_GRAPH_NLIST_MUL overrides).
inline int autoNlist(int n) {
    int mul = 4;
    if (const char* e = std::getenv("MFLAT_GRAPH_NLIST_MUL")) mul = std::max(1, atoi(e));
    const int s = static_cast<int>(std::sqrt(static_cast<double>(n))) * mul;
    return std::max(64, std::min(s, n));
}
// splitmix-ish deterministic hash for reproducible random restart seeds (CPU).
inline uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
inline uint32_t nextPow2(uint32_t x) {
    uint32_t p = 1; while (p < x) p <<= 1; return p;
}

// Raw-kNN cache (id matrix) so the slow self-search can be reused across builds.
bool saveKnn(const char* path, const std::vector<int32_t>& knn, int n, int k) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    const int32_t hdr[3] = { 0x4d464b4e /*'MFKN'*/, n, k };
    bool ok = std::fwrite(hdr, sizeof(int32_t), 3, f) == 3
            && std::fwrite(knn.data(), sizeof(int32_t), knn.size(), f) == knn.size();
    std::fclose(f);
    return ok;
}
bool loadKnn(const char* path, std::vector<int32_t>& knn, int n, int k) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    int32_t hdr[3];
    bool ok = std::fread(hdr, sizeof(int32_t), 3, f) == 3
            && hdr[0] == 0x4d464b4e && hdr[1] == n && hdr[2] == k;
    if (ok) { knn.resize(static_cast<size_t>(n) * k);
              ok = std::fread(knn.data(), sizeof(int32_t), knn.size(), f) == knn.size(); }
    std::fclose(f);
    return ok;
}

}  // namespace

struct GraphIndex::Impl {
    int    dim     = 0;
    Metric metric  = Metric::L2;
    int    R       = 32;
    int    dbCount = 0;
    int    entry   = 0;        // medoid id (a good default seed)
    bool   ready   = false;

    std::vector<float>   db;       // dbCount × dim (metric-normalized copy; CPU path)
    std::vector<int32_t> graph;    // dbCount × R adjacency (original ids, -1 pad)

    // LOSSLESS neighbour filter for the CPU traversal (see searchCpu).
    //   pqCodes: dbCount × pqM bytes (~16 MB at n=1M — sits in the SLC)
    //   pqErr  : dbCount floats, e_i = ||x_i - decode(code_i)||
    // ADC yields ||q - x̂||^2 EXACTLY, so ||q-x|| >= ||q-x̂|| - e_i is a hard
    // lower bound: a neighbour whose bound already exceeds the beam's worst kept
    // distance can never enter the beam, so its full row is never fetched. No
    // false negatives => results stay bit-identical to the plain exact walk, so
    // this buys latency without touching recall or CPU/GPU parity.
    std::unique_ptr<detail::PqTrainer> pq;
    std::vector<uint8_t>               pqCodes;
    std::vector<float>                 pqErr;
    int                                pqM = 0;
    // Codebook TRANSPOSED to [subspace][d][centroid]. PqTrainer's natural layout
    // ([centroid][d]) makes the per-query LUT a strided scalar dot per centroid —
    // measured ~25 us/query, which swamped the filter's savings. Transposed, the
    // LUT is dsub contiguous axpy passes over 256 floats: pure SIMD, ~5 us.
    std::vector<float>                 pqCenT;   // pqM × dsub × ksub
    std::vector<float>                 pqNorm;   // pqM × ksub  (||c||^2)

    // GPU beam-search path.
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLComputePipelineState> pipe     = nil;
    id<MTLBuffer>               dbBuf    = nil;   // db as fp16 (half), id order
    id<MTLBuffer>               graphBuf = nil;   // dbCount × R int32

    // Per-search buffers, kept alive across calls (see GpuScratch.h).
    struct Slot { enum { Query = 0, OutId, OutVal }; };
    detail::GpuScratch scratch;

    // The fp16 db, addressed from the CPU. dbBuf is StorageModeShared, so the
    // CPU traversal can score straight out of the GPU's copy — half the bytes
    // per row, zero extra memory. nil without a Metal device (CPU uses fp32).
    const __fp16* dbHalfCpu() const {
        return dbBuf ? static_cast<const __fp16*>([dbBuf contents]) : nullptr;
    }
};

GraphIndex::GraphIndex(int dim, Metric metric, int R)
    : mImpl(std::make_unique<Impl>()) {
    mImpl->dim = dim; mImpl->metric = metric;
    mImpl->R   = std::min(R, kMaxK - 1);   // R/2 forward => forward+1 must fit the GPU top-k

    mImpl->device = acquireMetalDevice();
    if (mImpl->device) {
        mImpl->queue = [mImpl->device newCommandQueue];
        mImpl->scratch.setDevice(mImpl->device);
        NSError* err = nil;
        id<MTLLibrary> lib = [mImpl->device newLibraryWithSource:kGraphShaderSrc
                                                         options:nil error:&err];
        if (!lib) {
            MFLAT_LOG_ERROR("graph shader compile failed: %s",
                            err ? [[err localizedDescription] UTF8String] : "?");
        } else {
            id<MTLFunction> fn = [lib newFunctionWithName:@"graph_search"];
            mImpl->pipe = [mImpl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!mImpl->pipe)
                MFLAT_LOG_ERROR("graph pipeline build failed: %s",
                                err ? [[err localizedDescription] UTF8String] : "?");
        }
    }
}
GraphIndex::~GraphIndex() = default;

bool GraphIndex::ready()  const { return mImpl->ready; }
int  GraphIndex::dim()    const { return mImpl->dim; }
int  GraphIndex::size()   const { return mImpl->dbCount; }
int  GraphIndex::degree() const { return mImpl->R; }

void GraphIndex::build(const float* vectors, int n, int nprobe) {
    if (n <= 0 || mImpl->dim <= 0) return;
    const int dim = mImpl->dim, R = mImpl->R;

    // Per-stage build timing (INFO) — build is the index's weak spot, so the
    // split between the GPU self-search and the CPU graph stages is worth
    // reporting rather than re-deriving with a profiler each time.
    using Clock = std::chrono::steady_clock;
    auto tick = Clock::now();
    auto lap  = [&tick] {
        const auto now = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - tick).count();
        tick = now;
        return ms;
    };

    mImpl->db.assign(vectors, vectors + static_cast<size_t>(n) * dim);
    if (mImpl->metric == Metric::Cosine) normalizeRows(mImpl->db, n, dim);
    const float* data = mImpl->db.data();
    const Metric metric = mImpl->metric;

    // --- 1. intermediate kNN via IVF self-search (reuse the GPU batched-kNN) --
    // K0 candidates per node, sorted nearest-first. The self-search is the slow
    // step; cache it (MFLAT_KNN_CACHE) so the pruning below can be tuned cheaply.
    // K0 candidates per node. Tempting to shrink (RNG pruning keeps only ~11 of
    // them as forward edges, and the self-search's per-thread top-k is 2*K0
    // registers, so K0 drives its occupancy) — but MEASURED, K0=32 costs real
    // recall: 0.904 vs 0.941 @L=32, 0.966 vs 0.984 @L=64 (SIFT 200k, exact
    // subset GT). The pruner needs the deep candidate list even though it keeps
    // few. Cut build cost via the coarse partition instead (see autoNlist).
    // MFLAT_GRAPH_K0 overrides.
    int K0 = kMaxK;
    if (const char* e = std::getenv("MFLAT_GRAPH_K0")) K0 = atoi(e);
    K0 = std::max(8, std::min(K0, kMaxK));
    std::vector<int32_t> knn;
    const char* kc = std::getenv("MFLAT_KNN_CACHE");
    if (!(kc && loadKnn(kc, knn, n, K0))) {
        IvfIndex ivf(dim, metric, autoNlist(n));
        ivf.build(data, n);
        const double bms = lap();
        knn = ivf.search(data, n, K0, nprobe).ids;    // N × K0, sorted by distance
        MFLAT_LOG_INFO("graph build: ivf train %.0f ms, knn self-search %.0f ms "
                       "(n=%d K0=%d nprobe=%d)", bms, lap(), n, K0, nprobe);
        if (kc) saveKnn(kc, knn, n, K0);
    }
    tick = Clock::now();

    // --- 2. rank-based RNG pruning (CAGRA/NSG) → diverse forward edges --------
    // Keep candidate b (nearest-first) only if no already-kept neighbour c is
    // closer to b than a is (score(c,b) > score(a,b)). This removes "detourable"
    // edges and yields a graph a greedy bounded-beam can actually descend — the
    // property a raw kNN graph lacks. Forward edges fill columns [0,fwdCount).
    mImpl->graph.assign(static_cast<size_t>(n) * R, -1);
    std::vector<int32_t> fwdCount(n, 0);
    parallelFor(n, [&](int a) {
        const int32_t* cand = &knn[static_cast<size_t>(a) * K0];
        const float*   va   = data + static_cast<size_t>(a) * dim;
        int32_t* row = &mImpl->graph[static_cast<size_t>(a) * R];
        int w = 0;
        for (int j = 0; j < K0 && w < R; ++j) {
            const int b = cand[j];
            if (b == a || b < 0) continue;             // drop self / pads
            const float* vb = data + static_cast<size_t>(b) * dim;
            const float sab = score(metric, va, vb, dim);
            bool occluded = false;
            for (int e = 0; e < w; ++e) {              // already-kept neighbours
                const float* vc = data + static_cast<size_t>(row[e]) * dim;
                if (score(metric, vc, vb, dim) > sab) { occluded = true; break; }
            }
            if (!occluded) row[w++] = b;
        }
        fwdCount[a] = w;
    });

    {   // forward-degree diagnostic
        const double pms = lap();
        long tot = 0; int mn = R, mx = 0;
        for (int i = 0; i < n; ++i) { tot += fwdCount[i]; mn = std::min(mn, (int)fwdCount[i]); mx = std::max(mx, (int)fwdCount[i]); }
        MFLAT_LOG_INFO("graph build: rng prune %.0f ms; forward degree mean=%.1f min=%d max=%d",
                       pms, (double)tot / n, mn, mx);
    }

    // --- 3. reverse edges into the leftover slots ----------------------------
    // If a -> b survived pruning, register b -> a in b's free columns
    // [fwdCount[b], R). RNG pruning leaves most nodes below R, so there is room;
    // reverse links let the search reach nodes no forward edge points at.
    const bool addReverse = !(std::getenv("MFLAT_GRAPH_NOREV"));
    std::vector<std::atomic<int>> revCursor(n);
    for (int v = 0; v < n; ++v) revCursor[v].store(0, std::memory_order_relaxed);
    if (addReverse)
    parallelFor(n, [&](int a) {
        const int32_t* row = &mImpl->graph[static_cast<size_t>(a) * R];
        for (int e = 0; e < fwdCount[a]; ++e) {
            const int b = row[e];
            const int room = R - fwdCount[b];
            if (room <= 0) continue;
            const int pos = revCursor[b].fetch_add(1, std::memory_order_relaxed);
            if (pos < room)
                mImpl->graph[static_cast<size_t>(b) * R + fwdCount[b] + pos] = a;
        }
    });
    // Duplicate edges (a mutual neighbour landing in both halves) are harmless —
    // the traversal's visited set / the GPU dedup collapse them.

    // --- entry medoid (nearest db vector to the global mean) ----------------
    std::vector<float> mean(dim, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float* v = data + static_cast<size_t>(i) * dim;
        for (int d = 0; d < dim; ++d) mean[d] += v[d];
    }
    for (int d = 0; d < dim; ++d) mean[d] /= static_cast<float>(n);
    int    medoid = 0;
    float  bestD  = INFINITY;
    for (int i = 0; i < n; ++i) {
        float dd = sqL2(mean.data(), data + static_cast<size_t>(i) * dim, dim);
        if (dd < bestD) { bestD = dd; medoid = i; }
    }
    mImpl->entry   = medoid;
    mImpl->dbCount = n;
    mImpl->ready   = true;
    const double rms = lap();

    uploadGpu(mImpl.get());
    const double ums = lap();

    buildFilter(mImpl.get(), data, n);
    MFLAT_LOG_INFO("graph build: reverse+medoid %.0f ms, gpu upload %.0f ms, "
                   "filter codes %.0f ms", rms, ums, lap());
}

// Train the traversal filter: PQ codes + each vector's quantization error.
// pqM targets dim/8 (16 bytes at dim=128 => one cache line per neighbour, and a
// ~16 MB table at n=1M, which the system-level cache can hold — the whole point).
// L2 / Cosine only: the bound is a statement about Euclidean distance, and Cosine
// arrives L2-normalized so its ranking is the same. Failure is silent and safe —
// searchCpu just scores full rows, as before.
void GraphIndex::buildFilter(Impl* m, const float* data, int n) {
    m->pq.reset();
    m->pqCodes.clear();
    m->pqErr.clear();
    m->pqM = 0;
    if (m->metric == Metric::InnerProduct) return;
    if (std::getenv("MFLAT_GRAPH_NOFILTER")) return;

    // Code size drives the bound's TIGHTNESS, which is the whole game: the filter
    // only pays when it rejects a large fraction of neighbours, and rejection is
    // governed by e_i = ||x - x̂||. Measured on SIFT1M (skip rate / net latency):
    //   dsub=8 (16 B):   1.2% skipped -> a LOSS (all LUT cost, no savings)
    //   dsub=4 (32 B):  13.8% skipped -> ~break-even
    //   dsub=2 (64 B):  53.9% skipped -> ~10% faster, and 64 B is one cache line
    // So target dsub=2 within a 64-byte code budget, and DON'T enable the filter
    // at all when dim is too large to reach a tight bound within that budget —
    // a loose filter is pure overhead. MFLAT_GRAPH_PQM overrides.
    const int dim = m->dim;
    int sub = std::min(64, std::max(1, dim / 2));
    if (const char* e = std::getenv("MFLAT_GRAPH_PQM")) sub = atoi(e);   // tuning hook
    while (sub > 1 && dim % sub != 0) --sub;
    if (sub < 2) return;
    const int dsubWant = dim / sub;
    if (dsubWant > 4 && !std::getenv("MFLAT_GRAPH_PQM")) {
        MFLAT_LOG_INFO("graph: traversal filter off (dim=%d needs dsub=%d > 4 "
                       "within a 64B code; the bound would be too loose to pay)",
                       dim, dsubWant);
        return;
    }

    // Subsample the codebook training (256 points/centroid, faiss-style): the
    // codebook barely moves, and the bound stays EXACT regardless because e_i is
    // measured against whatever codebook comes out. The final assignment pass
    // still encodes all n. Measured: 3.8 s -> well under 1 s of build.
    auto pq = std::make_unique<detail::PqTrainer>(dim, sub);
    std::vector<uint8_t> codes;
    pq->train(data, n, 10, &codes, /*maxPointsPerCentroid=*/256);
    if (!pq->trained() || codes.size() != static_cast<size_t>(n) * sub) return;

    // e_i = ||x_i - decode(code_i)||, the exact reconstruction error. The filter
    // is only sound because this is the TRUE per-vector error, not an estimate.
    const int dsub = pq->dsub(), ksub = pq->ksub();
    const std::vector<float>& cen = pq->centroids();
    std::vector<float> err(n, 0.0f);
    parallelFor(n, [&](int i) {
        const float* x = data + static_cast<size_t>(i) * dim;
        const uint8_t* c = &codes[static_cast<size_t>(i) * sub];
        double acc = 0.0;
        for (int mm = 0; mm < sub; ++mm) {
            const float* ce = &cen[(static_cast<size_t>(mm) * ksub + c[mm]) * dsub];
            const float* xs = x + static_cast<size_t>(mm) * dsub;
            for (int d = 0; d < dsub; ++d) {
                const double e = static_cast<double>(xs[d]) - ce[d];
                acc += e * e;
            }
        }
        err[i] = static_cast<float>(std::sqrt(acc));
    });

    // Transpose the codebook for the SIMD LUT build (see pqCenT).
    std::vector<float> cenT(static_cast<size_t>(sub) * dsub * ksub);
    for (int mm = 0; mm < sub; ++mm)
        for (int j = 0; j < ksub; ++j)
            for (int d = 0; d < dsub; ++d)
                cenT[(static_cast<size_t>(mm) * dsub + d) * ksub + j] =
                    cen[(static_cast<size_t>(mm) * ksub + j) * dsub + d];

    m->pqCenT  = std::move(cenT);
    m->pqNorm  = pq->norms();
    m->pq      = std::move(pq);
    m->pqCodes = std::move(codes);
    m->pqErr   = std::move(err);
    m->pqM     = sub;
}


// Cast the float db to fp16 and upload it + the int32 graph as GPU buffers.
void GraphIndex::uploadGpu(Impl* m) {
    if (!m->pipe) return;
    std::vector<__fp16> dbHalf(m->db.size());
    for (size_t i = 0; i < dbHalf.size(); ++i) dbHalf[i] = static_cast<__fp16>(m->db[i]);
    m->dbBuf = [m->device newBufferWithBytes:dbHalf.data()
                                      length:dbHalf.size() * sizeof(__fp16)
                                     options:MTLResourceStorageModeShared];
    m->graphBuf = [m->device newBufferWithBytes:m->graph.data()
                                         length:m->graph.size() * sizeof(int32_t)
                                        options:MTLResourceStorageModeShared];
}

bool GraphIndex::save(const char* path) const {
    if (!mImpl->ready) return false;
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    // 'MFG2' appends the traversal filter (pqM, codes, per-vector error, the
    // transposed codebook + norms). Without it a loaded index would search the
    // same but SLOWER than the one that was built — a silent perf cliff.
    // pqM = 0 => no filter section (InnerProduct / dim too large / train failed).
    const int32_t pqM = (mImpl->pq && !mImpl->pqCodes.empty()) ? mImpl->pqM : 0;
    const int32_t hdr[7] = { 0x4d464732 /*'MFG2'*/, mImpl->dim, (int)mImpl->metric,
                             mImpl->R, mImpl->dbCount, mImpl->entry, pqM };
    bool ok = std::fwrite(hdr, sizeof(int32_t), 7, f) == 7
            && std::fwrite(mImpl->db.data(),    sizeof(float),   mImpl->db.size(),    f) == mImpl->db.size()
            && std::fwrite(mImpl->graph.data(), sizeof(int32_t), mImpl->graph.size(), f) == mImpl->graph.size();
    if (ok && pqM > 0) {
        ok = std::fwrite(mImpl->pqCodes.data(), 1, mImpl->pqCodes.size(), f) == mImpl->pqCodes.size()
          && std::fwrite(mImpl->pqErr.data(),  sizeof(float), mImpl->pqErr.size(),  f) == mImpl->pqErr.size()
          && std::fwrite(mImpl->pqCenT.data(), sizeof(float), mImpl->pqCenT.size(), f) == mImpl->pqCenT.size()
          && std::fwrite(mImpl->pqNorm.data(), sizeof(float), mImpl->pqNorm.size(), f) == mImpl->pqNorm.size();
    }
    std::fclose(f);
    return ok;
}

bool GraphIndex::load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    int32_t hdr[6];
    if (std::fread(hdr, sizeof(int32_t), 6, f) != 6) { std::fclose(f); return false; }
    const bool v2 = (hdr[0] == 0x4d464732 /*'MFG2'*/);
    if (!v2 && hdr[0] != 0x4d464752 /*'MFGR' — pre-filter files still load*/) {
        std::fclose(f); return false;
    }
    int32_t pqM = 0;
    if (v2 && std::fread(&pqM, sizeof(int32_t), 1, f) != 1) { std::fclose(f); return false; }
    mImpl->dim = hdr[1]; mImpl->metric = (Metric)hdr[2]; mImpl->R = hdr[3];
    mImpl->dbCount = hdr[4]; mImpl->entry = hdr[5];
    mImpl->db.resize(static_cast<size_t>(mImpl->dbCount) * mImpl->dim);
    mImpl->graph.resize(static_cast<size_t>(mImpl->dbCount) * mImpl->R);
    bool ok = std::fread(mImpl->db.data(),    sizeof(float),   mImpl->db.size(),    f) == mImpl->db.size()
           && std::fread(mImpl->graph.data(), sizeof(int32_t), mImpl->graph.size(), f) == mImpl->graph.size();

    mImpl->pq.reset();
    mImpl->pqCodes.clear(); mImpl->pqErr.clear();
    mImpl->pqCenT.clear();  mImpl->pqNorm.clear();
    mImpl->pqM = 0;
    if (ok && pqM > 0 && mImpl->dim % pqM == 0) {
        auto pq = std::make_unique<detail::PqTrainer>(mImpl->dim, pqM);
        const size_t nC = static_cast<size_t>(mImpl->dbCount) * pqM;
        const size_t nT = static_cast<size_t>(pqM) * pq->dsub() * pq->ksub();
        const size_t nN = static_cast<size_t>(pqM) * pq->ksub();
        std::vector<uint8_t> codes(nC);
        std::vector<float>   err(mImpl->dbCount), cenT(nT), nrm(nN);
        ok = std::fread(codes.data(), 1, nC, f) == nC
          && std::fread(err.data(),  sizeof(float), err.size(), f) == err.size()
          && std::fread(cenT.data(), sizeof(float), nT, f) == nT
          && std::fread(nrm.data(),  sizeof(float), nN, f) == nN;
        if (ok) {
            mImpl->pq      = std::move(pq);
            mImpl->pqCodes = std::move(codes);
            mImpl->pqErr   = std::move(err);
            mImpl->pqCenT  = std::move(cenT);
            mImpl->pqNorm  = std::move(nrm);
            mImpl->pqM     = pqM;
        }
    }
    std::fclose(f);
    if (!ok) return false;
    mImpl->ready = true;
    uploadGpu(mImpl.get());
    return true;
}

// Filter effectiveness counters (MFLAT_GRAPH_FILTERSTATS=1 prints the keep rate
// at process exit) — a filter that rejects little is pure overhead, so this is
// the number that decides whether the whole idea pays.
std::atomic<long> gSeen{0}, gKept{0};
struct FilterStats {
    ~FilterStats() {
        if (!std::getenv("MFLAT_GRAPH_FILTERSTATS")) return;
        const long s = gSeen.load(), k = gKept.load();
        if (s) std::fprintf(stderr, "[filter] neighbours %ld, rows fetched %ld (%.1f%% kept, %.1f%% skipped)\n",
                            s, k, 100.0 * k / s, 100.0 * (s - k) / s);
    }
} gFilterStats;

// Open-addressing visited set (id+1 stored, 0 = empty; linear probing, grows at
// 50% load) — the CPU twin of the kernel's vhash. std::unordered_set was ~half
// the single-query latency budget (hash + node allocations per insert).
struct VisitedSet {
    std::vector<int32_t> slots;
    uint32_t mask  = 0;
    uint32_t count = 0;
    explicit VisitedSet(uint32_t cap) {
        uint32_t h = 64; while (h < cap) h <<= 1;
        slots.assign(h, 0); mask = h - 1;
    }
    bool insert(int id) {   // true if newly inserted
        if ((count + 1) * 2 > slots.size()) grow();
        uint32_t h = mix(static_cast<uint32_t>(id)) & mask;
        for (;;) {
            const int32_t s = slots[h];
            if (s == 0)      { slots[h] = id + 1; ++count; return true; }
            if (s == id + 1) return false;
            h = (h + 1) & mask;
        }
    }
    void grow() {
        std::vector<int32_t> old;
        old.swap(slots);
        slots.assign(old.size() * 2, 0);
        mask = static_cast<uint32_t>(slots.size()) - 1;
        for (const int32_t s : old)
            if (s) {
                uint32_t h = mix(static_cast<uint32_t>(s - 1)) & mask;
                while (slots[h]) h = (h + 1) & mask;
                slots[h] = s;
            }
    }
};

// 4-accumulator metric score, local to the graph traversal hot path. The
// Distance.h primitives accumulate serially by contract (bit-stable reference);
// here the serial float dependency chain IS the bottleneck (one FMA latency per
// element), and the graph CPU path is judged on recall, not bit-parity, so
// independent chains (which clang then vectorizes) are fair game.
static inline float score4(Metric metric, const float* a, const float* b, int dim) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int c = 0;
    if (metric == Metric::L2) {
        for (; c + 4 <= dim; c += 4) {
            const float e0 = a[c] - b[c],         e1 = a[c + 1] - b[c + 1];
            const float e2 = a[c + 2] - b[c + 2], e3 = a[c + 3] - b[c + 3];
            s0 += e0 * e0; s1 += e1 * e1; s2 += e2 * e2; s3 += e3 * e3;
        }
        float s = (s0 + s1) + (s2 + s3);
        for (; c < dim; ++c) { const float e = a[c] - b[c]; s += e * e; }
        return -s;
    }
    for (; c + 4 <= dim; c += 4) {
        s0 += a[c] * b[c];         s1 += a[c + 1] * b[c + 1];
        s2 += a[c + 2] * b[c + 2]; s3 += a[c + 3] * b[c + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; c < dim; ++c) s += a[c] * b[c];
    return s;
}

// Same, over the fp16 database. The graph traversal is MEMORY-bound — each hop
// gathers R random rows, so a query touches thousands of scattered rows and the
// single-core limit is bytes pulled, not FLOPs — and halving the row (fp32 ->
// fp16) halves exactly that traffic. Reads the fp16 rows the GPU path already
// stores (the shared MTLBuffer is CPU-addressable), so it costs no extra memory
// and no extra conversion pass. fp16 storage with fp32 accumulation is the same
// precision the GPU kernels use, and recall is unchanged (measured).
static inline float score4H(Metric metric, const float* a, const __fp16* b, int dim) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int c = 0;
    if (metric == Metric::L2) {
        for (; c + 4 <= dim; c += 4) {
            const float e0 = a[c]     - static_cast<float>(b[c]);
            const float e1 = a[c + 1] - static_cast<float>(b[c + 1]);
            const float e2 = a[c + 2] - static_cast<float>(b[c + 2]);
            const float e3 = a[c + 3] - static_cast<float>(b[c + 3]);
            s0 += e0 * e0; s1 += e1 * e1; s2 += e2 * e2; s3 += e3 * e3;
        }
        float s = (s0 + s1) + (s2 + s3);
        for (; c < dim; ++c) { const float e = a[c] - static_cast<float>(b[c]); s += e * e; }
        return -s;
    }
    for (; c + 4 <= dim; c += 4) {
        s0 += a[c]     * static_cast<float>(b[c]);
        s1 += a[c + 1] * static_cast<float>(b[c + 1]);
        s2 += a[c + 2] * static_cast<float>(b[c + 2]);
        s3 += a[c + 3] * static_cast<float>(b[c + 3]);
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; c < dim; ++c) s += a[c] * static_cast<float>(b[c]);
    return s;
}

// Dynamic best-first graph traversal (CPU) — the recall reference / fallback,
// and since the small-batch routing also the SINGLE-QUERY hot path. hnswlib-
// style discipline: below-threshold candidates are never pushed to the
// frontier (the threshold only rises, so they could never be expanded anyway
// — identical traversal, far fewer heap ops), the visited check runs over a
// parent's whole row before any scoring so the prefetches overlap the scores.
static SearchResult searchCpu(const float* db, const __fp16* dbH,
                              const int32_t* graph, int dim, int R,
                              int n, int entry, Metric metric,
                              const float* queries, int m, int k, int L, int numStart,
                              int W, int pqM, int ksub,
                              const uint8_t* codes, const float* err,
                              const float* cenT, const float* pqNorm) {
    SearchResult out;
    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k, emptyValue(metric));

    // LOSSLESS neighbour filter. The traversal is latency-bound on random row
    // gathers (~30 per hop, each 4 cache lines from DRAM), NOT on FLOPs or
    // bandwidth. ADC gives ||q - x̂||^2 exactly, and e_i = ||x_i - x̂_i|| is
    // stored, so  ||q-x|| >= ||q-x̂|| - e_i  is a HARD lower bound. If that bound
    // already loses to the beam's worst kept entry, the neighbour cannot make
    // the beam and its row is never touched — we paid one SLC-resident code
    // (16 B) instead of a 256 B DRAM row. Because the bound admits no false
    // negatives, the beam sees exactly the same insertions as the unfiltered
    // walk: identical results, identical recall, identical CPU/GPU parity.
    const bool useFilter = pqM > 0 && codes && err && cenT && pqNorm
                        && metric != Metric::InnerProduct;

    // Score against the fp16 rows when they exist (halves the gathered bytes,
    // which is what this traversal is actually limited by); fp32 otherwise.
    const size_t rowB = dbH ? sizeof(__fp16) : sizeof(float);
    auto rowAddr = [&](int id) -> const char* {
        return (dbH ? reinterpret_cast<const char*>(dbH) : reinterpret_cast<const char*>(db))
             + static_cast<size_t>(id) * dim * rowB;
    };
    auto scoreExact = [&](const float* q, int id) {
        return dbH ? score4H(metric, q, dbH + static_cast<size_t>(id) * dim, dim)
                   : score4(metric, q, db + static_cast<size_t>(id) * dim, dim);
    };


    parallelFor(m, [&](int qi) {
        const float* qsrc = queries + static_cast<size_t>(qi) * dim;
        std::vector<float> qn;
        const float* q = qsrc;
        if (metric == Metric::Cosine) { qn.assign(qsrc, qsrc + dim); normalizeRows(qn, 1, dim); q = qn.data(); }

        // Per-query ADC table (pqM × 256 floats ≈ 16 KB — L1/L2-resident) and
        // ||q||^2, which turns the table's sum into ||q - x̂||^2.
        std::vector<float> lut;
        float qNorm2 = 0.0f;
        if (useFilter) {
            const int dsub = dim / pqM;
            lut.resize(static_cast<size_t>(pqM) * ksub);
            // lut[mm][j] = ||c||^2 - 2 q_mm·c. With the transposed codebook this
            // is dsub contiguous axpy passes over the 256 centroids — vectorized,
            // vs a strided scalar dot per centroid (~5x faster, measured).
            for (int mm = 0; mm < pqM; ++mm) {
                float* o = &lut[static_cast<size_t>(mm) * ksub];
                const float* nrm = pqNorm + static_cast<size_t>(mm) * ksub;
                for (int j = 0; j < ksub; ++j) o[j] = nrm[j];
                const float* qs = q + static_cast<size_t>(mm) * dsub;
                for (int d = 0; d < dsub; ++d) {
                    const float a = -2.0f * qs[d];
                    const float* Ct = cenT + (static_cast<size_t>(mm) * dsub + d) * ksub;
                    for (int j = 0; j < ksub; ++j) o[j] += a * Ct[j];
                }
            }
            for (int c = 0; c < dim; ++c) qNorm2 += q[c] * q[c];
        }
        // Hard lower bound on the TRUE squared distance to `id`, or -1 when the
        // filter is off. Reads one code (SLC) + one float — never the full row.
        auto lowerBoundD2 = [&](int id) -> float {
            const uint8_t* c = codes + static_cast<size_t>(id) * pqM;
            float s = qNorm2;
            for (int mm = 0; mm < pqM; ++mm) s += lut[static_cast<size_t>(mm) * ksub + c[mm]];
            const float dHat = std::sqrt(std::max(0.0f, s));   // ||q - x̂|| (exact)
            const float lb   = dHat - err[id];                 // triangle inequality
            return lb > 0.0f ? lb * lb : 0.0f;
        };

        // ONE bounded, sorted beam instead of two heaps. The old shape kept a
        // result min-heap plus an unbounded frontier max-heap and pushed every
        // improving candidate to BOTH — but a candidate outside the top-L can
        // never be expanded (the threshold only rises), so those pushes were
        // pure overhead, and the frontier grew to thousands of entries whose
        // heap traffic dominated the traversal. A flat array of L (score, id)
        // held descending gives: expansion = the first unexpanded entry, the
        // cutoff = beam[L-1], and insertion = one binary search + memmove of a
        // few cache lines. Same traversal, same results, far less work.
        struct Beam { float s; int id; bool exp; };
        std::vector<Beam> beam;
        beam.reserve(static_cast<size_t>(L) + 1);
        VisitedSet visited(static_cast<uint32_t>(8 * L));

        auto push = [&](float s, int id) {
            const int sz = static_cast<int>(beam.size());
            if (sz >= L && s <= beam[L - 1].s) return;          // cannot make the beam
            int lo = 0, hi = sz;                                // descending by score
            while (lo < hi) { const int mid = (lo + hi) >> 1;
                             if (beam[mid].s > s) lo = mid + 1; else hi = mid; }
            if (sz < L) beam.insert(beam.begin() + lo, Beam{s, id, false});
            else {
                std::memmove(&beam[lo + 1], &beam[lo], sizeof(Beam) * (L - 1 - lo));
                beam[lo] = Beam{s, id, false};
            }
        };

        if (visited.insert(entry)) push(scoreExact(q, entry), entry);
        for (int s = 1; s < numStart; ++s) {
            const int id = static_cast<int>(mix(static_cast<uint32_t>(qi) * 2654435761u + s) % n);
            if (visited.insert(id)) push(scoreExact(q, id), id);
        }

        // Expand W beam nodes per iteration, not one. A single query's traversal
        // is a SERIAL chain of dependent hops — each hop must finish its random
        // row gathers before the next parent is known — so it is latency-bound,
        // not bandwidth- or FLOP-bound (measured: halving the row to fp16 bought
        // 14%, halving the degree bought nothing at iso-recall, removing the
        // heaps bought nothing). Expanding W parents together issues W*R
        // independent gathers at once, so the memory-level parallelism hides the
        // latency the chain cannot. It costs some extra distance computations
        // (nodes a strict best-first would have skipped) — a good trade exactly
        // while latency, not throughput, is the binding constraint.
        std::vector<int> parents;
        parents.reserve(static_cast<size_t>(W));
        std::vector<int> fresh;
        fresh.reserve(static_cast<size_t>(W) * R);
        for (;;) {
            parents.clear();
            for (int i = 0; i < static_cast<int>(beam.size()) && static_cast<int>(parents.size()) < W; ++i)
                if (!beam[i].exp) { beam[i].exp = true; parents.push_back(beam[i].id); }
            if (parents.empty()) break;                         // beam fully expanded

            // Pass 1: collect this hop's unvisited neighbours and prefetch what
            // the NEXT pass will read — the codes (one line each) when filtering,
            // else the rows straight away.
            fresh.clear();
            for (const int parent : parents) {
                const int32_t* row = graph + static_cast<size_t>(parent) * R;
                for (int e = 0; e < R; ++e) {
                    const int id = row[e];
                    if (id >= 0 && visited.insert(id)) {
                        fresh.push_back(id);
                        if (useFilter) {
                            __builtin_prefetch(codes + static_cast<size_t>(id) * pqM);
                            __builtin_prefetch(err + id);
                        } else {
                            // Row is 1-2 cache lines; issue both.
                            const char* p = rowAddr(id);
                            __builtin_prefetch(p);
                            __builtin_prefetch(p + 64);
                        }
                    }
                }
            }

            // Pass 2 (filtered): drop the neighbours that PROVABLY cannot make
            // the beam, and prefetch full rows only for the survivors — so the
            // expensive DRAM gathers are issued only where they can matter.
            if (useFilter && static_cast<int>(beam.size()) >= L) {
                const float worstD2 = -beam[L - 1].s;   // beam holds -d^2
                int w = 0;
                for (const int id : fresh)
                    if (lowerBoundD2(id) <= worstD2) {
                        fresh[w++] = id;
                        const char* p = rowAddr(id);
                        __builtin_prefetch(p);
                        __builtin_prefetch(p + 64);
                    }
                gSeen.fetch_add(fresh.size(), std::memory_order_relaxed);
                gKept.fetch_add(w, std::memory_order_relaxed);
                fresh.resize(w);
            }

            // Pass 3: exact score for everything that survived. The beam only
            // ever holds exact distances, so the walk is the same walk.
            for (const int id : fresh) {
                const float s = scoreExact(q, id);
                if (static_cast<int>(beam.size()) >= L && s <= beam[L - 1].s) continue;
                push(s, id);
            }
        }

        const int kk = std::min<int>(k, static_cast<int>(beam.size()));
        for (int i = 0; i < kk; ++i) {
            out.ids[static_cast<size_t>(qi) * k + i]       = beam[i].id;
            out.distances[static_cast<size_t>(qi) * k + i] = scoreToValue(metric, beam[i].s);
        }
    });
    return out;
}

// CPU mirror of the GPU kernel's EXACT algorithm (bounded beam L, expand the
// single best unexpanded node, fixed maxIter, same hashed seeds, dedup). Used to
// prove the kernel is correct and to isolate graph-quality from kernel bugs — its
// recall should match the GPU path. (Distinct from the dynamic best-first
// searchCpu reference, which has an unbounded frontier.)
static SearchResult searchCpuBeam(const float* db, const int32_t* graph, int dim, int R,
                                  int n, int entry, Metric metric, const float* queries,
                                  int m, int k, int L, int maxIter, int numStart) {
    SearchResult out;
    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k, emptyValue(metric));

    parallelFor(m, [&](int qi) {
        const float* qsrc = queries + static_cast<size_t>(qi) * dim;
        std::vector<float> qn;
        const float* q = qsrc;
        if (metric == Metric::Cosine) { qn.assign(qsrc, qsrc + dim); normalizeRows(qn, 1, dim); q = qn.data(); }

        struct Node { float s; int id; };
        std::vector<Node> beam(L, {-INFINITY, -1});
        for (int s = 0; s < numStart && s < L; ++s) {
            int id = (s == 0) ? entry : (int)(mix((uint32_t)qi * 2654435761u + s) % n);
            beam[s] = { score(metric, q, db + (size_t)id * dim, dim), id };
        }
        auto bySc = [](const Node& a, const Node& b){ return a.s > b.s; };
        std::sort(beam.begin(), beam.end(), bySc);

        // PERSISTENT visited set: each node is expanded at most once, so the
        // beam can't cycle on a fixed set (DiskANN/NSG greedy search).
        std::unordered_set<int> visited; visited.reserve(static_cast<size_t>(maxIter) * 2);
        std::vector<Node> scratch; scratch.reserve(L + R);
        for (int it = 0; it < maxIter; ++it) {
            int parent = -1;
            for (int i = 0; i < L; ++i) if (beam[i].id >= 0 && !visited.count(beam[i].id)) { parent = beam[i].id; break; }
            if (parent < 0) break;
            visited.insert(parent);
            scratch.assign(beam.begin(), beam.end());
            const int32_t* row = graph + (size_t)parent * R;
            for (int e = 0; e < R; ++e) { int nb = row[e]; if (nb >= 0)
                scratch.push_back({ score(metric, q, db + (size_t)nb * dim, dim), nb }); }
            std::sort(scratch.begin(), scratch.end(), bySc);
            // dedup by id, keep the L best distinct
            std::unordered_set<int> seen; seen.reserve(L * 2);
            int w = 0;
            for (auto& nd : scratch) { if (w >= L) break; if (nd.id < 0) continue;
                if (seen.insert(nd.id).second) beam[w++] = nd; }
            for (; w < L; ++w) beam[w] = {-INFINITY, -1};
        }
        const int kk = std::min(k, L);
        for (int i = 0; i < kk; ++i) {
            out.ids[(size_t)qi * k + i]       = beam[i].id;
            out.distances[(size_t)qi * k + i] = scoreToValue(metric, beam[i].s);
        }
    });
    return out;
}

SearchResult GraphIndex::search(const float* queries, int m, int k, int L, int maxIter, int numStart, int searchWidth) {
    SearchResult out;
    if (m <= 0 || !mImpl->ready) return out;
    const int dim = mImpl->dim, R = mImpl->R, n = mImpl->dbCount, entry = mImpl->entry;
    const Metric metric = mImpl->metric;
    if (k < 1) k = 1;
    if (L < k) L = k;
    if (numStart < 1) numStart = 1;
    if (numStart > L) numStart = L;
    const int W = std::max(1, std::min(searchWidth, 16));   // parents expanded per iteration
    // Total node expansions is ~independent of W; expanding W per iteration just
    // does W* fewer (bigger) iterations. Size maxIter/H off the total, not W.
    const int totalExp = 2 * L + 100;
    if (maxIter <= 0) maxIter = (totalExp + W - 1) / W;      // generous; early-break stops at convergence

    // Diagnostic: CPU mirror of the exact GPU kernel algorithm (bounded beam).
    if (std::getenv("MFLAT_GRAPH_CPUBEAM"))
        return searchCpuBeam(mImpl->db.data(), mImpl->graph.data(), dim, R, n, entry,
                             metric, queries, m, k, L, W * maxIter, numStart);

    // Threadgroup budget: all scratch is fixed-size (independent of tgs). The
    // visited hash is sized ~2x the total expansions so it never saturates.
    // Fall back to the CPU reference if the GPU path is unavailable / too big.
    const uint32_t Lc = nextPow2(static_cast<uint32_t>(L + W * R));
    const uint32_t H  = std::min<uint32_t>(nextPow2(static_cast<uint32_t>(2 * (W * maxIter + numStart))), 8192u);
    const size_t tgMem = static_cast<size_t>(dim) * 4              // qsh
                       + static_cast<size_t>(L) * (4 + 4)          // lScore/lId
                       + static_cast<size_t>(Lc) * (4 + 4)         // mScore/mId
                       + static_cast<size_t>(H) * 4;               // vhash
    // The cooperative bitonic sort requires one thread per scratch slot (Lc);
    // running it strided (tgs < Lc) desyncs score/id and corrupts results, so the
    // GPU path needs a threadgroup of at least Lc threads. Fall back to CPU when
    // Lc exceeds the device's max threadgroup, or the pipeline is unavailable.
    //
    // CPU-vs-GPU routing, and the crossover depends on BOTH m and L.
    //
    //   m: the kernel runs ONE threadgroup per query, so a small batch cannot
    //      fill the GPU (a single query is round-trip-bound).
    //   L: the kernel re-sorts its whole beam every iteration — a cooperative
    //      bitonic over Lc = nextPow2(L + W*R) entries — so its per-iteration
    //      cost grows with L, while the CPU walk inserts into a bounded beam in
    //      O(log L) and prunes neighbours with the lossless filter (which the
    //      kernel does NOT have). Past L~64 the CPU wins at EVERY batch size.
    //
    // Measured on SIFT1M (temp/gx.mm, gpu ms vs cpu ms, R=63, W=1):
    //   L=32:  m=256  3.0 / 4.2  -> GPU     ; m=8192  75.8 / 137.3 -> GPU
    //   L=64:  m=256  9.4 / 7.2  -> CPU     ; m=512   15.8 / 16.8  -> GPU
    //   L=128: m=1000 88.9 / 57.6 -> CPU    ; m=8192 723.2 / 375.2 -> CPU
    //
    // These cutoffs are a curve fit to THIS machine, which is a weakness (see
    // the note in here.txt). The real fix is the kernel: give it the filter and
    // stop re-sorting the beam every iteration, and the GPU should take L=128
    // back. MFLAT_GRAPH_CPU=1/0 forces the CPU/GPU path.
    const int gpuMinBatch = (L <= 32) ? 256
                          : (L <= 64) ? 512
                          : INT_MAX;          // L > 64: the CPU wins outright
    bool cpuRoute = (m < gpuMinBatch);
    if (const char* e = std::getenv("MFLAT_GRAPH_CPU")) cpuRoute = (e[0] != '0');
    const NSUInteger maxT = mImpl->pipe ? mImpl->pipe.maxTotalThreadsPerThreadgroup : 0;
    const bool gpu = !cpuRoute && mImpl->pipe && mImpl->dbBuf && k <= kMaxK
                   && tgMem < 30000 && Lc <= maxT;
    if (!gpu)
        return searchCpu(mImpl->db.data(), mImpl->dbHalfCpu(), mImpl->graph.data(),
                         dim, R, n, entry, metric, queries, m, k, L, numStart, W,
                         mImpl->pqM, mImpl->pq ? mImpl->pq->ksub() : 256,
                         mImpl->pqCodes.empty() ? nullptr : mImpl->pqCodes.data(),
                         mImpl->pqErr.empty()   ? nullptr : mImpl->pqErr.data(),
                         mImpl->pqCenT.empty()  ? nullptr : mImpl->pqCenT.data(),
                         mImpl->pqNorm.empty()  ? nullptr : mImpl->pqNorm.data());

    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k, emptyValue(metric));

    std::vector<float> qNorm;
    const float* qPtr = queries;
    if (metric == Metric::Cosine) {
        qNorm.assign(queries, queries + static_cast<size_t>(m) * dim);
        normalizeRows(qNorm, m, dim);
        qPtr = qNorm.data();
    }

    @autoreleasepool {
        // Persistent scratch, not a fresh allocation per call (see GpuScratch.h).
        id<MTLBuffer> qBuf = mImpl->scratch.upload(
            Impl::Slot::Query, qPtr, static_cast<size_t>(m) * dim * sizeof(float));
        id<MTLBuffer> outIdBuf = mImpl->scratch.ensure(
            Impl::Slot::OutId, static_cast<size_t>(m) * k * sizeof(int32_t));
        id<MTLBuffer> outValBuf = mImpl->scratch.ensure(
            Impl::Slot::OutVal, static_cast<size_t>(m) * k * sizeof(float));

        GraphParams p;
        p.dim = (uint32_t)dim; p.k = (uint32_t)k; p.L = (uint32_t)L; p.R = (uint32_t)R;
        p.Lc = Lc; p.numSeeds = (uint32_t)numStart; p.maxIter = (uint32_t)maxIter;
        p.metric = (uint32_t)metric; p.queryCount = (uint32_t)m;
        p.dbCount = (uint32_t)n; p.entry = (uint32_t)entry; p.H = H; p.W = (uint32_t)W;

        id<MTLCommandBuffer>         cb  = [mImpl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mImpl->pipe];
        [enc setBuffer:mImpl->dbBuf    offset:0 atIndex:0];
        [enc setBuffer:mImpl->graphBuf offset:0 atIndex:1];
        [enc setBuffer:qBuf            offset:0 atIndex:2];
        [enc setBuffer:outIdBuf        offset:0 atIndex:3];
        [enc setBuffer:outValBuf       offset:0 atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];

        // One threadgroup per query, exactly Lc threads (>= gated above) so the
        // cooperative bitonic sort is one-thread-per-slot (no strided desync).
        NSUInteger tg = Lc;

        [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(dim) * sizeof(float) atIndex:0];
        [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(L)  * sizeof(float)  atIndex:1];
        [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(L)  * sizeof(int)    atIndex:2];
        [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(Lc) * sizeof(float)  atIndex:3];
        [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(Lc) * sizeof(int)    atIndex:4];
        [enc setThreadgroupMemoryLength:static_cast<NSUInteger>(H)  * sizeof(int)    atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(m), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(out.ids.data(),       [outIdBuf contents],  out.ids.size() * sizeof(int32_t));
        std::memcpy(out.distances.data(), [outValBuf contents], out.distances.size() * sizeof(float));
    }
    return out;
}

}  // namespace mflat
