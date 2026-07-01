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
#include <cstdio>
#include <cstring>
#include <queue>
#include <unordered_set>
#include <vector>

#include "metalflat/GraphIndex.h"
#include "metalflat/IvfIndex.h"
#include "Internal.h"        // detail::parallelFor, normalizeRows, sqL2, score, acquireMetalDevice
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

inline int autoNlist(int n) {
    int s = static_cast<int>(std::sqrt(static_cast<double>(n)));
    return std::max(64, s);
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

    // GPU beam-search path.
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLComputePipelineState> pipe     = nil;
    id<MTLBuffer>               dbBuf    = nil;   // db as fp16 (half), id order
    id<MTLBuffer>               graphBuf = nil;   // dbCount × R int32
};

GraphIndex::GraphIndex(int dim, Metric metric, int R)
    : mImpl(std::make_unique<Impl>()) {
    mImpl->dim = dim; mImpl->metric = metric;
    mImpl->R   = std::min(R, kMaxK - 1);   // R/2 forward => forward+1 must fit the GPU top-k

    mImpl->device = acquireMetalDevice();
    if (mImpl->device) {
        mImpl->queue = [mImpl->device newCommandQueue];
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

    mImpl->db.assign(vectors, vectors + static_cast<size_t>(n) * dim);
    if (mImpl->metric == Metric::Cosine) normalizeRows(mImpl->db, n, dim);
    const float* data = mImpl->db.data();
    const Metric metric = mImpl->metric;

    // --- 1. intermediate kNN via IVF self-search (reuse the GPU batched-kNN) --
    // K0 candidates per node, sorted nearest-first. The self-search is the slow
    // step; cache it (MFLAT_KNN_CACHE) so the pruning below can be tuned cheaply.
    const int K0 = std::min(kMaxK, 64);
    std::vector<int32_t> knn;
    const char* kc = std::getenv("MFLAT_KNN_CACHE");
    if (!(kc && loadKnn(kc, knn, n, K0))) {
        IvfIndex ivf(dim, metric, autoNlist(n));
        ivf.build(data, n);
        knn = ivf.search(data, n, K0, nprobe).ids;    // N × K0, sorted by distance
        if (kc) saveKnn(kc, knn, n, K0);
    }

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
        long tot = 0; int mn = R, mx = 0;
        for (int i = 0; i < n; ++i) { tot += fwdCount[i]; mn = std::min(mn, (int)fwdCount[i]); mx = std::max(mx, (int)fwdCount[i]); }
        MFLAT_LOG_INFO("graph pruned forward degree: mean=%.1f min=%d max=%d",
                       (double)tot / n, mn, mx);
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

    uploadGpu(mImpl.get());
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
    const int32_t hdr[6] = { 0x4d464752 /*'MFGR'*/, mImpl->dim, (int)mImpl->metric,
                             mImpl->R, mImpl->dbCount, mImpl->entry };
    bool ok = std::fwrite(hdr, sizeof(int32_t), 6, f) == 6
            && std::fwrite(mImpl->db.data(),    sizeof(float),   mImpl->db.size(),    f) == mImpl->db.size()
            && std::fwrite(mImpl->graph.data(), sizeof(int32_t), mImpl->graph.size(), f) == mImpl->graph.size();
    std::fclose(f);
    return ok;
}

bool GraphIndex::load(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    int32_t hdr[6];
    if (std::fread(hdr, sizeof(int32_t), 6, f) != 6 || hdr[0] != 0x4d464752) { std::fclose(f); return false; }
    mImpl->dim = hdr[1]; mImpl->metric = (Metric)hdr[2]; mImpl->R = hdr[3];
    mImpl->dbCount = hdr[4]; mImpl->entry = hdr[5];
    mImpl->db.resize(static_cast<size_t>(mImpl->dbCount) * mImpl->dim);
    mImpl->graph.resize(static_cast<size_t>(mImpl->dbCount) * mImpl->R);
    bool ok = std::fread(mImpl->db.data(),    sizeof(float),   mImpl->db.size(),    f) == mImpl->db.size()
           && std::fread(mImpl->graph.data(), sizeof(int32_t), mImpl->graph.size(), f) == mImpl->graph.size();
    std::fclose(f);
    if (!ok) return false;
    mImpl->ready = true;
    uploadGpu(mImpl.get());
    return true;
}

// Dynamic best-first graph traversal (CPU) — the recall reference / fallback.
static SearchResult searchCpu(const float* db, const int32_t* graph, int dim, int R,
                              int n, int entry, Metric metric,
                              const float* queries, int m, int k, int L, int numStart) {
    SearchResult out;
    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k, emptyValue(metric));

    parallelFor(m, [&](int qi) {
        const float* qsrc = queries + static_cast<size_t>(qi) * dim;
        std::vector<float> qn;
        const float* q = qsrc;
        if (metric == Metric::Cosine) { qn.assign(qsrc, qsrc + dim); normalizeRows(qn, 1, dim); q = qn.data(); }

        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> result;   // min-heap: worst on top
        std::priority_queue<std::pair<float,int>> frontier;             // max-heap: best on top
        std::unordered_set<int> visited;
        visited.reserve(static_cast<size_t>(L) * 4);

        auto consider = [&](int id) {
            if (id < 0 || !visited.insert(id).second) return;
            const float s = score(metric, q, db + static_cast<size_t>(id) * dim, dim);
            frontier.push({s, id});
            if (static_cast<int>(result.size()) < L || s > result.top().first) {
                result.push({s, id});
                if (static_cast<int>(result.size()) > L) result.pop();
            }
        };

        consider(entry);
        for (int s = 1; s < numStart; ++s)
            consider(static_cast<int>(mix(static_cast<uint32_t>(qi) * 2654435761u + s) % n));

        while (!frontier.empty()) {
            auto [cs, cid] = frontier.top(); frontier.pop();
            if (static_cast<int>(result.size()) >= L && cs < result.top().first) break;
            const int32_t* row = graph + static_cast<size_t>(cid) * R;
            for (int e = 0; e < R; ++e) consider(row[e]);
        }

        std::vector<std::pair<float,int>> best;
        best.reserve(result.size());
        while (!result.empty()) { best.push_back(result.top()); result.pop(); }
        std::sort(best.begin(), best.end(), [](const auto& a, const auto& b){ return a.first > b.first; });
        const int kk = std::min<int>(k, static_cast<int>(best.size()));
        for (int i = 0; i < kk; ++i) {
            out.ids[static_cast<size_t>(qi) * k + i]       = best[i].second;
            out.distances[static_cast<size_t>(qi) * k + i] = scoreToValue(metric, best[i].first);
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
    const bool forceCpu = std::getenv("MFLAT_GRAPH_CPU") != nullptr;
    const NSUInteger maxT = mImpl->pipe ? mImpl->pipe.maxTotalThreadsPerThreadgroup : 0;
    const bool gpu = !forceCpu && mImpl->pipe && mImpl->dbBuf && k <= kMaxK
                   && tgMem < 30000 && Lc <= maxT;
    if (!gpu)
        return searchCpu(mImpl->db.data(), mImpl->graph.data(), dim, R, n, entry,
                         metric, queries, m, k, L, numStart);

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
        id<MTLBuffer> qBuf = [mImpl->device
            newBufferWithBytes:qPtr length:static_cast<size_t>(m) * dim * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> outIdBuf = [mImpl->device
            newBufferWithLength:static_cast<size_t>(m) * k * sizeof(int32_t)
                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> outValBuf = [mImpl->device
            newBufferWithLength:static_cast<size_t>(m) * k * sizeof(float)
                        options:MTLResourceStorageModeShared];

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
