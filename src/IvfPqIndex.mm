// SPDX-License-Identifier: Apache-2.0
// MetalFlat — IvfPqIndex implementation (IVF + Product Quantization).
//
// Build:
//   1. coarse k-means -> nlist centroids; assign each vector to a cell.
//   2. PQ training (NON-residual in v1): split each vector into m sub-vectors
//      of width dsub=dim/m; k-means each sub-space into 256 centroids.
//   3. encode every vector to m bytes (one nearest-sub-centroid id per
//      sub-space); reorder the codes into a per-cell CSR layout.
//
// Search (ADC, one threadgroup per query):
//   coarse — nprobe nearest cells (reuses FlatIndex over the centroids).
//   fine   — build the per-query lookup table once (lut[m][j] = ||pqc||^2 -
//            2 q_m·pqc[m][j], i.e. ||q_m - pqc[m][j]||^2 up to the per-query
//            constant ||q_m||^2). Because we PQ the vector (not a per-cell
//            residual), this LUT is cell-independent -> built ONCE per query.
//            The LUT lives in threadgroup memory when it fits, else in a
//            device scratch (large m_sub). Each candidate's distance is m
//            table lookups + adds (word-vectorized when m % 4 == 0); per-
//            thread register top-k, then a simdgroup-shuffle + tree
//            reduction (scratch is (tgs/32)*kk entries, so k never crushes
//            the threadgroup size). kRun > 16 routes to the multithreaded
//            CPU ADC, which is measured faster there.
//
// k-means uses FlatIndex for the assignment step (GEMM on GPU, exact CPU
// fallback when no device), so build works with or without Metal. The fine
// scan has a CPU fallback (searchCpu) that is also the recall reference.

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

#include "metalflat/IvfPqIndex.h"
#include "metalflat/FlatIndex.h"
#include "Internal.h"
#include "Log_internal.h"
#include "PqTrainer.h"
#include "CoarseQuantizer.h"
#include "Opq.h"

namespace mflat {

using namespace detail;   // parallelFor, normalizeRows, kmeansGpu, ...

namespace {

constexpr int kKsub = 256;   // centroids per sub-quantizer (8-bit codes)

// CPU-side mirror of the kernel's PqParams, field-for-field.
struct PqParams {
    uint32_t dim;
    uint32_t k;
    uint32_t nprobe;
    uint32_t m;
    uint32_t dsub;
    uint32_t queryCount;
    uint32_t residual;   // 1 = residual ADC (precomp + coarse terms active)
    uint32_t tgLut;      // 1 = LUT in threadgroup memory, 0 = device scratch
    uint32_t qBase;      // first query of this dispatch (device-LUT chunking)
};

NSString* const kShaderSrc = @R"(
#include <metal_stdlib>
using namespace metal;

struct PqParams { uint dim; uint k; uint nprobe; uint m; uint dsub; uint queryCount; uint residual; uint tgLut; uint qBase; };
constant uint kMaxK = 64;
constant uint kKsub = 256;

// ---- top-k reduction helpers (same scheme as FlatIndex's kernels) --------
// Per-thread lists are ASCENDING over kk = nextPow2(k) slots (index 0 =
// worst kept; unfilled slots hold -INF/-1). Power-of-two length lets two
// lists merge via the bitonic trick: c[i] = max(a[i], b[kk-1-i]) holds the
// kk largest of the union and is bitonic, so a log2(kk)-stage bitonic merge
// re-sorts it. Reduction = simdgroup shuffle butterfly (register-level, no
// scratch) + a cross-simdgroup tree merge over only (tgs/32) lists — so k
// no longer constrains the threadgroup size (the old tgs*k tree-merge
// scratch forced tgs=32 at k=64, leaving the GPU ~90% idle).

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

// ---- ADC candidate distance ----------------------------------------------
// M table lookups (+ the per-cell residual table when res). Word-vectorized
// when M % 4 == 0 (each code row is then 4-byte aligned: row stride M).
// Two variants because the LUT may live in threadgroup or device memory.
inline float adcDistTg(device const uchar* code, uint M, bool res,
                       threadgroup const float* lut, device const float* pc,
                       float base) {
    float dist = base;
    if ((M & 3u) == 0u) {
        device const uint* w = (device const uint*)code;
        for (uint i = 0; i < (M >> 2); ++i) {
            const uint c4 = w[i];
            const uint o  = (i << 2) * kKsub;
            const uint b0 = o + (c4 & 0xffu);
            const uint b1 = o + kKsub + ((c4 >> 8) & 0xffu);
            const uint b2 = o + 2u * kKsub + ((c4 >> 16) & 0xffu);
            const uint b3 = o + 3u * kKsub + (c4 >> 24);
            dist += lut[b0] + lut[b1] + lut[b2] + lut[b3];
            if (res) dist += pc[b0] + pc[b1] + pc[b2] + pc[b3];
        }
    } else {
        for (uint mm = 0; mm < M; ++mm) {
            const uint o = mm * kKsub + code[mm];
            dist += lut[o];
            if (res) dist += pc[o];
        }
    }
    return dist;
}
inline float adcDistDev(device const uchar* code, uint M, bool res,
                        device const float* lut, device const float* pc,
                        float base) {
    float dist = base;
    if ((M & 3u) == 0u) {
        device const uint* w = (device const uint*)code;
        for (uint i = 0; i < (M >> 2); ++i) {
            const uint c4 = w[i];
            const uint o  = (i << 2) * kKsub;
            const uint b0 = o + (c4 & 0xffu);
            const uint b1 = o + kKsub + ((c4 >> 8) & 0xffu);
            const uint b2 = o + 2u * kKsub + ((c4 >> 16) & 0xffu);
            const uint b3 = o + 3u * kKsub + (c4 >> 24);
            dist += lut[b0] + lut[b1] + lut[b2] + lut[b3];
            if (res) dist += pc[b0] + pc[b1] + pc[b2] + pc[b3];
        }
    } else {
        for (uint mm = 0; mm < M; ++mm) {
            const uint o = mm * kKsub + code[mm];
            dist += lut[o];
            if (res) dist += pc[o];
        }
    }
    return dist;
}

// One threadgroup per query. Build the per-query ADC table once, then scan
// the probed cells' codes by table lookup, keeping a top-k per thread, then
// reduce. Distances rank by score = -dist (larger = nearer).
//
// Non-residual: dist = sum_mm lut[mm][code], lut = ||pqc||^2 - 2 q·pqc
//               (cell-independent; reported value adds ||q||^2).
// Residual (x ~ c + r): dist = ||q-c||^2 + sum_mm (lut[mm][code]
//               + precomp[cell][mm][code]) with precomp = 2 c·pqc — the
//               classic decomposition that keeps the query LUT byte-identical
//               to the non-residual one (built once per query, not per probe).
//               ||q-c||^2 comes from the host in exact fp32 (coarseVals).
//
// LUT placement: threadgroup memory when it fits next to the reduction
// scratch (p.tgLut). Larger code sizes (m_sub >= ~28) spill the LUT to a
// per-dispatch device scratch (lutSpill, one lutN slice per threadgroup);
// the host then chunks the query batch so the scratch stays bounded.
kernel void ivfpq_adc(
    device const uchar* codes      [[buffer(0)]],   // dbCount × m
    device const int*   ids        [[buffer(1)]],   // dbCount
    device const int*   cellStart  [[buffer(2)]],   // nlist+1
    device const float* queries    [[buffer(3)]],   // queryCount × dim
    device const int*   probed     [[buffer(4)]],   // queryCount × nprobe
    device const float* pqc        [[buffer(5)]],   // m × 256 × dsub
    device const float* pqNorm     [[buffer(6)]],   // m × 256  (||pqc||^2)
    device int*         outIds     [[buffer(7)]],
    device float*       outVal     [[buffer(8)]],
    constant PqParams&  p          [[buffer(9)]],
    device const float* precomp    [[buffer(10)]],  // nlist × m × 256 (residual)
    device const float* coarseVals [[buffer(11)]],  // queryCount × nprobe (residual)
    device float*       lutSpill   [[buffer(12)]],  // chunk × m × 256 (device-LUT mode)
    threadgroup float*  lutTg      [[threadgroup(0)]],   // m × 256 (tgLut mode)
    threadgroup float*  redScore   [[threadgroup(1)]],   // (tgs/32) × kk
    threadgroup int*    redId      [[threadgroup(2)]],
    uint tgpos [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tgs   [[threads_per_threadgroup]],
    uint sgid  [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint qi = p.qBase + tgpos;
    if (qi >= p.queryCount) return;
    const uint M = p.m, dsub = p.dsub, dim = p.dim;
    const uint k = min(p.k, kMaxK);
    uint kk = 1; while (kk < k) kk <<= 1;
    const bool res   = (p.residual != 0u);
    const bool useTg = (p.tgLut != 0u);
    device const float* q = queries + (uint64_t)qi * dim;

    // Per-query LUT (cell-independent and BYTE-IDENTICAL in both modes):
    // lut[mm*256 + j] = ||pqc[mm][j]||^2 - 2 q_mm·pqc[mm][j].
    const uint lutN = M * kKsub;
    device float* dl = lutSpill + (uint64_t)tgpos * lutN;   // only touched if !useTg
    for (uint e = tid; e < lutN; e += tgs) {
        const uint mm = e / kKsub;
        const uint j  = e % kKsub;
        device const float* sub = pqc + ((uint64_t)mm * kKsub + j) * dsub;
        device const float* qm  = q + (uint64_t)mm * dsub;
        float dot = 0.0f;
        for (uint d = 0; d < dsub; ++d) dot += qm[d] * sub[d];
        const float v = pqNorm[e] - 2.0f * dot;
        if (useTg) lutTg[e] = v; else dl[e] = v;
    }
    if (useTg) threadgroup_barrier(mem_flags::mem_threadgroup);
    else       threadgroup_barrier(mem_flags::mem_device);

    float bestScore[kMaxK];
    int   bestId[kMaxK];
    for (uint i = 0; i < kk; ++i) { bestScore[i] = -INFINITY; bestId[i] = -1; }
    float worst = -INFINITY;

    device const int* myCells = probed + (uint64_t)qi * p.nprobe;
    for (uint pp = 0; pp < p.nprobe; ++pp) {
        const int cell = myCells[pp];
        if (cell < 0) continue;
        const int lo = cellStart[cell];
        const int hi = cellStart[cell + 1];
        device const float* pc = precomp + (uint64_t)cell * M * kKsub;
        const float base = res ? coarseVals[(uint64_t)qi * p.nprobe + pp] : 0.0f;
        if (useTg) {
            for (int c = lo + (int)tid; c < hi; c += (int)tgs) {
                const float score = -adcDistTg(codes + (uint64_t)c * M, M, res,
                                               lutTg, pc, base);
                if (score > worst) {
                    insertTopk(bestScore, bestId, kk, score, ids[c]);
                    worst = bestScore[0];
                }
            }
        } else {
            for (int c = lo + (int)tid; c < hi; c += (int)tgs) {
                const float score = -adcDistDev(codes + (uint64_t)c * M, M, res,
                                                dl, pc, base);
                if (score > worst) {
                    insertTopk(bestScore, bestId, kk, score, ids[c]);
                    worst = bestScore[0];
                }
            }
        }
    }

    reduceTopkTg(bestScore, bestId, kk, redScore, redId, tid, tgs, sgid, lane);

    if (tid == 0u) {
        // Residual dist already IS ~||q-x||^2 (coarse term included); the
        // non-residual LUT omits the per-query ||q||^2 constant, so add it back.
        float qn = 0.0f;
        if (!res) for (uint d = 0; d < dim; ++d) qn += q[d] * q[d];
        device int*   oi = outIds + (uint64_t)qi * k;
        device float* ov = outVal + (uint64_t)qi * k;
        for (uint i = 0; i < k; ++i) {
            uint src = kk - 1u - i;
            oi[i] = redId[src];
            ov[i] = -redScore[src] + qn;
        }
    }
}
)";

// Shared helpers (normalizeRows, parallelFor, accumulateCentroids, kmeansGpu,
// acquireMetalDevice) live in Internal.h (mflat::detail).

}  // namespace

struct IvfPqIndex::Impl {
    int    dim     = 0;
    Metric metric  = Metric::L2;
    int    nlist   = 0;
    int    m       = 0;     // sub-quantizers / bytes per code
    int    dsub    = 0;     // dim / m
    int    dbCount = 0;
    bool   ready   = false;

    std::unique_ptr<CoarseQuantizer>   cq;   // centroids + CSR + coarse FlatIndex + cell/id bufs
    std::unique_ptr<detail::PqTrainer> pq;   // PQ codebooks + norms + encode/LUT
    std::vector<uint8_t> reorderedCodes;  // dbCount × m (payload, CSR slot order)

    bool                 storeFull = false;  // keep full vectors for reranking
    std::vector<float>   fullById;           // dbCount × dim, ORIGINAL id order
                                             // (rotated space when OPQ is on)

    bool               residual = true;      // requested: PQ on x - coarse_centroid
    bool               opq      = false;     // requested: learned rotation pre-transform
    // Latched at build(): what the CURRENT index actually contains. search()
    // consults ONLY these — toggling the setters after build() cannot corrupt
    // an already-built index (it applies to the next build()).
    bool               residualActive = true;
    bool               opqActive      = false;
    std::vector<float> opqR;                 // dim × dim rotation (row-major)
    std::vector<float> precomp;              // nlist × m × 256 residual ADC table

    id<MTLDevice>               device     = nil;
    id<MTLCommandQueue>         queue      = nil;
    id<MTLComputePipelineState> adcPipe    = nil;
    id<MTLBuffer>               codesBuf   = nil;
    id<MTLBuffer>               pqcBuf     = nil;
    id<MTLBuffer>               pqnBuf     = nil;
    id<MTLBuffer>               precompBuf = nil;

    // GPU ADC search producing the kRun-NN shortlist (qPtr pre-normalized).
    // Member so it can touch the GPU buffers; caller ensures kRun<=kMaxK + GPU.
    void runPqGpu(const float* qPtr, int m, int kRun, int nprobe, SearchResult& pq);
};

IvfPqIndex::IvfPqIndex(int dim, Metric metric, int nlist, int m)
    : mImpl(std::make_unique<Impl>()) {
    mImpl->dim    = dim;
    mImpl->metric = metric;
    mImpl->nlist  = nlist;
    mImpl->m      = m;
    mImpl->dsub   = (m > 0) ? dim / m : 0;
    mImpl->pq     = std::make_unique<detail::PqTrainer>(dim, m);

    if (m <= 0 || dim <= 0 || dim % m != 0) {
        MFLAT_LOG_ERROR("IvfPqIndex: m (%d) must divide dim (%d)", m, dim);
        return;
    }
    mImpl->device = acquireMetalDevice();
    if (mImpl->device) {
        mImpl->queue = [mImpl->device newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [mImpl->device newLibraryWithSource:kShaderSrc options:nil error:&err];
        if (!lib) {
            MFLAT_LOG_ERROR("ivfpq shader compile failed: %s",
                            err ? [[err localizedDescription] UTF8String] : "?");
        } else {
            id<MTLFunction> fn = [lib newFunctionWithName:@"ivfpq_adc"];
            mImpl->adcPipe = [mImpl->device newComputePipelineStateWithFunction:fn error:&err];
            if (!mImpl->adcPipe)
                MFLAT_LOG_ERROR("ivfpq pipeline build failed: %s",
                                err ? [[err localizedDescription] UTF8String] : "?");
        }
    }
    // Coarse quantizer uses the device only when the ADC pipeline is usable.
    mImpl->cq = std::make_unique<CoarseQuantizer>(mImpl->adcPipe ? mImpl->device : nil, dim);
}

IvfPqIndex::~IvfPqIndex() = default;

void IvfPqIndex::setRerank(bool enable)   { mImpl->storeFull = enable; }
void IvfPqIndex::setResidual(bool enable) { mImpl->residual  = enable; }
void IvfPqIndex::setOpq(bool enable)      { mImpl->opq       = enable; }

int IvfPqIndex::dim()           const { return mImpl->dim; }
int IvfPqIndex::size()          const { return mImpl->dbCount; }
int IvfPqIndex::nlist()         const { return mImpl->nlist; }
int IvfPqIndex::subquantizers() const { return mImpl->m; }
bool IvfPqIndex::ready()        const { return mImpl->ready; }

void IvfPqIndex::build(const float* vectors, int n) {
    if (n <= 0 || mImpl->dim <= 0 || mImpl->m <= 0 || mImpl->dim % mImpl->m != 0) return;
    const int dim = mImpl->dim, M = mImpl->m;

    std::vector<float> data(vectors, vectors + static_cast<size_t>(n) * dim);
    if (mImpl->metric == Metric::Cosine) normalizeRows(data, n, dim);

    constexpr int kIters = 12;

    // OPQ pre-transform: learn R on a shuffled sample, then rotate the whole
    // db. Everything downstream (coarse, residuals, PQ, rerank store, queries)
    // lives in rotated space; orthogonality preserves L2/Cosine exactly.
    if (mImpl->opq) {
        const int ns = std::min(n, 65536);
        std::vector<float> sample;
        const float* sPtr = data.data();
        if (ns < n) {
            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;
            std::mt19937 rng(12345);
            std::shuffle(perm.begin(), perm.end(), rng);
            sample.resize(static_cast<size_t>(ns) * dim);
            parallelFor(ns, [&](int i) {
                std::copy_n(&data[static_cast<size_t>(perm[i]) * dim], dim,
                            &sample[static_cast<size_t>(i) * dim]);
            });
            sPtr = sample.data();
        }
        mImpl->opqR = trainOpqRotation(sPtr, ns, dim, M);
        std::vector<float> rotated(data.size());
        applyRotation(mImpl->opqR.data(), data.data(), rotated.data(), n, dim);
        data.swap(rotated);
    } else {
        mImpl->opqR.clear();
    }

    // Coarse quantizer FIRST (residual encoding needs the cell assignment).
    mImpl->cq->train(data.data(), n, mImpl->nlist, kIters,
                     CoarseQuantizer::KmeansBackend::ForceGpuAssign);
    mImpl->nlist = mImpl->cq->nlist();

    // Keep full vectors (original id order; rotated space when OPQ) for exact
    // reranking — snapshot BEFORE the residual subtraction below.
    if (mImpl->storeFull) mImpl->fullById = data;

    // Residual mode: PQ quantizes r = x - coarse_centroid(x). Residuals are
    // much smaller than raw vectors, so the same m bytes carry far more
    // precision. Assignment is reconstructed from the CSR (cells disjoint).
    if (mImpl->residual) {
        const std::vector<int>& cs  = mImpl->cq->cellStart();
        const std::vector<int>& rid = mImpl->cq->reorderedIds();
        std::vector<int32_t> assign(n);
        parallelFor(mImpl->nlist, [&](int c) {
            for (int j = cs[c]; j < cs[c + 1]; ++j) assign[rid[j]] = c;
        });
        const std::vector<float>& cent = mImpl->cq->centroids();
        parallelFor(n, [&](int i) {
            float*       v = &data[static_cast<size_t>(i) * dim];
            const float* c = &cent[static_cast<size_t>(assign[i]) * dim];
            for (int d = 0; d < dim; ++d) v[d] -= c[d];
        });
    }

    // PQ training on residuals (or raw vectors when residual is off); codes
    // come straight from the sub-space k-means assignment (bit-exact).
    std::vector<uint8_t> codes(static_cast<size_t>(n) * M);
    mImpl->pq->train(data.data(), n, kIters, &codes);

    // Reorder the m-byte codes into CSR slot order (id order -> slots).
    mImpl->reorderedCodes.assign(static_cast<size_t>(n) * M, 0);
    mImpl->cq->reorderPayload(codes.data(), mImpl->reorderedCodes.data(),
                              static_cast<size_t>(M));

    // Residual ADC cross table T[cell][mm][j] = 2 c·pqc (see kernel comment).
    if (mImpl->residual) {
        mImpl->precomp.assign(static_cast<size_t>(mImpl->nlist) * mImpl->pq->lutSize(), 0.0f);
        mImpl->pq->buildCellTable(mImpl->cq->centroids().data(), mImpl->nlist,
                                  mImpl->precomp.data());
    } else {
        mImpl->precomp.clear();
    }

    mImpl->dbCount = n;
    mImpl->ready   = true;

    // GPU setup: upload codes + PQ tables (cell/id buffers live in the CQ).
    if (mImpl->adcPipe && mImpl->cq->gpuReady()) {
        auto buf = [&](const void* p, size_t bytes) {
            return [mImpl->device newBufferWithBytes:p length:bytes
                                             options:MTLResourceStorageModeShared];
        };
        mImpl->codesBuf   = buf(mImpl->reorderedCodes.data(), mImpl->reorderedCodes.size());
        mImpl->pqcBuf     = buf(mImpl->pq->centroids().data(), mImpl->pq->centroids().size() * sizeof(float));
        mImpl->pqnBuf     = buf(mImpl->pq->norms().data(),     mImpl->pq->norms().size() * sizeof(float));
        mImpl->precompBuf = mImpl->residual
            ? buf(mImpl->precomp.data(), mImpl->precomp.size() * sizeof(float))
            : nil;
    }

    // Latch the mode flags this index was actually built with (see Impl).
    mImpl->residualActive = mImpl->residual;
    mImpl->opqActive      = mImpl->opq;
}

namespace {

// CPU ADC fallback (also the recall reference): exact over the stored codes.
// Takes raw fields (not the private Impl) so it stays a free function. Mirrors
// the kernel term-for-term, including the residual decomposition.
void searchCpu(int dim, const CoarseQuantizer& cq,
               const detail::PqTrainer&    pq,
               const std::vector<uint8_t>& reorderedCodes,
               bool residual, const std::vector<float>& precomp,
               const float* qPtr, int m, int k, int nprobe, SearchResult& out) {
    const int M = pq.codeBytes(), ksub = pq.ksub();
    const std::vector<int>&   cellStart    = cq.cellStart();
    const std::vector<int>&   reorderedIds = cq.reorderedIds();
    const std::vector<float>& cent         = cq.centroids();
    parallelFor(m, [&](int qi) {
        const float* q = qPtr + static_cast<size_t>(qi) * dim;
        std::vector<float> lut(static_cast<size_t>(pq.lutSize()));
        pq.buildAdcTable(q, lut.data());
        std::vector<int> cells(nprobe);
        cq.probeCellsCpu(q, nprobe, cells.data());
        std::vector<float> bestScore(k, -std::numeric_limits<float>::infinity());
        std::vector<int>   bestId(k, -1);
        for (int pp = 0; pp < nprobe; ++pp) {
            const int c = cells[pp];
            // Residual terms: exact fp32 coarse ||q-c||^2 + the per-cell table.
            const float base = residual
                ? sqL2(q, &cent[static_cast<size_t>(c) * dim], dim) : 0.0f;
            const float* Tc = residual
                ? &precomp[static_cast<size_t>(c) * M * ksub] : nullptr;
            for (int j = cellStart[c]; j < cellStart[c + 1]; ++j) {
                const uint8_t* code = &reorderedCodes[static_cast<size_t>(j) * M];
                float dist = base;
                if (residual) {
                    for (int mm = 0; mm < M; ++mm) {
                        const int cd = code[mm];
                        dist += lut[static_cast<size_t>(mm) * ksub + cd]
                              + Tc[static_cast<size_t>(mm) * ksub + cd];
                    }
                } else {
                    for (int mm = 0; mm < M; ++mm)
                        dist += lut[static_cast<size_t>(mm) * ksub + code[mm]];
                }
                const float s = -dist;
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
        // Residual dist already IS ~||q-x||^2; non-residual adds ||q||^2 back.
        const float qn = residual ? 0.0f : dot(q, q, dim);
        for (int i = 0; i < k; ++i) {
            const int src = k - 1 - i;
            const size_t o = static_cast<size_t>(qi) * k + i;
            out.ids[o]       = bestId[src];
            out.distances[o] = -bestScore[src] + qn;
        }
    });
}

}  // namespace

// GPU ADC PQ search producing the kRun-NN shortlist. qPtr is already
// metric-normalized; writes `pq` (m × kRun). Caller guarantees kRun <= kMaxK
// and a usable GPU.
void IvfPqIndex::Impl::runPqGpu(const float* qPtr, int m, int kRun,
                                int nprobe, SearchResult& pq) {
    std::vector<int32_t> probed;
    cq->probeCells(qPtr, m, nprobe, probed);   // m × nprobe nearest cells
    const int32_t* probedPtr = probed.data();

    // Residual coarse term ||q-c||^2 per probed cell, computed EXACTLY on the
    // host in fp32 (the coarse FlatIndex's own distances are fp16-derived —
    // reusing them would inject fp16 error into every candidate score).
    std::vector<float> coarseVals;
    if (residualActive) {
        coarseVals.resize(static_cast<size_t>(m) * nprobe);
        const std::vector<float>& cent = cq->centroids();
        const int d = this->dim;
        parallelFor(m, [&](int qi) {
            const float* q = qPtr + static_cast<size_t>(qi) * d;
            for (int pp = 0; pp < nprobe; ++pp) {
                const int c = probed[static_cast<size_t>(qi) * nprobe + pp];
                coarseVals[static_cast<size_t>(qi) * nprobe + pp] = (c >= 0)
                    ? sqL2(q, &cent[static_cast<size_t>(c) * d], d)
                    : std::numeric_limits<float>::infinity();
            }
        });
    }

    @autoreleasepool {
        auto bufWith = [&](const void* p, size_t bytes) {
            return [device newBufferWithBytes:p length:bytes
                                      options:MTLResourceStorageModeShared];
        };
        id<MTLBuffer> qBuf      = bufWith(qPtr, static_cast<size_t>(m) * dim * sizeof(float));
        id<MTLBuffer> probedBuf = bufWith(probedPtr, static_cast<size_t>(m) * nprobe * sizeof(int32_t));
        id<MTLBuffer> outIdBuf  = [device newBufferWithLength:static_cast<size_t>(m) * kRun * sizeof(int32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outValBuf = [device newBufferWithLength:static_cast<size_t>(m) * kRun * sizeof(float)
                                                     options:MTLResourceStorageModeShared];

        // Residual-only buffers; pqnBuf stands in when unused (the kernel never
        // reads buffers 10/11 with residual==0, but Metal wants a binding).
        id<MTLBuffer> cvBuf = residualActive
            ? bufWith(coarseVals.data(), coarseVals.size() * sizeof(float))
            : pqnBuf;

        PqParams p;
        p.dim = (uint32_t)dim; p.k = (uint32_t)kRun; p.nprobe = (uint32_t)nprobe;
        p.m = (uint32_t)this->m; p.dsub = (uint32_t)dsub; p.queryCount = (uint32_t)m;
        p.residual = residualActive ? 1u : 0u;

        // Threadgroup sizing: the reduction scratch is (tg/32) * kk entries
        // (simdgroup-first merge), so k no longer constrains tg — always run
        // the widest power-of-two group (the old tgs*k scratch forced tg=32
        // at k=64, measured 5-15x SLOWER than the CPU path).
        uint32_t kkP = 1; while (kkP < (uint32_t)kRun) kkP <<= 1;
        NSUInteger tg = std::min<NSUInteger>(adcPipe.maxTotalThreadsPerThreadgroup, 256);
        { NSUInteger pw = 32; while (pw * 2 <= tg) pw *= 2; tg = pw; }
        const NSUInteger scratch  = (((tg / 32) * kkP * 4) + 15) & ~NSUInteger(15);
        const NSUInteger lutBytes = ((static_cast<NSUInteger>(this->m) * kKsub * sizeof(float)) + 15) & ~NSUInteger(15);
        const bool tgLut = lutBytes + 2 * scratch + 32 <= [device maxThreadgroupMemoryLength];

        // LUT spill (m_sub too large for threadgroup memory): per-dispatch
        // device scratch, batch chunked so the scratch stays <= ~64 MB.
        int chunk = m;
        id<MTLBuffer> lutSpill = pqnBuf;   // dummy binding in tgLut mode (never touched)
        if (!tgLut) {
            const size_t lutFloats = static_cast<size_t>(this->m) * kKsub;
            chunk = static_cast<int>(std::max<size_t>(
                1, (64ull << 20) / (lutFloats * sizeof(float))));
            chunk = std::min(chunk, m);
            lutSpill = [device newBufferWithLength:static_cast<size_t>(chunk) * lutFloats * sizeof(float)
                                           options:MTLResourceStorageModePrivate];
        }
        p.tgLut = tgLut ? 1u : 0u;

        id<MTLCommandBuffer>         cb  = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:adcPipe];
        [enc setBuffer:codesBuf offset:0 atIndex:0];
        [enc setBuffer:cq->idBuffer()   offset:0 atIndex:1];
        [enc setBuffer:cq->cellBuffer() offset:0 atIndex:2];
        [enc setBuffer:qBuf     offset:0 atIndex:3];
        [enc setBuffer:probedBuf offset:0 atIndex:4];
        [enc setBuffer:pqcBuf   offset:0 atIndex:5];
        [enc setBuffer:pqnBuf   offset:0 atIndex:6];
        [enc setBuffer:outIdBuf offset:0 atIndex:7];
        [enc setBuffer:outValBuf offset:0 atIndex:8];
        [enc setBuffer:(residualActive ? precompBuf : pqnBuf) offset:0 atIndex:10];
        [enc setBuffer:cvBuf offset:0 atIndex:11];
        [enc setBuffer:lutSpill offset:0 atIndex:12];
        [enc setThreadgroupMemoryLength:(tgLut ? lutBytes : 16) atIndex:0];
        [enc setThreadgroupMemoryLength:scratch atIndex:1];
        [enc setThreadgroupMemoryLength:scratch atIndex:2];
        // Serial encoder: chunk t+1's LUT writes are ordered after chunk t's
        // reads of the shared spill scratch (single dispatch when tgLut).
        for (int start = 0; start < m; start += chunk) {
            p.qBase = static_cast<uint32_t>(start);
            [enc setBytes:&p length:sizeof(p) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(std::min(chunk, m - start)), 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        std::memcpy(pq.ids.data(),       [outIdBuf contents],  pq.ids.size() * sizeof(int32_t));
        std::memcpy(pq.distances.data(), [outValBuf contents], pq.distances.size() * sizeof(float));
    }
}

SearchResult IvfPqIndex::search(const float* queries, int m, int k, int nprobe, int rerank) {
    SearchResult out;
    if (m <= 0 || !mImpl->ready) return out;
    const int dim = mImpl->dim, nlist = mImpl->nlist;
    if (k < 1) k = 1;
    if (k > kMaxK) k = kMaxK;
    if (nprobe < 1) nprobe = 1;
    if (nprobe > nlist) nprobe = nlist;

    // Normalize queries once (Cosine); reused by the shortlist and the rerank.
    std::vector<float> qNorm;
    const float* qPtr = queries;
    if (mImpl->metric == Metric::Cosine) {
        qNorm.assign(queries, queries + static_cast<size_t>(m) * dim);
        normalizeRows(qNorm, m, dim);
        qPtr = qNorm.data();
    }

    // OPQ: rotate queries into the index's space, exactly once — the shortlist
    // AND the rerank both use qPtr (fullById is stored rotated too).
    std::vector<float> qRot;
    if (mImpl->opqActive && !mImpl->opqR.empty()) {
        qRot.resize(static_cast<size_t>(m) * dim);
        applyRotation(mImpl->opqR.data(), qPtr, qRot.data(), m, dim);
        qPtr = qRot.data();
    }

    const bool doRerank = rerank > k && mImpl->storeFull && !mImpl->fullById.empty();
    const int  kRun = doRerank ? std::min(rerank, mImpl->dbCount) : k;

    // ---- PQ shortlist (kRun-NN by ADC) ----------------------------------
    SearchResult pq;
    pq.ids.assign(static_cast<size_t>(m) * kRun, -1);
    pq.distances.assign(static_cast<size_t>(m) * kRun, std::numeric_limits<float>::infinity());

    // GPU-vs-CPU routing: the GPU ADC wins at kRun <= 16; above that the
    // per-thread top-k reduction cost grows with nextPow2(kRun) and the
    // multithreaded CPU ADC is measured 1.2-3x faster (M2 Pro,
    // temp/pq_shapes.mm; kRun > kMaxK has no GPU path at all).
    // MFLAT_PQ_CPU=1/0 forces the CPU/GPU path.
    bool cpuRoute = (kRun > 16);
    if (const char* e = std::getenv("MFLAT_PQ_CPU")) cpuRoute = (e[0] == '1');
    const bool useGpu = !cpuRoute && mImpl->adcPipe && mImpl->cq->gpuReady() && kRun <= kMaxK;
    if (useGpu) {
        mImpl->runPqGpu(qPtr, m, kRun, nprobe, pq);
    } else {
        searchCpu(dim, *mImpl->cq, *mImpl->pq, mImpl->reorderedCodes,
                  mImpl->residualActive, mImpl->precomp,
                  qPtr, m, kRun, nprobe, pq);   // *mImpl->cq/pq = subsystems, pq = out
    }

    if (!doRerank) return pq;

    // ---- exact rerank of the shortlist down to the true top-k -----------
    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k, std::numeric_limits<float>::infinity());
    const std::vector<float>& full = mImpl->fullById;   // exact rerank vectors
    parallelFor(m, [&](int qi) {
        const float* q = qPtr + static_cast<size_t>(qi) * dim;
        std::vector<float> bestScore(k, -std::numeric_limits<float>::infinity());
        std::vector<int>   bestId(k, -1);
        for (int sIdx = 0; sIdx < kRun; ++sIdx) {
            const int id = pq.ids[static_cast<size_t>(qi) * kRun + sIdx];
            if (id < 0) continue;
            const float* d = &full[static_cast<size_t>(id) * dim];
            const float s = detail::score(mImpl->metric, q, d, dim);
            if (s > bestScore[0]) {
                int pos = 0;
                while (pos + 1 < k && s > bestScore[pos + 1]) {
                    bestScore[pos] = bestScore[pos + 1];
                    bestId[pos]    = bestId[pos + 1];
                    ++pos;
                }
                bestScore[pos] = s;
                bestId[pos]    = id;
            }
        }
        for (int i = 0; i < k; ++i) {
            const int src = k - 1 - i;
            const size_t o = static_cast<size_t>(qi) * k + i;
            out.ids[o]       = bestId[src];
            out.distances[o] = detail::scoreToValue(mImpl->metric, bestScore[src]);
        }
    });
    return out;
}

}  // namespace mflat
