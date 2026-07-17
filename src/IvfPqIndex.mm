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
#include <chrono>
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
#include "TopkMsl.h"     // kTopkMslSrc + nextPow2K/topkScratchBytes (shared reduction)
#include "GpuScratch.h"  // persistent per-search buffers (no alloc per query)
#include "FastScan.h"    // 4-bit fast-scan ADC (NEON vqtbl1q_u8; CPU shortlist path)
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

// kTopkMslSrc (src/TopkMsl.h) is prepended at pipeline-build time: it supplies
// kMaxK, insertTopk and reduceTopkTg (the simdgroup-first top-k reduction,
// shared with FlatIndex / IvfIndex).
NSString* const kShaderBody = @R"(
struct PqParams { uint dim; uint k; uint nprobe; uint m; uint dsub; uint queryCount; uint residual; uint tgLut; uint qBase; };
constant uint kKsub = 256;

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
    std::vector<__fp16>  fullById;           // dbCount × dim, ORIGINAL id order,
                                             // fp16 (rotated space when OPQ on).
                                             // fp16, not fp32: the rerank is a
                                             // random DRAM gather of these rows,
                                             // so halving the bytes ~halves its
                                             // cost — and fp16 preserves ranking,
                                             // so recall is unchanged.

    bool               residual = true;      // requested: PQ on x - coarse_centroid
    bool               opq      = false;     // requested: learned rotation pre-transform
    bool               fastScan = false;     // requested: 4-bit fast-scan codes (CPU scan)
    // Latched at build(): what the CURRENT index actually contains. search()
    // consults ONLY these — toggling the setters after build() cannot corrupt
    // an already-built index (it applies to the next build()).
    bool               residualActive = true;
    bool               opqActive      = false;
    bool               fastScanActive = false;
    std::vector<float> opqR;                 // dim × dim rotation (row-major)
    std::vector<float> precomp;              // nlist × m × 256 residual ADC table

    // 4-bit fast-scan state (fastScanActive): m4 = 2*m subquantizers of 16
    // centroids — same bytes/vector, two codes per byte. Codes live in the
    // blocked nibble layout (FastScan.h); ids are per-cell padded to blocks.
    std::unique_ptr<detail::PqTrainer> pq4;
    std::vector<uint8_t>  fsCodes;       // sum(blocks) × m4 × 16
    std::vector<int32_t>  fsIds;         // sum(blocks) × 32, -1 padded
    std::vector<int>      fsBlockStart;  // per cell, first block index
    std::vector<float>    fsPrecomp;     // nlist × m4 × 16 (residual)
    int                   m4    = 0;
    int                   fsDim = 0;     // dim zero-padded to a multiple of m4

    id<MTLDevice>               device     = nil;
    id<MTLCommandQueue>         queue      = nil;
    id<MTLComputePipelineState> adcPipe    = nil;
    id<MTLBuffer>               codesBuf   = nil;
    id<MTLBuffer>               pqcBuf     = nil;
    id<MTLBuffer>               pqnBuf     = nil;
    id<MTLBuffer>               precompBuf = nil;

    // Per-search buffers, kept alive across calls (see GpuScratch.h).
    struct Slot { enum { Query = 0, Probed, OutId, OutVal, CoarseVals, LutSpill }; };
    detail::GpuScratch scratch_;

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
        mImpl->scratch_.setDevice(mImpl->device);
        NSError* err = nil;
        NSString* src = [detail::kTopkMslSrc stringByAppendingString:kShaderBody];
        id<MTLLibrary> lib = [mImpl->device newLibraryWithSource:src options:nil error:&err];
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
void IvfPqIndex::setFastScan(bool enable)  { mImpl->fastScan  = enable; }

int IvfPqIndex::dim()           const { return mImpl->dim; }
int IvfPqIndex::size()          const { return mImpl->dbCount; }
int IvfPqIndex::nlist()         const { return mImpl->nlist; }
int IvfPqIndex::subquantizers() const { return mImpl->m; }
bool IvfPqIndex::ready()        const { return mImpl->ready; }

void IvfPqIndex::build(const float* vectors, int n) {
    if (n <= 0 || mImpl->dim <= 0 || mImpl->m <= 0 || mImpl->dim % mImpl->m != 0) return;
    const int dim = mImpl->dim, M = mImpl->m;

    // Per-stage build timing (INFO): the build is a benchmark talking point, so
    // report where the time goes rather than re-deriving it with a profiler.
    using Clock = std::chrono::steady_clock;
    auto tick = Clock::now();
    auto lap  = [&tick] {
        const auto now = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - tick).count();
        tick = now;
        return ms;
    };
    double tOpq = 0, tCoarse = 0, tResid = 0, tPq = 0, tCell = 0, tGpu = 0;

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
        int opqIters = 20;   // MFLAT_OPQ_ITERS overrides (build-cost tuning)
        if (const char* e = std::getenv("MFLAT_OPQ_ITERS")) opqIters = std::max(1, atoi(e));
        mImpl->opqR = trainOpqRotation(sPtr, ns, dim, M, opqIters);
        std::vector<float> rotated(data.size());
        applyRotation(mImpl->opqR.data(), data.data(), rotated.data(), n, dim);
        data.swap(rotated);
    } else {
        mImpl->opqR.clear();
    }
    tOpq = lap();

    // Coarse quantizer FIRST (residual encoding needs the cell assignment).
    int cqIters = 25;   // faiss default; see IvfIndex.mm. MFLAT_KMEANS_ITERS overrides.
    if (const char* e = std::getenv("MFLAT_KMEANS_ITERS")) cqIters = atoi(e);
    mImpl->cq->train(data.data(), n, mImpl->nlist, cqIters,
                     CoarseQuantizer::KmeansBackend::ForceGpuAssign,
                     /*spherical=*/mImpl->metric == Metric::Cosine);
    mImpl->nlist = mImpl->cq->nlist();
    tCoarse = lap();

    // Keep full vectors (original id order; rotated space when OPQ) for exact
    // reranking — snapshot BEFORE the residual subtraction below.
    if (mImpl->storeFull) {
        mImpl->fullById.resize(static_cast<size_t>(n) * dim);
        parallelFor(n, [&](int i) {
            const float* src = &data[static_cast<size_t>(i) * dim];
            __fp16*      dst = &mImpl->fullById[static_cast<size_t>(i) * dim];
            for (int c = 0; c < dim; ++c) dst[c] = static_cast<__fp16>(src[c]);
        });
    }

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
        const std::vector<float>& cent = mImpl->cq->encodeCentroids();
        parallelFor(n, [&](int i) {
            float*       v = &data[static_cast<size_t>(i) * dim];
            const float* c = &cent[static_cast<size_t>(assign[i]) * dim];
            for (int d = 0; d < dim; ++d) v[d] -= c[d];
        });
    }
    tResid = lap();

    // 4-bit fast-scan: m4 = 2*M subquantizers of 16 centroids — SAME bytes per
    // vector (two codes per byte), but the per-subq LUT fits one NEON register
    // so the CPU scan is table-lookup-per-instruction (FastScan.h).
    //
    // Any dim is served by ZERO-PADDING to fsDim = ceil(dim/m4)*m4: padded
    // coordinates are 0 in the data, the residuals, the coarse centroids and
    // the query alike, so they contribute (0-0)^2 = 0 to every distance — the
    // padding is EXACT, it just wastes the tail subquantizers on zeros. This
    // matters in practice: the old dim %% (2m) == 0 gate silently dropped
    // glove-100 (m=20 -> 40 does not divide 100) back to the scalar 8-bit ADC,
    // which is the entire reason glove IVFPQ lost 1.7-2.7x to faiss's
    // fast-scan while SIFT won. Only m4 > 128 falls back now (u16 ADC
    // accumulation is exact only for m4 <= 128).
    const bool wantFs = mImpl->fastScan && (2 * M <= 128);
    if (mImpl->fastScan && !wantFs)
        MFLAT_LOG_WARN("ivfpq fastscan: m=%d (m4=%d > 128) would overflow the "
                       "u16 ADC accumulator — using 8-bit codes", M, 2 * M);

    if (wantFs) {
        const int m4    = 2 * M;
        const int fsDim = ((dim + m4 - 1) / m4) * m4;
        mImpl->m4    = m4;
        mImpl->fsDim = fsDim;
        mImpl->pq4 = std::make_unique<detail::PqTrainer>(fsDim, m4, detail::kFsKsub);

        // Padded training/encoding copy (residuals or raw, whichever `data`
        // holds). fsDim == dim skips the copy.
        const float* trainPtr = data.data();
        std::vector<float> padded;
        if (fsDim != dim) {
            padded.assign(static_cast<size_t>(n) * fsDim, 0.0f);
            parallelFor(n, [&](int i) {
                std::copy_n(&data[static_cast<size_t>(i) * dim], dim,
                            &padded[static_cast<size_t>(i) * fsDim]);
            });
            trainPtr = padded.data();
        }
        std::vector<uint8_t> codes4(static_cast<size_t>(n) * m4);
        mImpl->pq4->train(trainPtr, n, kIters, &codes4, /*maxPointsPerCentroid=*/256);
        padded.clear();
        padded.shrink_to_fit();

        // CSR slot order, then per-cell blocked nibble packing + padded ids.
        std::vector<uint8_t> csr(static_cast<size_t>(n) * m4);
        mImpl->cq->reorderPayload(codes4.data(), csr.data(), static_cast<size_t>(m4));
        const std::vector<int>& cs  = mImpl->cq->cellStart();
        const std::vector<int>& rid = mImpl->cq->reorderedIds();
        mImpl->fsBlockStart.assign(mImpl->nlist + 1, 0);
        for (int c = 0; c < mImpl->nlist; ++c)
            mImpl->fsBlockStart[c + 1] = mImpl->fsBlockStart[c]
                                       + detail::fsBlocksFor(cs[c + 1] - cs[c]);
        const int totalBlk = mImpl->fsBlockStart[mImpl->nlist];
        mImpl->fsCodes.assign(static_cast<size_t>(totalBlk) * m4 * 16, 0);
        mImpl->fsIds.assign(static_cast<size_t>(totalBlk) * detail::kFsBlock, -1);
        parallelFor(mImpl->nlist, [&](int c) {
            const int lo = cs[c], count = cs[c + 1] - cs[c];
            if (count <= 0) return;
            detail::fsPackCell(&csr[static_cast<size_t>(lo) * m4], count, m4,
                               &mImpl->fsCodes[static_cast<size_t>(mImpl->fsBlockStart[c]) * m4 * 16]);
            int32_t* ids = &mImpl->fsIds[static_cast<size_t>(mImpl->fsBlockStart[c]) * detail::kFsBlock];
            for (int t = 0; t < count; ++t) ids[t] = rid[lo + t];
        });

        if (mImpl->residual) {
            mImpl->fsPrecomp.assign(static_cast<size_t>(mImpl->nlist) * mImpl->pq4->lutSize(), 0.0f);
            const float* centPtr = mImpl->cq->encodeCentroids().data();
            std::vector<float> centPad;   // centroids zero-padded to fsDim
            if (fsDim != dim) {
                centPad.assign(static_cast<size_t>(mImpl->nlist) * fsDim, 0.0f);
                for (int c = 0; c < mImpl->nlist; ++c)
                    std::copy_n(centPtr + static_cast<size_t>(c) * dim, dim,
                                &centPad[static_cast<size_t>(c) * fsDim]);
                centPtr = centPad.data();
            }
            mImpl->pq4->buildCellTable(centPtr, mImpl->nlist, mImpl->fsPrecomp.data());
        } else {
            mImpl->fsPrecomp.clear();
        }
        // No 8-bit codes and no GPU upload: fast-scan is the CPU shortlist path
        // (the GPU only ever served kRun <= 16 raw-ADC configs, none of which
        // sit on the high-recall Pareto this mode exists for).
        mImpl->reorderedCodes.clear();
        mImpl->precomp.clear();
        mImpl->codesBuf = nil; mImpl->pqcBuf = nil; mImpl->pqnBuf = nil;
        mImpl->precompBuf = nil;
    } else {
        // PQ training on residuals (or raw vectors when residual is off); codes
        // come straight from the sub-space k-means assignment (bit-exact).
        std::vector<uint8_t> codes(static_cast<size_t>(n) * M);
        mImpl->pq->train(data.data(), n, kIters, &codes);

        // Reorder the m-byte codes into CSR slot order (id order -> slots).
        mImpl->reorderedCodes.assign(static_cast<size_t>(n) * M, 0);
        mImpl->cq->reorderPayload(codes.data(), mImpl->reorderedCodes.data(),
                                  static_cast<size_t>(M));

        tPq = lap();
        // Residual ADC cross table T[cell][mm][j] = 2 c·pqc (see kernel comment).
        if (mImpl->residual) {
            mImpl->precomp.assign(static_cast<size_t>(mImpl->nlist) * mImpl->pq->lutSize(), 0.0f);
            mImpl->pq->buildCellTable(mImpl->cq->encodeCentroids().data(), mImpl->nlist,
                                      mImpl->precomp.data());
        } else {
            mImpl->precomp.clear();
        }
        tCell = lap();
        mImpl->pq4.reset();
        mImpl->fsCodes.clear(); mImpl->fsIds.clear(); mImpl->fsBlockStart.clear();
        mImpl->fsPrecomp.clear();
        mImpl->m4 = 0;
    }

    mImpl->dbCount = n;
    mImpl->ready   = true;

    // GPU setup: upload codes + PQ tables (cell/id buffers live in the CQ).
    if (!wantFs && mImpl->adcPipe && mImpl->cq->gpuReady()) {
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
    tGpu = lap();

    // Latch the mode flags this index was actually built with (see Impl).
    mImpl->residualActive = mImpl->residual;
    mImpl->opqActive      = mImpl->opq;
    mImpl->fastScanActive = wantFs;
    MFLAT_LOG_INFO("ivfpq build: opq %.0f, coarse %.0f, residual %.0f, pq-train "
                   "%.0f, cell-table %.0f, gpu-upload %.0f ms (n=%d nlist=%d m=%d "
                   "opq=%d fs=%d)", tOpq, tCoarse, tResid, tPq, tCell, tGpu,
                   n, mImpl->nlist, M, mImpl->opq, wantFs);
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
    const std::vector<float>& cent         = cq.encodeCentroids();
    const size_t lutN = static_cast<size_t>(pq.lutSize());

    // Small k: shift-insert keeps the array fully sorted (worst at [0]) with
    // in-cache shifts. Large k (deep rerank shortlists, kRun up to ~1280):
    // the O(k)-per-accept shifting dominates the whole scan — use a bounded
    // min-heap on score instead (root [0] stays the worst kept, so the
    // `s > bestScore[0]` threshold contract is unchanged), O(log k) per accept.
    const bool heapK = k > 64;
    auto insertTopk = [&](float s, int id, std::vector<float>& bestScore,
                          std::vector<int>& bestId) {
        if (!heapK) {
            int pos = 0;
            while (pos + 1 < k && s > bestScore[pos + 1]) {
                bestScore[pos] = bestScore[pos + 1];
                bestId[pos]    = bestId[pos + 1];
                ++pos;
            }
            bestScore[pos] = s;
            bestId[pos]    = id;
            return;
        }
        int i = 0;
        for (;;) {                    // replace the root, sift the new value down
            const int l = 2 * i + 1;
            if (l >= k) break;
            int c = l;
            if (l + 1 < k && bestScore[l + 1] < bestScore[l]) c = l + 1;
            if (bestScore[c] >= s) break;
            bestScore[i] = bestScore[c];
            bestId[i]    = bestId[c];
            i = c;
        }
        bestScore[i] = s;
        bestId[i]    = id;
    };

    // Scan probe-list slice [ppLo, ppHi) of query `q` into bestScore/bestId.
    // A SLICE (not the whole list) so the same body serves both the
    // query-parallel shape and the cell-parallel one used at small m.
    auto scanSlice = [&](const float* q, const int* cells, int ppLo, int ppHi,
                         const float* lut, float* comb,
                         std::vector<float>& bestScore, std::vector<int>& bestId) {
        for (int pp = ppLo; pp < ppHi; ++pp) {
            const int c = cells[pp];
            if (c < 0) continue;
            const int lo = cellStart[c], hi = cellStart[c + 1];
            // Residual terms: exact fp32 coarse ||q-c||^2 + the per-cell table.
            // Folding the per-cell table into the query LUT once per cell
            // (M*ksub adds) halves the random loads in the hot loop; only
            // worth it when the cell has enough codes to amortize.
            float base = 0.0f;
            const float* T = lut;
            if (residual) {
                base = sqL2(q, &cent[static_cast<size_t>(c) * dim], dim);
                const float* Tc = &precomp[static_cast<size_t>(c) * M * ksub];
                if (static_cast<size_t>(hi - lo) * M >= 2 * lutN) {
                    for (size_t e = 0; e < lutN; ++e) comb[e] = lut[e] + Tc[e];
                    T = comb;
                    Tc = nullptr;
                }
                if (Tc) {   // small cell: classic two-table loop
                    for (int j = lo; j < hi; ++j) {
                        const uint8_t* code = &reorderedCodes[static_cast<size_t>(j) * M];
                        float dist = base;
                        for (int mm = 0; mm < M; ++mm) {
                            const int cd = code[mm];
                            dist += lut[static_cast<size_t>(mm) * ksub + cd]
                                  + Tc[static_cast<size_t>(mm) * ksub + cd];
                        }
                        const float s = -dist;
                        if (s > bestScore[0]) insertTopk(s, reorderedIds[j], bestScore, bestId);
                    }
                    continue;
                }
            }
            for (int j = lo; j < hi; ++j) {
                const uint8_t* code = &reorderedCodes[static_cast<size_t>(j) * M];
                float dist = base;
                for (int mm = 0; mm < M; ++mm)
                    dist += T[static_cast<size_t>(mm) * ksub + code[mm]];
                const float s = -dist;
                if (s > bestScore[0]) insertTopk(s, reorderedIds[j], bestScore, bestId);
            }
        }
    };
    auto emit = [&](int qi, const float* q, std::vector<float>& bestScore,
                    std::vector<int>& bestId) {
        if (heapK) {   // heap order -> ascending score, matching the array layout
            std::vector<std::pair<float, int>> tmp(k);
            for (int i = 0; i < k; ++i) tmp[i] = { bestScore[i], bestId[i] };
            std::sort(tmp.begin(), tmp.end());
            for (int i = 0; i < k; ++i) { bestScore[i] = tmp[i].first; bestId[i] = tmp[i].second; }
        }
        // Residual dist already IS ~||q-x||^2; non-residual adds ||q||^2 back.
        const float qn = residual ? 0.0f : dot(q, q, dim);
        for (int i = 0; i < k; ++i) {
            const int src = k - 1 - i;
            const size_t o = static_cast<size_t>(qi) * k + i;
            out.ids[o]       = bestId[src];
            out.distances[o] = -bestScore[src] + qn;
        }
    };

    // SMALL BATCHES: parallelise across CELLS, not queries. Parallelising over
    // queries leaves every core but one idle at m=1, while that one core still
    // scans all nprobe*(n/nlist) codes — the same bug IvfIndex had, and the
    // reason single-query IVF/IVFPQ looked algorithmically doomed.
    const int nt = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    if (m < nt && nprobe > 1) {
        for (int qi = 0; qi < m; ++qi) {
            const float* q = qPtr + static_cast<size_t>(qi) * dim;
            std::vector<float> lut(lutN);
            pq.buildAdcTable(q, lut.data());
            std::vector<int> cells(nprobe);
            cq.probeCellsCpu(q, nprobe, cells.data());

            const int slices = std::min(nt, nprobe);
            std::vector<std::vector<float>> pScore(slices,
                std::vector<float>(k, -std::numeric_limits<float>::infinity()));
            std::vector<std::vector<int>> pId(slices, std::vector<int>(k, -1));
            std::vector<std::vector<float>> pComb(slices,
                std::vector<float>(residual ? lutN : 0));   // per-thread scratch
            const int chunk = (nprobe + slices - 1) / slices;
            parallelFor(slices, [&](int t) {
                const int lo = t * chunk, hi = std::min(nprobe, lo + chunk);
                if (lo < hi)
                    scanSlice(q, cells.data(), lo, hi, lut.data(),
                              residual ? pComb[t].data() : nullptr, pScore[t], pId[t]);
            });

            std::vector<float> bestScore(k, -std::numeric_limits<float>::infinity());
            std::vector<int>   bestId(k, -1);
            for (int t = 0; t < slices; ++t)
                for (int e = 0; e < k; ++e)
                    if (pId[t][e] >= 0 && pScore[t][e] > bestScore[0])
                        insertTopk(pScore[t][e], pId[t][e], bestScore, bestId);
            emit(qi, q, bestScore, bestId);
        }
        return;
    }

    // LARGE BATCHES: one query per thread — the queries already fill the machine.
    parallelFor(m, [&](int qi) {
        const float* q = qPtr + static_cast<size_t>(qi) * dim;
        std::vector<float> lut(lutN);
        pq.buildAdcTable(q, lut.data());
        std::vector<float> comb(residual ? lutN : 0);
        std::vector<int> cells(nprobe);
        cq.probeCellsCpu(q, nprobe, cells.data());
        std::vector<float> bestScore(k, -std::numeric_limits<float>::infinity());
        std::vector<int>   bestId(k, -1);
        scanSlice(q, cells.data(), 0, nprobe, lut.data(),
                  residual ? comb.data() : nullptr, bestScore, bestId);
        emit(qi, q, bestScore, bestId);
    });
}

// 4-bit fast-scan CPU shortlist (see FastScan.h). Same shape as searchCpu —
// slice-scanner + cell-parallel small batches — but the inner loop scores 32
// candidates per pass with register-resident 16-entry LUTs (NEON vqtbl1q_u8)
// instead of a scalar gather per code byte. Shortlist ordering uses QUANTIZED
// distances (u8 LUT entries, u16 sums); the exact rerank repairs the rounding,
// exactly as it already repairs the 4-bit quantization itself.
void searchCpuFS(int dim, const CoarseQuantizer& cq, const detail::PqTrainer& pq4,
                 const std::vector<uint8_t>& fsCodes,
                 const std::vector<int>&     fsBlockStart,
                 const std::vector<int32_t>& fsIds,
                 bool residual, const std::vector<float>& fsPrecomp,
                 const float* qPtr, int m, int k, int nprobe, SearchResult& out) {
    const int m4    = pq4.m();
    const int fsDim = pq4.dim();   // dim zero-padded to a multiple of m4
    const size_t lutN = static_cast<size_t>(pq4.lutSize());   // m4 * 16
    const std::vector<int>&   cellStart = cq.cellStart();
    const std::vector<float>& cent      = cq.encodeCentroids();

    // The LUT is built from a query of length fsDim; when fsDim != dim the
    // tail is zero (matching the padded codebook — exact, see build()).
    auto padQuery = [&](const float* q, std::vector<float>& scratch) -> const float* {
        if (fsDim == dim) return q;
        scratch.assign(static_cast<size_t>(fsDim), 0.0f);
        std::copy_n(q, dim, scratch.begin());
        return scratch.data();
    };

    // Small k: shift-insert keeps the array fully sorted (worst at [0]) with
    // in-cache shifts. Large k (deep rerank shortlists, kRun up to ~1280):
    // the O(k)-per-accept shifting dominates the whole scan — use a bounded
    // min-heap on score instead (root [0] stays the worst kept, so the
    // `s > bestScore[0]` threshold contract is unchanged), O(log k) per accept.
    const bool heapK = k > 64;
    auto insertTopk = [&](float s, int id, std::vector<float>& bestScore,
                          std::vector<int>& bestId) {
        if (!heapK) {
            int pos = 0;
            while (pos + 1 < k && s > bestScore[pos + 1]) {
                bestScore[pos] = bestScore[pos + 1];
                bestId[pos]    = bestId[pos + 1];
                ++pos;
            }
            bestScore[pos] = s;
            bestId[pos]    = id;
            return;
        }
        int i = 0;
        for (;;) {                    // replace the root, sift the new value down
            const int l = 2 * i + 1;
            if (l >= k) break;
            int c = l;
            if (l + 1 < k && bestScore[l + 1] < bestScore[l]) c = l + 1;
            if (bestScore[c] >= s) break;
            bestScore[i] = bestScore[c];
            bestId[i]    = bestId[c];
            i = c;
        }
        bestScore[i] = s;
        bestId[i]    = id;
    };

    // Scan probe-list slice [ppLo, ppHi). lut = the query's float LUT; lutc and
    // lq are per-thread scratch (residual combine + quantized table).
    auto scanSlice = [&](const float* q, const int* cells, int ppLo, int ppHi,
                         const float* lut, float* lutc, uint8_t* lq,
                         std::vector<float>& bestScore, std::vector<int>& bestId) {
        for (int pp = ppLo; pp < ppHi; ++pp) {
            const int c = cells[pp];
            if (c < 0) continue;
            const int count = cellStart[c + 1] - cellStart[c];
            if (count <= 0) continue;

            float base = 0.0f;
            const float* useLut = lut;
            if (residual) {
                base = sqL2(q, &cent[static_cast<size_t>(c) * dim], dim);
                const float* Tc = &fsPrecomp[static_cast<size_t>(c) * lutN];
                for (size_t e = 0; e < lutN; ++e) lutc[e] = lut[e] + Tc[e];
                useLut = lutc;
            }
            float scale, bias;
            detail::fsQuantizeLut(useLut, m4, lq, &scale, &bias);

            const uint8_t* blocks = &fsCodes[static_cast<size_t>(fsBlockStart[c]) * m4 * 16];
            const int32_t* ids    = &fsIds[static_cast<size_t>(fsBlockStart[c]) * detail::kFsBlock];
            const int nBlk = detail::fsBlocksFor(count);
            uint16_t d16[detail::kFsBlock];
            for (int b = 0; b < nBlk; ++b) {
                const uint16_t mn = detail::fsScanBlock(
                    blocks + static_cast<size_t>(b) * m4 * 16, lq, m4, d16);
                // Block-level early-out: nothing here can beat the worst kept.
                // bestScore holds -dist, so worst kept dist = -bestScore[0].
                const bool  full   = bestId[0] >= 0;
                const float worstD = -bestScore[0];
                if (full && base + static_cast<float>(mn) / scale + bias >= worstD)
                    continue;
                const int32_t* bid = ids + static_cast<size_t>(b) * detail::kFsBlock;
                for (int t = 0; t < detail::kFsBlock; ++t) {
                    const int id = bid[t];
                    if (id < 0) continue;
                    const float dist = base + static_cast<float>(d16[t]) / scale + bias;
                    const float s = -dist;
                    if (s > bestScore[0]) insertTopk(s, id, bestScore, bestId);
                }
            }
        }
    };
    auto emit = [&](int qi, const float* q, std::vector<float>& bestScore,
                    std::vector<int>& bestId) {
        if (heapK) {   // heap order -> ascending score, matching the array layout
            std::vector<std::pair<float, int>> tmp(k);
            for (int i = 0; i < k; ++i) tmp[i] = { bestScore[i], bestId[i] };
            std::sort(tmp.begin(), tmp.end());
            for (int i = 0; i < k; ++i) { bestScore[i] = tmp[i].first; bestId[i] = tmp[i].second; }
        }
        const float qn = residual ? 0.0f : dot(q, q, dim);
        for (int i = 0; i < k; ++i) {
            const int src = k - 1 - i;
            const size_t o = static_cast<size_t>(qi) * k + i;
            out.ids[o]       = bestId[src];
            out.distances[o] = -bestScore[src] + qn;
        }
    };

    const int nt = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    if (m < nt && nprobe > 1) {
        // Small batches: parallelise across CELLS (see searchCpu — same bug fix).
        for (int qi = 0; qi < m; ++qi) {
            const float* q = qPtr + static_cast<size_t>(qi) * dim;
            std::vector<float> lut(lutN), qPad;
            pq4.buildAdcTable(padQuery(q, qPad), lut.data());
            std::vector<int> cells(nprobe);
            cq.probeCellsCpu(q, nprobe, cells.data());

            const int slices = std::min(nt, nprobe);
            std::vector<std::vector<float>> pScore(slices,
                std::vector<float>(k, -std::numeric_limits<float>::infinity()));
            std::vector<std::vector<int>> pId(slices, std::vector<int>(k, -1));
            std::vector<std::vector<float>>   pLutc(slices, std::vector<float>(residual ? lutN : 0));
            std::vector<std::vector<uint8_t>> pLq(slices, std::vector<uint8_t>(lutN));
            const int chunk = (nprobe + slices - 1) / slices;
            parallelFor(slices, [&](int t) {
                const int lo = t * chunk, hi = std::min(nprobe, lo + chunk);
                if (lo < hi)
                    scanSlice(q, cells.data(), lo, hi, lut.data(),
                              residual ? pLutc[t].data() : nullptr, pLq[t].data(),
                              pScore[t], pId[t]);
            });
            std::vector<float> bestScore(k, -std::numeric_limits<float>::infinity());
            std::vector<int>   bestId(k, -1);
            for (int t = 0; t < slices; ++t)
                for (int e = 0; e < k; ++e)
                    if (pId[t][e] >= 0 && pScore[t][e] > bestScore[0])
                        insertTopk(pScore[t][e], pId[t][e], bestScore, bestId);
            emit(qi, q, bestScore, bestId);
        }
        return;
    }

    parallelFor(m, [&](int qi) {
        const float* q = qPtr + static_cast<size_t>(qi) * dim;
        std::vector<float> lut(lutN), qPad;
        pq4.buildAdcTable(padQuery(q, qPad), lut.data());
        std::vector<float>   lutc(residual ? lutN : 0);
        std::vector<uint8_t> lq(lutN);
        std::vector<int> cells(nprobe);
        cq.probeCellsCpu(q, nprobe, cells.data());
        std::vector<float> bestScore(k, -std::numeric_limits<float>::infinity());
        std::vector<int>   bestId(k, -1);
        scanSlice(q, cells.data(), 0, nprobe, lut.data(),
                  residual ? lutc.data() : nullptr, lq.data(), bestScore, bestId);
        emit(qi, q, bestScore, bestId);
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
        const std::vector<float>& cent = cq->encodeCentroids();
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
        // Persistent scratch, not a fresh allocation per call: newBufferWithBytes
        // maps pages and touches the driver, and six of them per query is most of
        // what a single-query GPU search was actually paying for.
        id<MTLBuffer> qBuf      = scratch_.upload(Slot::Query, qPtr,
                                                  static_cast<size_t>(m) * dim * sizeof(float));
        id<MTLBuffer> probedBuf = scratch_.upload(Slot::Probed, probedPtr,
                                                  static_cast<size_t>(m) * nprobe * sizeof(int32_t));
        id<MTLBuffer> outIdBuf  = scratch_.ensure(Slot::OutId,
                                                  static_cast<size_t>(m) * kRun * sizeof(int32_t));
        id<MTLBuffer> outValBuf = scratch_.ensure(Slot::OutVal,
                                                  static_cast<size_t>(m) * kRun * sizeof(float));

        // Residual-only buffers; pqnBuf stands in when unused (the kernel never
        // reads buffers 10/11 with residual==0, but Metal wants a binding).
        id<MTLBuffer> cvBuf = residualActive
            ? scratch_.upload(Slot::CoarseVals, coarseVals.data(),
                              coarseVals.size() * sizeof(float))
            : pqnBuf;

        PqParams p;
        p.dim = (uint32_t)dim; p.k = (uint32_t)kRun; p.nprobe = (uint32_t)nprobe;
        p.m = (uint32_t)this->m; p.dsub = (uint32_t)dsub; p.queryCount = (uint32_t)m;
        p.residual = residualActive ? 1u : 0u;

        // Threadgroup sizing: the reduction scratch is (tg/32) * kk entries
        // (simdgroup-first merge), so k no longer constrains tg — always run
        // the widest power-of-two group (the old tgs*k scratch forced tg=32
        // at k=64, measured 5-15x SLOWER than the CPU path).
        const uint32_t kkP = detail::nextPow2K(kRun);
        NSUInteger tg = std::min<NSUInteger>(adcPipe.maxTotalThreadsPerThreadgroup, 256);
        { NSUInteger pw = 32; while (pw * 2 <= tg) pw *= 2; tg = pw; }
        const NSUInteger scratch  = detail::topkScratchBytes(tg, kkP);
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
            lutSpill = scratch_.ensure(Slot::LutSpill,
                                       static_cast<size_t>(chunk) * lutFloats * sizeof(float),
                                       MTLResourceStorageModePrivate);
        }
        p.tgLut = tgLut ? 1u : 0u;

        // commandBufferWithUnretainedReferences: skip per-resource retain/release —
        // measurable at m=1 where the fixed dispatch cost IS the latency. Safe
        // because every bound buffer is index-owned or persistent scratch
        // (GpuScratch.h) and search() awaits completion before returning.
        id<MTLCommandBuffer>         cb  = [queue commandBufferWithUnretainedReferences];
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
    // k > kMaxK is NOT clamped: the GPU ADC's per-thread top-k is a fixed
    // kMaxK-sized register array, so large k simply routes to the exact CPU ADC
    // below (searchCpu serves any k), exactly as FlatIndex/IvfIndex/GraphIndex
    // do. This used to silently truncate — ask for k=100, get 64 back — which
    // reads as a correctness bug, not a documented limit. k=100 is a standard
    // ann-benchmarks configuration.
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

    // GPU-vs-CPU routing.
    //  - kRun > 16: the GPU ADC's per-thread top-k cost grows with
    //    nextPow2(kRun); the multithreaded CPU ADC is measured 1.2-3x faster
    //    (and kRun > kMaxK has no GPU path at all).
    //  - m <= 4: one threadgroup per query cannot amortise the ~2-5 ms dispatch
    //    floor. MEASURED at SIFT1M, m=1, rerank=0 (the only config that reached
    //    the GPU at all): 1.8-6.3 ms on the GPU vs 0.37-0.50 ms on the CPU —
    //    the GPU path was 4-12x SLOWER for a single query. The CPU ADC
    //    parallelises across CELLS at small m, so the whole machine works the
    //    one query (see searchCpu).
    // MFLAT_PQ_CPU=1/0 forces the CPU/GPU path.
    bool cpuRoute = (kRun > 16) || (m <= 4);
    if (const char* e = std::getenv("MFLAT_PQ_CPU")) cpuRoute = (e[0] == '1');
    const bool useGpu = !mImpl->fastScanActive && !cpuRoute
                      && mImpl->adcPipe && mImpl->cq->gpuReady() && kRun <= kMaxK;
    if (useGpu) {
        mImpl->runPqGpu(qPtr, m, kRun, nprobe, pq);
    } else if (mImpl->fastScanActive) {
        searchCpuFS(dim, *mImpl->cq, *mImpl->pq4, mImpl->fsCodes,
                    mImpl->fsBlockStart, mImpl->fsIds,
                    mImpl->residualActive, mImpl->fsPrecomp,
                    qPtr, m, kRun, nprobe, pq);
    } else {
        searchCpu(dim, *mImpl->cq, *mImpl->pq, mImpl->reorderedCodes,
                  mImpl->residualActive, mImpl->precomp,
                  qPtr, m, kRun, nprobe, pq);   // *mImpl->cq/pq = subsystems, pq = out
    }

    if (!doRerank) return pq;

    // ---- exact rerank of the shortlist down to the true top-k -----------
    out.ids.assign(static_cast<size_t>(m) * k, -1);
    out.distances.assign(static_cast<size_t>(m) * k, std::numeric_limits<float>::infinity());
    const std::vector<__fp16>& full = mImpl->fullById;   // fp16 rerank vectors
    const Metric metric = mImpl->metric;
    constexpr int kPref = 8;                              // rerank prefetch depth
    parallelFor(m, [&](int qi) {
        const float* q = qPtr + static_cast<size_t>(qi) * dim;
        const int32_t* sl = &pq.ids[static_cast<size_t>(qi) * kRun];
        std::vector<float> bestScore(k, -std::numeric_limits<float>::infinity());
        std::vector<int>   bestId(k, -1);
        // Prime the gather pipeline: the rerank rows are random 256 B fetches
        // from a ~128 MB array, so run the prefetcher kPref candidates ahead.
        for (int p = 0; p < kPref && p < kRun; ++p)
            if (sl[p] >= 0) __builtin_prefetch(&full[static_cast<size_t>(sl[p]) * dim]);
        for (int sIdx = 0; sIdx < kRun; ++sIdx) {
            if (sIdx + kPref < kRun && sl[sIdx + kPref] >= 0)
                __builtin_prefetch(&full[static_cast<size_t>(sl[sIdx + kPref]) * dim]);
            const int id = sl[sIdx];
            if (id < 0) continue;
            const __fp16* d = &full[static_cast<size_t>(id) * dim];
            // scoreFastH: 4 accumulators over fp16 rows (half the gather traffic;
            // the reference serial-accum path would stall this tight loop).
            const float s = detail::scoreFastH(metric, q, d, dim);
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
            out.distances[o] = detail::scoreToValue(metric, bestScore[src]);
        }
    });
    return out;
}

}  // namespace mflat
