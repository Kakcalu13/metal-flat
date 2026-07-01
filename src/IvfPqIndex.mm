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
//            Then each candidate's distance is m table lookups + adds; keep
//            top-k; tree-merge across the threadgroup.
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
};

NSString* const kShaderSrc = @R"(
#include <metal_stdlib>
using namespace metal;

struct PqParams { uint dim; uint k; uint nprobe; uint m; uint dsub; uint queryCount; };
constant uint kMaxK = 64;
constant uint kKsub = 256;

// One threadgroup per query. Build the per-query ADC table once, then scan
// the probed cells' codes by table lookup, keeping a top-k per thread, then
// tree-merge. Distances rank by score = -dist (larger = nearer).
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
    threadgroup float*  lut        [[threadgroup(0)]],   // m × 256
    threadgroup float*  redScore   [[threadgroup(1)]],   // tgs × k
    threadgroup int*    redId      [[threadgroup(2)]],   // tgs × k
    uint qi  [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    if (qi >= p.queryCount) return;
    const uint M = p.m, dsub = p.dsub, dim = p.dim;
    const uint k = min(p.k, kMaxK);
    device const float* q = queries + (uint64_t)qi * dim;

    // Build the per-query lookup table once (cell-independent for non-residual
    // PQ): lut[mm*256 + j] = ||pqc[mm][j]||^2 - 2 q_mm·pqc[mm][j].
    const uint lutN = M * kKsub;
    for (uint e = tid; e < lutN; e += tgs) {
        const uint mm = e / kKsub;
        const uint j  = e % kKsub;
        device const float* sub = pqc + ((uint64_t)mm * kKsub + j) * dsub;
        device const float* qm  = q + (uint64_t)mm * dsub;
        float dot = 0.0;
        for (uint d = 0; d < dsub; ++d) dot += qm[d] * sub[d];
        lut[e] = pqNorm[e] - 2.0 * dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bestScore[kMaxK];
    int   bestId[kMaxK];
    for (uint i = 0; i < k; ++i) { bestScore[i] = -INFINITY; bestId[i] = -1; }

    device const int* myCells = probed + (uint64_t)qi * p.nprobe;
    for (uint pp = 0; pp < p.nprobe; ++pp) {
        const int cell = myCells[pp];
        if (cell < 0) continue;
        const int lo = cellStart[cell];
        const int hi = cellStart[cell + 1];
        for (int c = lo + (int)tid; c < hi; c += (int)tgs) {
            device const uchar* code = codes + (uint64_t)c * M;
            float dist = 0.0;
            for (uint mm = 0; mm < M; ++mm) dist += lut[mm * kKsub + code[mm]];
            const float score = -dist;
            if (score > bestScore[0]) {
                uint pos = 0;
                while (pos + 1u < k && score > bestScore[pos + 1u]) {
                    bestScore[pos] = bestScore[pos + 1u];
                    bestId[pos]    = bestId[pos + 1u];
                    ++pos;
                }
                bestScore[pos] = score;
                bestId[pos]    = ids[c];
            }
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
        float qn = 0.0;
        for (uint d = 0; d < dim; ++d) qn += q[d] * q[d];   // report approx ||q-x||^2
        device int*   oi = outIds + (uint64_t)qi * k;
        device float* ov = outVal + (uint64_t)qi * k;
        for (uint i = 0; i < k; ++i) {
            uint src = k - 1u - i;
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

    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLComputePipelineState> adcPipe  = nil;
    id<MTLBuffer>               codesBuf = nil;
    id<MTLBuffer>               pqcBuf   = nil;
    id<MTLBuffer>               pqnBuf   = nil;

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

void IvfPqIndex::setRerank(bool enable) { mImpl->storeFull = enable; }

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

    // PQ training (codes come straight from the sub-space k-means assignment)
    // and the coarse quantizer (k-means + CSR + coarse FlatIndex + cell/id bufs).
    std::vector<uint8_t> codes(static_cast<size_t>(n) * M);
    mImpl->pq->train(data.data(), n, kIters, &codes);
    mImpl->cq->train(data.data(), n, mImpl->nlist, kIters,
                     CoarseQuantizer::KmeansBackend::ForceGpuAssign);
    mImpl->nlist = mImpl->cq->nlist();

    // Reorder the m-byte codes into CSR slot order (id order -> slots).
    mImpl->reorderedCodes.assign(static_cast<size_t>(n) * M, 0);
    mImpl->cq->reorderPayload(codes.data(), mImpl->reorderedCodes.data(),
                              static_cast<size_t>(M));

    // Keep full vectors (original id order) for exact reranking, if enabled.
    if (mImpl->storeFull) mImpl->fullById = data;

    mImpl->dbCount = n;
    mImpl->ready   = true;

    // GPU setup: upload codes + PQ tables (cell/id buffers live in the CQ).
    if (mImpl->adcPipe && mImpl->cq->gpuReady()) {
        auto buf = [&](const void* p, size_t bytes) {
            return [mImpl->device newBufferWithBytes:p length:bytes
                                             options:MTLResourceStorageModeShared];
        };
        mImpl->codesBuf = buf(mImpl->reorderedCodes.data(), mImpl->reorderedCodes.size());
        mImpl->pqcBuf   = buf(mImpl->pq->centroids().data(), mImpl->pq->centroids().size() * sizeof(float));
        mImpl->pqnBuf   = buf(mImpl->pq->norms().data(),     mImpl->pq->norms().size() * sizeof(float));
    }
}

namespace {

// CPU ADC fallback (also the recall reference): exact over the stored codes.
// Takes raw fields (not the private Impl) so it stays a free function.
void searchCpu(int dim, const CoarseQuantizer& cq,
               const detail::PqTrainer&    pq,
               const std::vector<uint8_t>& reorderedCodes,
               const float* qPtr, int m, int k, int nprobe, SearchResult& out) {
    const int M = pq.codeBytes(), ksub = pq.ksub();
    const std::vector<int>& cellStart    = cq.cellStart();
    const std::vector<int>& reorderedIds = cq.reorderedIds();
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
            for (int j = cellStart[c]; j < cellStart[c + 1]; ++j) {
                const uint8_t* code = &reorderedCodes[static_cast<size_t>(j) * M];
                float dist = 0.0f;
                for (int mm = 0; mm < M; ++mm) dist += lut[static_cast<size_t>(mm) * ksub + code[mm]];
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
        const float qn = dot(q, q, dim);
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

        PqParams p;
        p.dim = (uint32_t)dim; p.k = (uint32_t)kRun; p.nprobe = (uint32_t)nprobe;
        p.m = (uint32_t)this->m; p.dsub = (uint32_t)dsub; p.queryCount = (uint32_t)m;

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
        [enc setBytes:&p length:sizeof(p) atIndex:9];

        const NSUInteger kk  = static_cast<NSUInteger>(kRun);
        const NSUInteger lutBytes = static_cast<NSUInteger>(this->m) * kKsub * sizeof(float);
        NSUInteger memTg = (lutBytes + 256 < 32000)
            ? (32000 - lutBytes - 256) / (kk * (sizeof(float) + sizeof(int)))
            : 1;
        NSUInteger tg = std::min<NSUInteger>(adcPipe.maxTotalThreadsPerThreadgroup, 256);
        tg = std::min<NSUInteger>(tg, std::max<NSUInteger>(memTg, 32));
        NSUInteger pw = 32; while (pw * 2 <= tg) pw *= 2; tg = pw;

        [enc setThreadgroupMemoryLength:lutBytes atIndex:0];
        [enc setThreadgroupMemoryLength:tg * kk * sizeof(float) atIndex:1];
        [enc setThreadgroupMemoryLength:tg * kk * sizeof(int)   atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(m), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
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

    const bool doRerank = rerank > k && mImpl->storeFull && !mImpl->fullById.empty();
    const int  kRun = doRerank ? std::min(rerank, mImpl->dbCount) : k;

    // ---- PQ shortlist (kRun-NN by ADC) ----------------------------------
    SearchResult pq;
    pq.ids.assign(static_cast<size_t>(m) * kRun, -1);
    pq.distances.assign(static_cast<size_t>(m) * kRun, std::numeric_limits<float>::infinity());

    const bool useGpu = mImpl->adcPipe && mImpl->cq->gpuReady() && kRun <= kMaxK;
    if (useGpu) {
        mImpl->runPqGpu(qPtr, m, kRun, nprobe, pq);
    } else {
        searchCpu(dim, *mImpl->cq, *mImpl->pq, mImpl->reorderedCodes,
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
