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

namespace mflat {

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

// One thread per query. Scans only the query's probed cells (their
// contiguous CSR blocks), computing the metric directly and keeping a
// local top-k by insertion. Cell membership ids come back via
// reorderedIds, so output ids are the caller's original indices. L2
// ranks by -dist², so one path serves all metrics; cosine arrives as
// dot products over already-normalized vectors.
kernel void ivf_scan(
    device const float* reorderedDb  [[buffer(0)]],
    device const int*   reorderedIds [[buffer(1)]],
    device const int*   cellStart    [[buffer(2)]],
    device const float* queries      [[buffer(3)]],
    device const int*   probedCells  [[buffer(4)]],  // queryCount × nprobe
    device int*         outIds       [[buffer(5)]],
    device float*       outVal       [[buffer(6)]],
    constant IvfParams& p            [[buffer(7)]],
    uint                qi           [[thread_position_in_grid]])
{
    if (qi >= p.queryCount) return;
    const uint dim  = p.dim;
    const uint k    = min(p.k, kMaxK);
    const bool isL2 = (p.metric == 0u);
    device const float* q = queries + (uint64_t)qi * dim;

    float bestScore[kMaxK];
    int   bestId[kMaxK];
    for (uint i = 0; i < k; ++i) { bestScore[i] = -INFINITY; bestId[i] = -1; }

    device const int* myCells = probedCells + (uint64_t)qi * p.nprobe;
    for (uint pp = 0; pp < p.nprobe; ++pp) {
        const int cell = myCells[pp];
        if (cell < 0) continue;
        const int lo = cellStart[cell];
        const int hi = cellStart[cell + 1];
        for (int j = lo; j < hi; ++j) {
            device const float* d = reorderedDb + (uint64_t)j * dim;
            float score;
            if (isL2) {
                float acc = 0.0;
                for (uint c = 0; c < dim; ++c) { float e = q[c] - d[c]; acc += e * e; }
                score = -acc;
            } else {
                float acc = 0.0;
                for (uint c = 0; c < dim; ++c) acc += q[c] * d[c];
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

    device int*   oi = outIds + (uint64_t)qi * k;
    device float* ov = outVal + (uint64_t)qi * k;
    for (uint i = 0; i < k; ++i) {
        uint src = k - 1u - i;
        oi[i] = bestId[src];
        ov[i] = isL2 ? -bestScore[src] : bestScore[src];
    }
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

float sqL2(const float* a, const float* b, int dim) {
    float acc = 0.0f;
    for (int c = 0; c < dim; ++c) { float e = a[c] - b[c]; acc += e * e; }
    return acc;
}

template <typename Fn>
void parallelFor(int n, Fn fn) {
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    if (n <= 1 || nt == 1) { for (int i = 0; i < n; ++i) fn(i); return; }
    std::vector<std::thread> pool;
    const int chunk = (n + static_cast<int>(nt) - 1) / static_cast<int>(nt);
    for (unsigned t = 0; t < nt; ++t) {
        const int lo = static_cast<int>(t) * chunk;
        const int hi = std::min(n, lo + chunk);
        if (lo < hi) pool.emplace_back([lo, hi, &fn] {
            for (int i = lo; i < hi; ++i) fn(i);
        });
    }
    for (auto& th : pool) th.join();
}

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

// Same robustness fallback as FlatIndex — some session contexts return
// nil from MTLCreateSystemDefaultDevice even with a GPU present.
id<MTLDevice> acquireMetalDevice() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev) return dev;
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    return all.count > 0 ? all[0] : nil;
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
    id<MTLBuffer>               dbBuf    = nil;   // reorderedDb
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
        fprintf(stderr, "[metalflat] ivf shader compile failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "?");
        return;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"ivf_scan"];
    mImpl->scanPipe = [mImpl->device newComputePipelineStateWithFunction:fn
                                                                   error:&err];
    if (!mImpl->scanPipe) {
        fprintf(stderr, "[metalflat] ivf pipeline build failed: %s\n",
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
    kmeans(data.data(), n, dim, nlist, kIters, mImpl->centroids);

    std::vector<int> assign(n, 0);
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

        mImpl->dbBuf = [mImpl->device
            newBufferWithBytes:mImpl->reorderedDb.data()
                        length:mImpl->reorderedDb.size() * sizeof(float)
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
    if (k > kMaxK) k = kMaxK;
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

    // GPU path needs the coarse k = nprobe within FlatIndex::kMaxK.
    const bool gpuPath = mImpl->gpuReady && mImpl->coarse &&
                         nprobe <= FlatIndex::kMaxK;
    if (!gpuPath) {
        searchCpu(mImpl->centroids, mImpl->cellStart, mImpl->reorderedDb,
                  mImpl->reorderedIds, dim, nlist, mImpl->metric,
                  qPtr, m, k, nprobe, out);
        return out;
    }

    // --- coarse: nprobe nearest centroids per query (exact, on GPU) ---
    // FlatIndex(L2) over the centroids; pre-normalized queries for cosine
    // are passed as-is (L2 over unit vectors = nearest by cosine).
    SearchResult coarse = mImpl->coarse->search(qPtr, m, nprobe);
    // coarse.ids are cell indices (m × nprobe).

    @autoreleasepool {
        id<MTLBuffer> qBuf = [mImpl->device
            newBufferWithBytes:qPtr
                        length:static_cast<size_t>(m) * dim * sizeof(float)
                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> probedBuf = [mImpl->device
            newBufferWithBytes:coarse.ids.data()
                        length:coarse.ids.size() * sizeof(int32_t)
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
        const NSUInteger tg = std::min<NSUInteger>(
            64, mImpl->scanPipe.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(m), 1, 1)
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
