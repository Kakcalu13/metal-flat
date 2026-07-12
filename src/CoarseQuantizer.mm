// SPDX-License-Identifier: Apache-2.0
// CoarseQuantizer.mm — shared IVF coarse layer implementation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "CoarseQuantizer.h"
#include "Internal.h"   // detail::{kmeansGpu, parallelFor, sqL2}

namespace mflat {

using namespace detail;   // kmeansGpu, parallelFor, sqL2

namespace {

// Scalar CPU k-means — the no-Metal fallback (was IvfIndex's; single-threaded
// double-accum centroid update, distinct from kmeansGpu's path — kept exact for
// no-Metal parity).
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

struct CoarseQuantizer::Impl {
    id<MTLDevice>              device  = nil;
    int                       dim     = 0;
    int                       nlist   = 0;
    int                       dbCount = 0;
    bool                      gpuReady = false;
    std::vector<float>        centroids;
    std::vector<int>          cellStart;
    std::vector<int>          reorderedIds;
    std::unique_ptr<FlatIndex> coarse;
    id<MTLBuffer>             cellBuf = nil;
    id<MTLBuffer>             idBuf   = nil;
};

CoarseQuantizer::CoarseQuantizer(id<MTLDevice> device, int dim)
    : mImpl(std::make_unique<Impl>()) {
    mImpl->device = device;
    mImpl->dim    = dim;
}
CoarseQuantizer::~CoarseQuantizer() = default;

int CoarseQuantizer::dim()   const { return mImpl->dim; }
int CoarseQuantizer::nlist() const { return mImpl->nlist; }
int CoarseQuantizer::size()  const { return mImpl->dbCount; }
const std::vector<float>& CoarseQuantizer::centroids()    const { return mImpl->centroids; }
const std::vector<int>&   CoarseQuantizer::cellStart()    const { return mImpl->cellStart; }
const std::vector<int>&   CoarseQuantizer::reorderedIds() const { return mImpl->reorderedIds; }
bool          CoarseQuantizer::gpuReady()   const { return mImpl->gpuReady; }
id<MTLBuffer> CoarseQuantizer::cellBuffer() const { return mImpl->cellBuf; }
id<MTLBuffer> CoarseQuantizer::idBuffer()   const { return mImpl->idBuf; }

void CoarseQuantizer::train(const float* data, int n, int nlistReq, int iters,
                            KmeansBackend backend) {
    const int dim   = mImpl->dim;
    const int nlist = std::min(nlistReq, n);
    mImpl->nlist   = nlist;
    mImpl->dbCount = n;

    std::vector<int> assign;
    if (backend == KmeansBackend::ForceGpuAssign || mImpl->device) {
        // Fused GPU assignment (kmeans_assign kernel; GEMM/CPU fallback inside).
        // Training subsamples to 256 points/centroid (faiss convention) — the
        // final all-points assignment pass is unaffected.
        kmeansGpu(data, n, dim, nlist, iters, mImpl->centroids, assign,
                  /*maxPointsPerCentroid=*/256);
    } else {
        // No device: scalar CPU k-means + one assignment pass.
        kmeans(data, n, dim, nlist, iters, mImpl->centroids);
        assign.assign(n, 0);
        parallelFor(n, [&](int i) {
            const float* v = data + static_cast<size_t>(i) * dim;
            float best = std::numeric_limits<float>::infinity();
            int   bestC = 0;
            for (int c = 0; c < nlist; ++c) {
                float dd = sqL2(v, &mImpl->centroids[static_cast<size_t>(c) * dim], dim);
                if (dd < best) { best = dd; bestC = c; }
            }
            assign[i] = bestC;
        });
    }

    // CSR: cell histogram -> prefix sum -> stable slot permutation.
    mImpl->cellStart.assign(nlist + 1, 0);
    for (int i = 0; i < n; ++i) ++mImpl->cellStart[assign[i] + 1];
    for (int c = 0; c < nlist; ++c) mImpl->cellStart[c + 1] += mImpl->cellStart[c];
    mImpl->reorderedIds.assign(n, 0);
    std::vector<int> cursor(mImpl->cellStart.begin(), mImpl->cellStart.end());
    for (int i = 0; i < n; ++i) {
        const int c = assign[i];
        mImpl->reorderedIds[cursor[c]++] = i;
    }

    // GPU coarse quantizer + CSR buffers (only if a device was provided).
    if (mImpl->device) {
        mImpl->coarse = std::make_unique<FlatIndex>(dim, Metric::L2);
        mImpl->coarse->add(mImpl->centroids.data(), nlist);
        mImpl->cellBuf = [mImpl->device
            newBufferWithBytes:mImpl->cellStart.data()
                        length:mImpl->cellStart.size() * sizeof(int32_t)
                       options:MTLResourceStorageModeShared];
        mImpl->idBuf = [mImpl->device
            newBufferWithBytes:mImpl->reorderedIds.data()
                        length:mImpl->reorderedIds.size() * sizeof(int32_t)
                       options:MTLResourceStorageModeShared];
        mImpl->gpuReady = mImpl->coarse->ready();
    }
}

void CoarseQuantizer::reorderPayload(const void* src, void* dst, std::size_t elemBytes) const {
    const std::vector<int>& rid = mImpl->reorderedIds;
    parallelFor(mImpl->dbCount, [&](int pos) {
        std::memcpy(static_cast<char*>(dst) + static_cast<size_t>(pos) * elemBytes,
                    static_cast<const char*>(src) + static_cast<size_t>(rid[pos]) * elemBytes,
                    elemBytes);
    });
}

void CoarseQuantizer::probeCells(const float* queries, int m, int nprobe,
                                 std::vector<int32_t>& probed) const {
    if (mImpl->coarse && nprobe <= FlatIndex::kMaxK) {
        SearchResult c = mImpl->coarse->search(queries, m, nprobe);
        probed = std::move(c.ids);   // m × nprobe cell ids, nearest-first
        return;
    }
    const int dim = mImpl->dim, nlist = mImpl->nlist;
    probed.assign(static_cast<size_t>(m) * nprobe, -1);
    parallelFor(m, [&](int qi) {
        const float* q = queries + static_cast<size_t>(qi) * dim;
        std::vector<std::pair<float, int>> cd(nlist);
        for (int c = 0; c < nlist; ++c)
            cd[c] = {sqL2(q, &mImpl->centroids[static_cast<size_t>(c) * dim], dim), c};
        std::partial_sort(cd.begin(), cd.begin() + nprobe, cd.end());
        int32_t* row = &probed[static_cast<size_t>(qi) * nprobe];
        for (int p = 0; p < nprobe; ++p) row[p] = cd[p].second;
    });
}

void CoarseQuantizer::probeCellsCpu(const float* q, int nprobe, int* outCells) const {
    const int dim = mImpl->dim, nlist = mImpl->nlist;
    std::vector<std::pair<float, int>> cd(nlist);
    for (int c = 0; c < nlist; ++c)
        cd[c] = {sqL2(q, &mImpl->centroids[static_cast<size_t>(c) * dim], dim), c};
    std::partial_sort(cd.begin(), cd.begin() + nprobe, cd.end());
    for (int p = 0; p < nprobe; ++p) outCells[p] = cd[p].second;
}

}  // namespace mflat
