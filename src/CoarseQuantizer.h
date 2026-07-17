// SPDX-License-Identifier: Apache-2.0
// CoarseQuantizer.h — the shared IVF coarse layer (mflat).
//
// Owns the nlist centroids, k-means training, point->cell assignment, the CSR
// inverted-list scaffold (cellStart + reorderedIds permutation), the coarse
// FlatIndex over the centroids, and the GPU cellStart/reorderedIds buffers. The
// per-cell PAYLOAD (fp16 full vectors for IVF, PQ codes for IVFPQ) and the fine
// scan stay in each index; the index reorders its own payload via
// reorderPayload() using this class's permutation.
//
// Metric-agnostic: always operates in L2 over data the caller has already
// metric-normalized (Cosine), exactly as both indexes did before. Internal
// (Obj-C++) header — never included by the public C++ headers.
#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "metalflat/FlatIndex.h"   // FlatIndex, SearchResult, Metric

namespace mflat {

class CoarseQuantizer {
public:
    // Auto           = kmeansGpu when a device is present, else scalar CPU k-means
    //                  (reproduces IvfIndex's device-gated behavior).
    // ForceGpuAssign = always the FlatIndex/GEMM assignment (reproduces
    //                  IvfPqIndex's unconditional kmeansGpu).
    enum class KmeansBackend { Auto, ForceGpuAssign };

    // `device` may be nil (CPU-only: no GPU coarse buffers / coarse FlatIndex).
    CoarseQuantizer(id<MTLDevice> device, int dim);
    ~CoarseQuantizer();
    CoarseQuantizer(const CoarseQuantizer&)            = delete;
    CoarseQuantizer& operator=(const CoarseQuantizer&) = delete;

    // Train (seed 12345), assign, build CSR (cellStart + reorderedIds), and — if
    // a device was given — the coarse FlatIndex(dim,L2) + cell/id GPU buffers.
    // `data` is row-major n×dim, already metric-normalized. nlistReq clamped to n.
    // spherical: renormalize centroids each k-means iteration (Cosine data —
    // unit centroids make L2 assignment match the angular structure).
    void train(const float* data, int n, int nlistReq, int iters = 12,
               KmeansBackend backend = KmeansBackend::Auto,
               bool spherical = false);

    int dim()  const;
    int nlist() const;
    int size() const;                                   // n (dbCount)
    const std::vector<float>& centroids()    const;     // nlist × dim
    // Residual-encoding reference points. Identical to centroids() for plain
    // k-means. Spherical training projects centroids to the unit sphere — right
    // for assignment/probing, but it inflates ||x - c|| and with it the PQ
    // quantization error. So spherical training also stores each cell's exact
    // MEAN (over all assigned points) and residual encoders use that instead:
    // the ADC math is exact for any per-cell reference; the mean minimizes the
    // residual energy the PQ has to spend bits on.
    const std::vector<float>& encodeCentroids() const;
    const std::vector<int>&   cellStart()    const;     // nlist + 1 (CSR offsets)
    const std::vector<int>&   reorderedIds() const;     // n: slot -> original id

    // dst[pos] = src[reorderedIds[pos]] for each of n rows (id order -> slot order).
    // elemBytes = dim*sizeof(float) (IVF float payload) or codeBytes (IVFPQ).
    void reorderPayload(const void* src, void* dst, std::size_t elemBytes) const;

    // Fill `probed` (resized to m*nprobe, nearest-first cell ids). Uses the GPU
    // coarse FlatIndex when nprobe<=FlatIndex::kMaxK, else CPU partial_sort.
    // NOTE: only cell IDS are exposed — the coarse FlatIndex's distances are
    // fp16-derived, so residual-ADC coarse terms must be recomputed exactly
    // (fp32 sqL2 against centroids()) by the caller.
    void probeCells(const float* queries, int m, int nprobe,
                    std::vector<int32_t>& probed) const;
    // Per-query CPU cell selection (writes nprobe cell ids) for reference paths.
    void probeCellsCpu(const float* q, int nprobe, int* outCells) const;

    bool          gpuReady()   const;   // coarse FlatIndex ready (GPU coarse usable)
    id<MTLBuffer> cellBuffer() const;   // cellStart as int32 (nlist+1)
    id<MTLBuffer> idBuffer()   const;   // reorderedIds as int32 (n)

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

}  // namespace mflat
