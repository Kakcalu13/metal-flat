// metalflat/include/metalflat/IvfPqIndex.h
//
// MetalFlat — GPU IVFPQ: inverted-file index with Product Quantization.
//
// Like IvfIndex, the database is clustered into `nlist` cells (coarse
// quantizer) and a query scans only its `nprobe` nearest cells. UNLIKE
// IvfIndex, vectors are not stored in full: each is product-quantized into
// `m` bytes (the vector is split into `m` sub-vectors, each replaced by the
// id of its nearest of 256 learned sub-centroids). A D=128 float32 vector
// (512 B) becomes m=16 bytes — ~32x smaller — so far larger databases fit
// in (unified) memory, and the fine scan is an ADC table lookup (`m` adds
// per candidate) instead of a full distance.
//
// Tradeoff: PQ is lossy, so recall is below IvfIndex/FlatIndex at the same
// nprobe — the classic memory/recall dial. The win is scale: 10M-100M+
// vectors that would not fit (or would be bandwidth-bound) as full floats.
//
// v1 supports Metric::L2 and Metric::Cosine (Cosine = L2 over L2-normalized
// vectors). Pure InnerProduct is not yet supported on this index.
//
// Pure C++17 public surface (Metal hidden behind the pimpl), same as the
// other indexes. SearchResult is shared with FlatIndex.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "metalflat/Metric.h"
#include "metalflat/FlatIndex.h"   // SearchResult

namespace mflat {

class IvfPqIndex {
public:
    // Largest k the GPU path supports (fixed per-thread top-k buffers),
    // matching the other indexes. Requests above this are clamped.
    static constexpr int kMaxK = 64;

    // `m` = number of PQ sub-quantizers = bytes per stored code. Must divide
    // `dim`. Each sub-quantizer has 256 centroids (8-bit codes) in v1.
    IvfPqIndex(int dim, Metric metric, int nlist, int m);
    ~IvfPqIndex();

    IvfPqIndex(const IvfPqIndex&)            = delete;
    IvfPqIndex& operator=(const IvfPqIndex&) = delete;

    // True once build() has trained the quantizers and loaded the codes.
    bool ready() const;

    // Opt-in reranking: when enabled BEFORE build(), the index also keeps the
    // full float vectors so search() can re-rank a PQ shortlist by exact
    // distance (recovering recall lost to quantization — see search()'s
    // `rerank`). Costs the full-vector memory, so it is off by default; for
    // pure compression leave it off. Must be called before build().
    void setRerank(bool enable);

    // Train (coarse k-means + per-subspace PQ k-means), encode every vector
    // to an m-byte code, and arrange the codes into per-cell inverted lists.
    // `n` row-major vectors (n * dim floats). One-time, O(n) memory in codes.
    void build(const float* vectors, int n);

    // For each of `m` queries return the approximate `k` nearest over the
    // `nprobe` nearest cells.
    //
    // `rerank` (0/1 = off): with reranking enabled (setRerank + build), fetch
    // the top-`rerank` candidates by fast PQ-ADC, then re-rank THEM by exact
    // distance and return the true top-k of that shortlist. A modest shortlist
    // (e.g. rerank = 16*k) lifts recall from the raw-PQ ceiling toward ~exact,
    // for a small extra cost. `rerank` may exceed kMaxK. Without reranking the
    // GPU path clamps k to kMaxK; k > kMaxK uses the exact-on-codes CPU path.
    SearchResult search(const float* queries, int m, int k, int nprobe,
                        int rerank = 0);

    int dim() const;
    int size() const;
    int nlist() const;
    int subquantizers() const;   // = m = bytes per code

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

}  // namespace mflat
