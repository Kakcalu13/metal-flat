// SPDX-License-Identifier: Apache-2.0
// metal-flat/include/metalflat/IvfIndex.h
//
// IVF (inverted file) approximate nearest-neighbour index.
//
// Clusters the database into `nlist` cells via k-means. A query scans
// only its `nprobe` nearest cells instead of the whole database — doing
// roughly nlist/nprobe times less work. That trades exactness (recall
// < 1.0, tunable via nprobe) for the large speedup that exact flat
// search physically cannot reach (flat must touch every pair).
//
// Pure C++17 public header (Metal hidden in the .mm). Reuses Metric and
// SearchResult from the flat API.
//
// v1: CPU k-means + CPU two-stage search — correct, the recall
// reference. The GPU fine-scan kernel is the next stage.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "metalflat/Metric.h"
#include "metalflat/FlatIndex.h"   // SearchResult

namespace mflat {

class IvfIndex {
public:
    static constexpr int kMaxK = 64;

    // `nlist` = number of cells (clusters). Rule of thumb: ~√N to a few×
    // √N. More cells = smaller cells = faster but needs higher nprobe
    // for the same recall. Clamped to the database size at build time.
    IvfIndex(int dim, Metric metric, int nlist);
    ~IvfIndex();

    IvfIndex(const IvfIndex&)            = delete;
    IvfIndex& operator=(const IvfIndex&) = delete;

    // Train k-means on the data, then assign every vector to its cell.
    // (v1 combines train + add; Faiss separates them.) For Cosine the
    // stored vectors are L2-normalized. O(n · nlist · dim · iters) — a
    // one-time build cost.
    void build(const float* vectors, int n);

    // The k nearest of `m` queries, scanning each query's `nprobe`
    // nearest cells. `nprobe` is clamped to [1, nlist] (nprobe == nlist
    // degenerates to exact flat search); `k` to [1, kMaxK]. Recall rises
    // with nprobe, speed falls — that's the knob.
    SearchResult search(const float* queries, int m, int k, int nprobe);

    int  dim()   const;
    int  size()  const;   // number of database vectors
    int  nlist() const;
    bool ready() const;   // true once build() has produced cells

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

}  // namespace mflat
