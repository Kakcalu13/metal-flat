// SPDX-License-Identifier: Apache-2.0
// metalflat/include/metalflat/FlatIndex.h
//
// MetalFlat — GPU-accelerated flat (exact, brute-force) vector search
// for Apple Silicon.
//
// "Flat" means no index structure: every query is scored against every
// database vector on the GPU, then the top-k are selected. Exact (100%
// recall) by construction. The right tool for small-to-medium datasets,
// for high-recall needs, and as the baseline every ANN index is judged
// against. (Graph / IVF indexes for large N are the roadmap, not v0.)
//
// This header is pure C++17 — no Metal, no Objective-C — so it drops
// into any C++ project. All GPU code lives behind the pimpl in the .mm.
//
// API shape intentionally mirrors Faiss (add / search) so it's familiar.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "metalflat/Metric.h"

namespace mflat {

struct SearchResult {
    // Row-major: m rows of k. For query r, its j-th nearest neighbour is
    //   ids[r*k + j]      — database index (or -1 when the db holds < k)
    //   distances[r*k + j]— the metric value (squared-L2, or IP/cosine
    //                       similarity). Ordered nearest-first.
    std::vector<int32_t> ids;
    std::vector<float>   distances;
};

class FlatIndex {
public:
    // Largest k the all-GPU selection kernels hold in per-thread registers.
    // Larger k still works and stays exact: search() switches to GPU GEMM
    // distances + multithreaded CPU selection for k > kMaxK.
    static constexpr int kMaxK = 64;

    FlatIndex(int dim, Metric metric);
    ~FlatIndex();

    FlatIndex(const FlatIndex&)            = delete;
    FlatIndex& operator=(const FlatIndex&) = delete;

    // True when a usable Metal device was found. When false, search()
    // still works via an exact CPU fallback — so callers always get
    // correct results, just slower.
    bool ready() const;

    // Append `n` row-major vectors (n * dim floats). For Cosine the
    // stored copies are L2-normalized. O(n*dim) host copy; the GPU
    // buffer is (re)synced lazily on the next search().
    void add(const float* vectors, int n);

    // Drop all database vectors.
    void reset();

    int    size() const;    // number of database vectors
    int    dim() const;
    Metric metric() const;

    // For each of `m` row-major queries (m * dim floats) return the `k`
    // nearest database vectors (any k >= 1; k <= kMaxK stays fully on the
    // GPU, larger k selects on the CPU over GPU-computed distances).
    // Tiny workloads route to an exact multithreaded CPU scan (the GPU
    // dispatch floor dwarfs the work there); MFLAT_FLAT_CPU=1/0 forces
    // the CPU/GPU path. Thread-safety against other const calls is NOT
    // guaranteed in v0 (the GPU buffer syncs lazily) — serialize searches
    // or use one index per thread.
    SearchResult search(const float* queries, int m, int k);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

}  // namespace mflat
