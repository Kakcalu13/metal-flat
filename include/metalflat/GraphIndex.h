// SPDX-License-Identifier: Apache-2.0
// metalflat/GraphIndex.h — CAGRA-style graph ANN index for Apple Silicon.
//
// Builds a fixed-degree (R) graph — reuse the GPU IVF for an intermediate k-NN,
// RNG/detour-prune it (NSG/DiskANN-style, for greedy navigability), add reverse
// edges — then answers queries with a fixed-iteration beam search on the GPU
// (one threadgroup/query; per-query visited hash; search-width W; early-break).
// A CPU best-first search is the reference/fallback. Targets and takes the
// high-recall / low-latency regime that graph indexes (HNSW) own.
//
// Pure C++17 public surface (Metal hidden behind the pimpl), like the others.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "metalflat/Metric.h"
#include "metalflat/FlatIndex.h"   // SearchResult

namespace mflat {

class GraphIndex {
public:
    static constexpr int kMaxK = 64;

    // R = graph out-degree (neighbours stored per node). Must keep R+1 <= kMaxK
    // so the kNN build stays on the GPU top-k path.
    GraphIndex(int dim, Metric metric, int R = 32);
    ~GraphIndex();
    GraphIndex(const GraphIndex&)            = delete;
    GraphIndex& operator=(const GraphIndex&) = delete;

    bool ready() const;

    // Build the R-degree kNN graph. `nprobe` is the build-quality knob for the
    // internal IVF self-search (higher = better graph, slower build).
    void build(const float* vectors, int n, int nprobe = 64);

    // Beam search. `L` = beam width (the recall/speed knob, >= k); `maxIter` =
    // fixed GPU expansion iterations (<=0 => auto; ignored by the CPU reference,
    // which converges dynamically); `numStart` = random restart seeds;
    // `searchWidth` = nodes expanded per iteration (>1 amortises the per-iteration
    // sort over more candidates → fewer iterations at ~equal recall, capped 16).
    // Workload-adaptive: the GPU beam kernel runs one threadgroup per query, so
    // small batches (m < ~96) route to the CPU best-first search — measured
    // 3-14x lower latency at identical recall (single query ~0.2 ms vs ~3 ms).
    // MFLAT_GRAPH_CPU=1/0 forces the CPU/GPU path.
    SearchResult search(const float* queries, int m, int k,
                        int L = 64, int maxIter = -1, int numStart = 32, int searchWidth = 1);

    // Persist / restore the built index (fp16-free float db + int32 graph). Lets
    // a slow one-time build be reused; load() re-uploads the GPU buffers.
    bool save(const char* path) const;
    bool load(const char* path);

    int dim()    const;
    int size()   const;
    int degree() const;   // R

private:
    struct Impl;
    static void uploadGpu(Impl* m);   // (re)create the GPU db+graph buffers
    static void buildFilter(Impl* m, const float* data, int n);  // traversal filter
    std::unique_ptr<Impl> mImpl;
};

}  // namespace mflat
