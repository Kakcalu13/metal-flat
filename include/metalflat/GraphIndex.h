// SPDX-License-Identifier: Apache-2.0
// metalflat/GraphIndex.h — CAGRA-style graph ANN index for Apple Silicon.
//
// Builds a fixed-degree (R) approximate k-NN graph over the database (reusing
// the GPU IVF index as the batched-kNN primitive), then answers queries by
// best-first graph traversal. Targets the high-recall / low-latency regime that
// graph indexes (HNSW) own. v1: GPU build + a reference best-first search; the
// GPU beam-search kernel is layered on next.
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
    // Runs the GPU beam kernel when available, else the CPU best-first reference.
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
    std::unique_ptr<Impl> mImpl;
};

}  // namespace mflat
