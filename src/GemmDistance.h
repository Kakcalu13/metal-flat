// SPDX-License-Identifier: Apache-2.0
// metal-flat/src/GemmDistance.h  (internal — NOT part of the public API)
//
// Computes the dense query×database dot-product matrix G = Q · Dᵀ on the
// GPU via MetalPerformanceShaders GEMM (Apple's tuned matmul, a large
// fraction of FP32 peak). FlatIndex turns G into final distances + the
// top-k. Single responsibility: the matrix multiply, nothing else.
//
// Objective-C++ header — included only by the .mm side, never by the
// plain-C++ public header.

#pragma once

#import <Metal/Metal.h>

namespace mflat {

class GemmDistance {
public:
    explicit GemmDistance(id<MTLDevice> device);

    // False if MetalPerformanceShaders can't drive this device — caller
    // should fall back (FlatIndex drops to its CPU path).
    bool ready() const { return mReady; }

    // Encode G = Q(m×d) · Dᵀ(n×d)ᵀ  → `out` (m*n floats, row-major) onto
    // `cb`. All buffers are row-major float32; `out` must hold m*n floats.
    // `dbRowOffset` selects an n-row sub-block of the database starting at
    // that row — the hook that lets FlatIndex tile over the database
    // without copying (the block is addressed in place via a buffer
    // offset), so the full m×N score matrix is never materialized.
    // (A transposed D·Qᵀ variant for "coalesced" serial selection was
    // measured strictly slower on M2 — per-thread unit-stride streams
    // already saturate; experiment deleted, don't re-add.)
    void encode(id<MTLCommandBuffer> cb,
                id<MTLBuffer> queries, int m,
                id<MTLBuffer> db, int dbRowOffset, int n, int d,
                id<MTLBuffer> out);

private:
    id<MTLDevice> mDevice;
    bool          mReady = false;
};

}  // namespace mflat
