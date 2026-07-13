// SPDX-License-Identifier: Apache-2.0
// Internal.h — helpers shared by the IVF / IVFPQ implementations.
//
// NOT part of the public API. Included only by the Objective-C++ (.mm) sources
// (it uses Metal types), so everything is `inline` to satisfy the ODR across
// translation units. Lives in mflat::detail.

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "metalflat/FlatIndex.h"   // FlatIndex, SearchResult, Metric
#include "Distance.h"              // dot, sqL2, normalizeRows, score, ... (mflat::detail)
#include "Log_internal.h"          // MFLAT_LOG_* (kmeans_assign compile diagnostics)

namespace mflat {
namespace detail {

// normalizeRows / sqL2 / parallelFor now live in Distance.h (same namespace);
// the callers here and in the IVF sources resolve them unchanged.

// Recompute centroids = mean of assigned points. Per-thread partial sums then a
// merge — avoids write contention / the single-threaded O(N*D)/iter floor.
// Empty cells are reseeded from a random point.
inline void accumulateCentroids(const float* data, int n, int dim, int nlist,
                                const std::vector<int>& assign,
                                std::vector<float>& centroids, std::mt19937& rng) {
    const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::vector<double>> tsums(
        nt, std::vector<double>(static_cast<size_t>(nlist) * dim, 0.0));
    std::vector<std::vector<int>> tcnt(nt, std::vector<int>(nlist, 0));
    const int chunk = (n + static_cast<int>(nt) - 1) / static_cast<int>(nt);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < nt; ++t) {
        const int lo = static_cast<int>(t) * chunk;
        const int hi = std::min(n, lo + chunk);
        if (lo >= hi) continue;
        pool.emplace_back([&, t, lo, hi] {
            auto& s = tsums[t]; auto& cnt = tcnt[t];
            for (int i = lo; i < hi; ++i) {
                const float* v = data + static_cast<size_t>(i) * dim;
                const int c = assign[i];
                double* sc = &s[static_cast<size_t>(c) * dim];
                for (int d = 0; d < dim; ++d) sc[d] += v[d];
                ++cnt[c];
            }
        });
    }
    for (auto& th : pool) th.join();
    parallelFor(nlist, [&](int c) {
        long cnt = 0;
        for (unsigned t = 0; t < nt; ++t) cnt += tcnt[t][c];
        float* ce = &centroids[static_cast<size_t>(c) * dim];
        if (cnt > 0) {
            for (int d = 0; d < dim; ++d) {
                double acc = 0.0;
                for (unsigned t = 0; t < nt; ++t)
                    acc += tsums[t][static_cast<size_t>(c) * dim + d];
                ce[d] = static_cast<float>(acc / static_cast<double>(cnt));
            }
        }
    });
    for (int c = 0; c < nlist; ++c) {
        long cnt = 0;
        for (unsigned t = 0; t < nt; ++t) cnt += tcnt[t][c];
        if (cnt == 0) {
            const int r = static_cast<int>(rng() % static_cast<unsigned>(n));
            std::copy_n(data + static_cast<size_t>(r) * dim, dim,
                        &centroids[static_cast<size_t>(c) * dim]);
        }
    }
}

// Robust Metal device acquisition — some session contexts return nil from
// MTLCreateSystemDefaultDevice even with a GPU present.
inline id<MTLDevice> acquireMetalDevice() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev) return dev;
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    return all.count > 0 ? all[0] : nil;
}

// Fused k-means assignment kernel for dim <= 128 (rows zero-padded to a
// register-friendly kD4 by the host) — the common IVF / PQ-subspace shapes.
// One dispatch computes, for every point, the argmin over ALL centroids of
// the squared L2 distance, replacing the per-iteration FlatIndex GEMM +
// top-k merge pair, whose m×tileW score tile round-tripped through device
// memory (measured 443 ms GPU per pass on SIFT1M/nlist=1000 vs 230 ms
// fused). Larger dims stay on the GEMM path: once the query no longer fits
// in registers, GEMM's data reuse wins (a staged-tile fused kernel measured
// 1.2-15x SLOWER than GEMM across dim 132-768).
//
// kD4 = dim/4 is a function constant, so the query array is compile-time
// sized: each thread holds its WHOLE query in registers and scans centroid
// rows straight from device memory — every SIMD lane reads the same address
// in lockstep, so the load is a broadcast served from L2 (the full centroid
// table is at most 2 MB). Eight rows per iteration keep eight independent
// accumulator chains in flight: one chain is FMA-latency-bound (measured
// 1085 ms for SIFT1M x 1000 cells; 8 chains: 230 ms). No cross-thread reduce,
// no threadgroup memory: each thread owns its argmin. Direct (q−c)² fp32
// accumulation; ties keep the lowest centroid index (ascending scan), the
// same first-seen convention as topk_merge.
inline NSString* const kKmeansAssignSrc = @R"(
#include <metal_stdlib>
using namespace metal;

struct AssignParams { uint n; uint dim; uint C; };

constant uint kD4 [[function_constant(0)]];

kernel void kmeans_assign_reg(
    device const float4*  points [[buffer(0)]],   // n × kD4
    device const float4*  cents  [[buffer(1)]],   // C × kD4
    device int*           outA   [[buffer(2)]],   // n
    constant AssignParams& p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float4 q[32];   // kD4 <= 32; unused tail is compiled away
    for (uint c = 0; c < kD4; ++c) q[c] = points[(uint64_t)gid * kD4 + c];

    float best  = INFINITY;
    int   bestC = 0;
    uint  r     = 0;
    for (; r + 7 < p.C; r += 8) {
        device const float4* cr = cents + (uint64_t)r * kD4;
        float4 a0 = float4(0.0), a1 = float4(0.0), a2 = float4(0.0), a3 = float4(0.0);
        float4 a4 = float4(0.0), a5 = float4(0.0), a6 = float4(0.0), a7 = float4(0.0);
        for (uint c = 0; c < kD4; ++c) {
            const float4 qc = q[c];
            float4 e;
            e = qc - cr[c];            a0 += e * e;
            e = qc - cr[c +     kD4];  a1 += e * e;
            e = qc - cr[c + 2 * kD4];  a2 += e * e;
            e = qc - cr[c + 3 * kD4];  a3 += e * e;
            e = qc - cr[c + 4 * kD4];  a4 += e * e;
            e = qc - cr[c + 5 * kD4];  a5 += e * e;
            e = qc - cr[c + 6 * kD4];  a6 += e * e;
            e = qc - cr[c + 7 * kD4];  a7 += e * e;
        }
        const float s[8] = {a0.x + a0.y + a0.z + a0.w, a1.x + a1.y + a1.z + a1.w,
                            a2.x + a2.y + a2.z + a2.w, a3.x + a3.y + a3.z + a3.w,
                            a4.x + a4.y + a4.z + a4.w, a5.x + a5.y + a5.z + a5.w,
                            a6.x + a6.y + a6.z + a6.w, a7.x + a7.y + a7.z + a7.w};
        for (uint j = 0; j < 8; ++j)
            if (s[j] < best) { best = s[j]; bestC = (int)(r + j); }
    }
    for (; r < p.C; ++r) {   // tail rows (C % 8)
        device const float4* cr = cents + (uint64_t)r * kD4;
        float4 acc = float4(0.0);
        for (uint c = 0; c < kD4; ++c) { const float4 e = q[c] - cr[c]; acc += e * e; }
        const float s = acc.x + acc.y + acc.z + acc.w;
        if (s < best) { best = s; bestC = (int)r; }
    }
    outA[gid] = bestC;
}
)";

// CPU mirror of the kernel's AssignParams, field-for-field.
struct KmeansAssignParams { uint32_t n, dim, C; };

// Process-wide pipeline cache (kmeansGpu is called once per IVF train but m
// times per PQ train), keyed by (device, dim/4) — the kernel is specialized
// per dim via the kD4 function constant.
inline id<MTLComputePipelineState> kmeansAssignPipeline(id<MTLDevice> dev, uint32_t d4) {
    struct Entry {
        void*                       dev;
        uint32_t                    d4;
        id<MTLComputePipelineState> pipe;
    };
    static std::mutex mu;
    static std::vector<Entry>& cache = *new std::vector<Entry>();
    static void*         libDev = nullptr;
    static id<MTLLibrary> lib   = nil;
    std::lock_guard<std::mutex> lock(mu);
    for (const Entry& e : cache)
        if (e.dev == (__bridge void*)dev && e.d4 == d4) return e.pipe;

    NSError* err = nil;
    if (libDev != (__bridge void*)dev) {
        lib = [dev newLibraryWithSource:kKmeansAssignSrc options:nil error:&err];
        libDev = (__bridge void*)dev;
    }
    id<MTLComputePipelineState> pipe = nil;
    if (lib) {
        MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
        [cv setConstantValue:&d4 type:MTLDataTypeUInt atIndex:0];
        id<MTLFunction> fn = [lib newFunctionWithName:@"kmeans_assign_reg"
                                       constantValues:cv
                                                error:&err];
        if (fn) pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
    }
    if (!pipe)
        MFLAT_LOG_WARN("kmeans_assign pipeline unavailable (GEMM fallback): %s",
                       err ? [[err localizedDescription] UTF8String] : "?");
    cache.push_back({(__bridge void*)dev, d4, pipe});
    return pipe;
}

// k-means with a GPU-fused assignment step (kmeans_assign above; FlatIndex
// GEMM fallback when Metal or the tile budget is unavailable). The training
// set and assignment buffers are uploaded/allocated ONCE and reused across
// all iterations — no per-pass copies. `maxPointsPerCentroid` > 0 subsamples
// training to min(n, maxPointsPerCentroid*nlist) points (faiss convention;
// 0 trains on all n). The FINAL assignment always covers all n points, so
// `assignOut` stays consistent with the returned centroids.
inline void kmeansGpu(const float* data, int n, int dim, int nlist, int iters,
                      std::vector<float>& centroids, std::vector<int>& assignOut,
                      int maxPointsPerCentroid = 0) {
    centroids.assign(static_cast<size_t>(nlist) * dim, 0.0f);
    std::mt19937 rng(12345);
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    // nlist > n (e.g. PQ's fixed 256 sub-centroids on a tiny training set):
    // wrap the permutation — duplicate seeds are degenerate but defined, and
    // accumulateCentroids' empty-cluster reseed separates them over iterations.
    for (int c = 0; c < nlist; ++c)
        std::copy_n(data + static_cast<size_t>(perm[c % n]) * dim, dim,
                    centroids.begin() + static_cast<size_t>(c) * dim);

    // Training subsample: the first trainN entries of the (uniform) shuffle,
    // gathered contiguous. Seeds are perm[0..nlist), so they are members of
    // the training set whenever it is at least nlist long.
    const float* trainPtr = data;
    int trainN = n;
    std::vector<float> trainVec;
    if (maxPointsPerCentroid > 0 &&
        static_cast<int64_t>(maxPointsPerCentroid) * nlist < n) {
        trainN = maxPointsPerCentroid * nlist;
        trainVec.resize(static_cast<size_t>(trainN) * dim);
        parallelFor(trainN, [&](int i) {
            std::copy_n(data + static_cast<size_t>(perm[i]) * dim, dim,
                        &trainVec[static_cast<size_t>(i) * dim]);
        });
        trainPtr = trainVec.data();
    }

    assignOut.assign(n, 0);

    // Fused GPU path — dim <= 128 only (see the kernel doc: GEMM wins beyond
    // that). Rows are zero-padded to padDim so kD4 lands where the Metal
    // compiler keeps q[] in registers: kD4 <= 8 and kD4 >= 28 compile clean;
    // kD4 in [9, 27] spills q[] to thread memory (measured 10-30x slower,
    // maxTotalThreadsPerThreadgroup 640 -> 384). Zero padding leaves the
    // (q-c)^2 argmin exact, and lets dim % 4 != 0 shapes use this path too.
    const int padDim = (dim <= 32)  ? ((dim + 3) & ~3)
                     : (dim <= 128) ? std::max(112, (dim + 3) & ~3)
                                    : 0;
    id<MTLDevice> dev = (padDim > 0) ? acquireMetalDevice() : nil;
    id<MTLComputePipelineState> pipe =
        dev ? kmeansAssignPipeline(dev, static_cast<uint32_t>(padDim / 4)) : nil;
    if (pipe) {
        bool gpuDone = false;
        @autoreleasepool {
            id<MTLCommandQueue> queue = [dev newCommandQueue];
            // Row-padded upload (plain newBufferWithBytes when no padding).
            auto pointsBuf = [&](const float* src, int m) -> id<MTLBuffer> {
                if (padDim == dim)
                    return [dev newBufferWithBytes:src
                                            length:static_cast<size_t>(m) * dim * sizeof(float)
                                           options:MTLResourceStorageModeShared];
                id<MTLBuffer> b = [dev
                    newBufferWithLength:static_cast<size_t>(m) * padDim * sizeof(float)
                                options:MTLResourceStorageModeShared];
                if (b) {
                    float* dst = static_cast<float*>([b contents]);
                    parallelFor(m, [&](int i) {
                        std::memcpy(dst + static_cast<size_t>(i) * padDim,
                                    src + static_cast<size_t>(i) * dim,
                                    static_cast<size_t>(dim) * sizeof(float));
                        std::memset(dst + static_cast<size_t>(i) * padDim + dim, 0,
                                    static_cast<size_t>(padDim - dim) * sizeof(float));
                    });
                }
                return b;
            };
            id<MTLBuffer> fullBuf  = pointsBuf(data, n);
            id<MTLBuffer> trainBuf = (trainPtr == data) ? fullBuf
                                                        : pointsBuf(trainPtr, trainN);
            id<MTLBuffer> centBuf = [dev
                newBufferWithLength:static_cast<size_t>(nlist) * padDim * sizeof(float)
                            options:MTLResourceStorageModeShared];
            id<MTLBuffer> assignBuf = [dev
                newBufferWithLength:static_cast<size_t>(n) * sizeof(int32_t)
                            options:MTLResourceStorageModeShared];
            const NSUInteger tgs =
                std::min<NSUInteger>(pipe.maxTotalThreadsPerThreadgroup, 256);
            if (fullBuf && trainBuf && centBuf && assignBuf) {
                if (padDim != dim)   // pad columns stay zero across iterations
                    std::memset([centBuf contents], 0,
                                static_cast<size_t>(nlist) * padDim * sizeof(float));

                auto assignPass = [&](id<MTLBuffer> pts, int m) {
                    float* cdst = static_cast<float*>([centBuf contents]);
                    if (padDim == dim) {
                        std::memcpy(cdst, centroids.data(),
                                    static_cast<size_t>(nlist) * dim * sizeof(float));
                    } else {
                        for (int c = 0; c < nlist; ++c)
                            std::memcpy(cdst + static_cast<size_t>(c) * padDim,
                                        &centroids[static_cast<size_t>(c) * dim],
                                        static_cast<size_t>(dim) * sizeof(float));
                    }
                    KmeansAssignParams p;
                    p.n = static_cast<uint32_t>(m);
                    p.dim = static_cast<uint32_t>(padDim);
                    p.C = static_cast<uint32_t>(nlist);
                    id<MTLCommandBuffer>         cb  = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:pipe];
                    [enc setBuffer:pts       offset:0 atIndex:0];
                    [enc setBuffer:centBuf   offset:0 atIndex:1];
                    [enc setBuffer:assignBuf offset:0 atIndex:2];
                    [enc setBytes:&p length:sizeof(p) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake((m + tgs - 1) / tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
                    [enc endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                };

                std::vector<int> a(trainN);
                for (int it = 0; it < iters; ++it) {
                    assignPass(trainBuf, trainN);
                    std::memcpy(a.data(), [assignBuf contents],
                                static_cast<size_t>(trainN) * sizeof(int32_t));
                    accumulateCentroids(trainPtr, trainN, dim, nlist, a, centroids, rng);
                }
                assignPass(fullBuf, n);   // final assignment, all points
                std::memcpy(assignOut.data(), [assignBuf contents],
                            static_cast<size_t>(n) * sizeof(int32_t));
                gpuDone = true;
            }   // buffers allocated; else fall through to the FlatIndex path
        }
        if (gpuDone) return;
    }

    // FlatIndex path: MPS GEMM assignment (exact CPU when no Metal). Serves
    // dim > 128 — where GEMM's data reuse beats a fused kernel — and any
    // Metal/allocation failure above.
    FlatIndex assigner(dim, Metric::L2);
    auto assignNow = [&](const float* pts, int m, std::vector<int>& out) {
        assigner.reset();
        assigner.add(centroids.data(), nlist);          // tiny: nlist x dim
        SearchResult r = assigner.search(pts, m, 1);    // nearest centroid
        for (int i = 0; i < m; ++i) out[i] = r.ids[static_cast<size_t>(i)];
    };
    std::vector<int> a(trainN);
    for (int it = 0; it < iters; ++it) {
        assignNow(trainPtr, trainN, a);
        accumulateCentroids(trainPtr, trainN, dim, nlist, a, centroids, rng);
    }
    assignNow(data, n, assignOut);  // final assignment consistent with centroids
}

}  // namespace detail
}  // namespace mflat
