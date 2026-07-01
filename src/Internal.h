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
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include "metalflat/FlatIndex.h"   // FlatIndex, SearchResult, Metric
#include "Distance.h"              // dot, sqL2, normalizeRows, score, ... (mflat::detail)

namespace mflat {
namespace detail {

// normalizeRows / sqL2 now live in Distance.h (same namespace); the callers here
// and in the IVF sources resolve them unchanged.

// Parallel map over [0, n) across hardware threads.
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

// k-means whose assignment step runs through a reused FlatIndex (the MPS GEMM
// path on GPU, FlatIndex's exact CPU fallback otherwise). `assignOut` is left
// consistent with the FINAL centroids so callers can use it directly.
inline void kmeansGpu(const float* data, int n, int dim, int nlist, int iters,
                      std::vector<float>& centroids, std::vector<int>& assignOut) {
    centroids.assign(static_cast<size_t>(nlist) * dim, 0.0f);
    std::mt19937 rng(12345);
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int c = 0; c < nlist; ++c)
        std::copy_n(data + static_cast<size_t>(perm[c]) * dim, dim,
                    centroids.begin() + static_cast<size_t>(c) * dim);

    FlatIndex assigner(dim, Metric::L2);
    assignOut.assign(n, 0);
    auto assignNow = [&] {
        assigner.reset();
        assigner.add(centroids.data(), nlist);          // tiny: nlist x dim
        SearchResult r = assigner.search(data, n, 1);    // nearest centroid, GEMM
        for (int i = 0; i < n; ++i) assignOut[i] = r.ids[static_cast<size_t>(i)];
    };
    for (int it = 0; it < iters; ++it) {
        assignNow();
        accumulateCentroids(data, n, dim, nlist, assignOut, centroids, rng);
    }
    assignNow();  // final assignment consistent with the updated centroids
}

// Robust Metal device acquisition — some session contexts return nil from
// MTLCreateSystemDefaultDevice even with a GPU present.
inline id<MTLDevice> acquireMetalDevice() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev) return dev;
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
    return all.count > 0 ? all[0] : nil;
}

}  // namespace detail
}  // namespace mflat
