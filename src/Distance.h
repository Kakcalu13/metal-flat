// SPDX-License-Identifier: Apache-2.0
// Distance.h — central CPU distance / metric primitives (mflat::detail).
//
// Pure C++17, header-only, all `inline`. Includes ONLY the C++ stdlib +
// metalflat/Metric.h — never Metal/Foundation/FlatIndex.h — so EVERY translation
// unit can share it, including FlatIndex.mm (which deliberately avoids Internal.h
// to dodge the FlatIndex-uses-FlatIndex include cycle).
//
// Accumulation types are part of the contract and must not change: dot/sqL2 and
// PQ norms accumulate in float; row norms / normalization accumulate in double.
// Changing them shifts CPU top-k tie-breaks. The GPU kernels (topk_merge,
// ivf_scan, ivfpq_adc) mirror these same formulas.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

#include "metalflat/Metric.h"

namespace mflat {
namespace detail {

// Parallel map over [0, n) across hardware threads. Lives here (not
// Internal.h) so FlatIndex.mm's CPU paths can use it too.
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

// Dot product, float accumulation.
inline float dot(const float* a, const float* b, int dim) {
    float acc = 0.0f;
    for (int c = 0; c < dim; ++c) acc += a[c] * b[c];
    return acc;
}

// Squared L2 distance, float accumulation.
inline float sqL2(const float* a, const float* b, int dim) {
    float acc = 0.0f;
    for (int c = 0; c < dim; ++c) { float e = a[c] - b[c]; acc += e * e; }
    return acc;
}

// ||row||^2 for each of n rows (double accumulation, matching normalizeRows).
inline void rowSqNorms(const float* v, int n, int dim, float* out) {
    for (int i = 0; i < n; ++i) {
        const float* row = v + static_cast<size_t>(i) * dim;
        double s = 0.0;
        for (int c = 0; c < dim; ++c) s += double(row[c]) * row[c];
        out[i] = static_cast<float>(s);
    }
}

// L2-normalize each of n rows in place (double accumulation).
inline void normalizeRows(float* v, int n, int dim) {
    for (int i = 0; i < n; ++i) {
        float* row = v + static_cast<size_t>(i) * dim;
        double s = 0.0;
        for (int c = 0; c < dim; ++c) s += double(row[c]) * row[c];
        if (s > 0.0) {
            const float inv = static_cast<float>(1.0 / std::sqrt(s));
            for (int c = 0; c < dim; ++c) row[c] *= inv;
        }
    }
}
inline void normalizeRows(std::vector<float>& v, int n, int dim) {
    normalizeRows(v.data(), n, dim);
}

// Metric score for a query/db pair: larger = nearer. L2 ranks by -dist^2;
// InnerProduct/Cosine rank by the dot product.
inline float score(Metric m, const float* q, const float* d, int dim) {
    return m == Metric::L2 ? -sqL2(q, d, dim) : dot(q, d, dim);
}
// Convert a ranking score back to the reported distance / similarity value.
inline float scoreToValue(Metric m, float s) { return m == Metric::L2 ? -s : s; }
// The "no neighbour" sentinel distance/similarity for a metric.
inline float emptyValue(Metric m) {
    return scoreToValue(m, -std::numeric_limits<float>::infinity());
}

// CPU mirror of the GEMM L2 identity used in FlatIndex's top-k merge:
//   -max(0, ||q||^2 + ||d||^2 - 2 q·d)   (clamped >= 0 for fp ordering).
inline float l2ScoreFromDot(float qNorm, float dNorm, float dotqd) {
    return -std::fmax(0.0f, qNorm + dNorm - 2.0f * dotqd);
}

// One ADC lookup-table entry for (non-residual) PQ: the per-subspace ||q||^2 is
// dropped (constant per query subspace), so ranking uses ||pqc||^2 - 2 q·pqc.
// For residual PQ the caller passes (q_sub - coarse_centroid_sub) as qSub.
inline float adcLutEntry(float pqNorm, const float* qSub, const float* pqc, int dsub) {
    return pqNorm - 2.0f * dot(qSub, pqc, dsub);
}

}  // namespace detail
}  // namespace mflat
