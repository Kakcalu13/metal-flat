// SPDX-License-Identifier: Apache-2.0
// Opq.h — Optimized Product Quantization rotation trainer (mflat::detail).
//
// Learns an orthogonal dim×dim rotation R that minimizes PQ reconstruction
// error (Ge et al., "Optimized Product Quantization", CVPR 2013), by
// alternating minimization:
//   1. fix R: train a PQ on the rotated sample Y = R·X, reconstruct Y_hat;
//   2. fix the PQ: solve the orthogonal Procrustes problem
//        min_{R orthogonal} ||R·X - Y_hat||_F
//      whose solution is R = U·V^T from the SVD  Y_hat·X^T = U·Σ·V^T.
// The SVD is a one-sided Jacobi (Hestenes) in double precision — pure C++,
// no dependencies, exact enough for dim <= 1024 (typical 128).
//
// Internal, Obj-C++-only (PqTrainer pulls Internal.h). Applied as a
// pre-transform: the whole IVFPQ pipeline (coarse + residual + PQ) then runs
// in the rotated space; orthogonality preserves L2/IP, so search results are
// reported in the original metric unchanged.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "Internal.h"      // parallelFor + Distance.h
#include "Log_internal.h"
#include "PqTrainer.h"

namespace mflat {
namespace detail {

// Apply the row-major dim×dim rotation R to n row-vectors: dst_i = R · src_i.
// dst must not alias src.
inline void applyRotation(const float* R, const float* src, float* dst,
                          int n, int dim) {
    parallelFor(n, [&](int i) {
        const float* x = src + static_cast<size_t>(i) * dim;
        float*       y = dst + static_cast<size_t>(i) * dim;
        for (int r = 0; r < dim; ++r)
            y[r] = dot(R + static_cast<size_t>(r) * dim, x, dim);
    });
}

// Orthogonal polar factor R = U·V^T of the d×d matrix M (row-major), via
// one-sided Jacobi SVD in double precision. Returns false when M is (near)
// rank-deficient — the caller should then skip this rotation update.
inline bool polarOrthogonal(const std::vector<float>& M, int d, std::vector<float>& R) {
    // Column-major workspaces: G starts as M and converges to U·Σ (columns
    // orthogonal); V accumulates the right rotations so that M = G·V^T.
    std::vector<double> G(static_cast<size_t>(d) * d), V(static_cast<size_t>(d) * d, 0.0);
    for (int r = 0; r < d; ++r)
        for (int c = 0; c < d; ++c)
            G[static_cast<size_t>(c) * d + r] = M[static_cast<size_t>(r) * d + c];
    for (int i = 0; i < d; ++i) V[static_cast<size_t>(i) * d + i] = 1.0;

    const double eps = 1e-14;
    for (int sweep = 0; sweep < 60; ++sweep) {
        bool rotated = false;
        for (int p = 0; p < d - 1; ++p) {
            for (int q = p + 1; q < d; ++q) {
                double* gp = &G[static_cast<size_t>(p) * d];
                double* gq = &G[static_cast<size_t>(q) * d];
                double a = 0.0, b = 0.0, c = 0.0;
                for (int i = 0; i < d; ++i) { a += gp[i] * gp[i]; b += gq[i] * gq[i]; c += gp[i] * gq[i]; }
                if (c * c <= eps * eps * a * b) continue;
                rotated = true;
                const double theta = 0.5 * std::atan2(2.0 * c, a - b);
                const double cs = std::cos(theta), sn = std::sin(theta);
                double* vp = &V[static_cast<size_t>(p) * d];
                double* vq = &V[static_cast<size_t>(q) * d];
                for (int i = 0; i < d; ++i) {
                    const double gpi = gp[i], gqi = gq[i];
                    gp[i] = cs * gpi + sn * gqi;
                    gq[i] = -sn * gpi + cs * gqi;
                    const double vpi = vp[i], vqi = vq[i];
                    vp[i] = cs * vpi + sn * vqi;
                    vq[i] = -sn * vpi + cs * vqi;
                }
            }
        }
        if (!rotated) break;
    }

    // Singular values = column norms of G; guard rank deficiency.
    std::vector<double> sigma(d);
    double sigMax = 0.0;
    for (int i = 0; i < d; ++i) {
        double s = 0.0;
        const double* gi = &G[static_cast<size_t>(i) * d];
        for (int r = 0; r < d; ++r) s += gi[r] * gi[r];
        sigma[i] = std::sqrt(s);
        sigMax = std::max(sigMax, sigma[i]);
    }
    for (int i = 0; i < d; ++i)
        if (sigma[i] < 1e-9 * sigMax) return false;

    // R = U·V^T with U = G·diag(1/σ):  R[r][c] = Σ_i (G[i][r]/σ_i)·V[i][c].
    R.assign(static_cast<size_t>(d) * d, 0.0f);
    parallelFor(d, [&](int r) {
        for (int c = 0; c < d; ++c) {
            double acc = 0.0;
            for (int i = 0; i < d; ++i)
                acc += (G[static_cast<size_t>(i) * d + r] / sigma[i]) * V[static_cast<size_t>(i) * d + c];
            R[static_cast<size_t>(r) * d + c] = static_cast<float>(acc);
        }
    });
    return true;
}

// Random orthogonal dim×dim rotation (row-major): N(0,1) entries orthonormalized
// with two passes of modified Gram-Schmidt in double ("twice is enough").
// Random init beats identity for OPQ — identity risks a local optimum when the
// variance concentrates on axes already aligned with the subspace split.
inline std::vector<float> randomRotation(int dim, uint32_t seed = 12345) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<double> A(static_cast<size_t>(dim) * dim);
    for (auto& x : A) x = gauss(rng);
    for (int pass = 0; pass < 2; ++pass) {
        for (int r = 0; r < dim; ++r) {
            double* vr = &A[static_cast<size_t>(r) * dim];
            for (int p = 0; p < r; ++p) {
                const double* vp = &A[static_cast<size_t>(p) * dim];
                double proj = 0.0;
                for (int c = 0; c < dim; ++c) proj += vr[c] * vp[c];
                for (int c = 0; c < dim; ++c) vr[c] -= proj * vp[c];
            }
            double nrm = 0.0;
            for (int c = 0; c < dim; ++c) nrm += vr[c] * vr[c];
            nrm = std::sqrt(nrm);
            for (int c = 0; c < dim; ++c) vr[c] /= nrm;
        }
    }
    std::vector<float> R(A.size());
    for (size_t i = 0; i < A.size(); ++i) R[i] = static_cast<float>(A[i]);
    return R;
}

// Train the OPQ rotation on a sample (n×dim, already metric-normalized).
// `m` = PQ sub-quantizers of the final index. Returns row-major dim×dim R
// (identity on failure). kmIters is deliberately small — each alternation only
// needs an approximate PQ; the final index retrains its PQ fully in R-space.
inline std::vector<float> trainOpqRotation(const float* data, int n, int dim,
                                           int m, int iters = 20, int kmIters = 4) {
    std::vector<float> R(static_cast<size_t>(dim) * dim, 0.0f);
    for (int i = 0; i < dim; ++i) R[static_cast<size_t>(i) * dim + i] = 1.0f;
    if (n <= 0 || dim <= 0 || m <= 0 || dim % m != 0) return R;
    R = randomRotation(dim);

    const int dsub = dim / m;
    std::vector<float> rot(static_cast<size_t>(n) * dim);   // R·X
    std::vector<float> rec(static_cast<size_t>(n) * dim);   // PQ reconstruction of R·X
    std::vector<uint8_t> codes;
    double prevErr = std::numeric_limits<double>::infinity();

    for (int it = 0; it < iters; ++it) {
        applyRotation(R.data(), data, rot.data(), n, dim);

        PqTrainer pq(dim, m);
        pq.train(rot.data(), n, kmIters, &codes);
        if (!pq.trained()) break;

        // Reconstruct and measure the quantization error in rotated space.
        const std::vector<float>& cent = pq.centroids();
        parallelFor(n, [&](int i) {
            float* y = &rec[static_cast<size_t>(i) * dim];
            const uint8_t* code = &codes[static_cast<size_t>(i) * m];
            for (int mm = 0; mm < m; ++mm)
                std::copy_n(&cent[(static_cast<size_t>(mm) * 256 + code[mm]) * dsub],
                            dsub, y + static_cast<size_t>(mm) * dsub);
        });
        double err = 0.0;
        for (size_t i = 0; i < rot.size(); ++i) {
            const double e = double(rot[i]) - double(rec[i]);
            err += e * e;
        }
        MFLAT_LOG_INFO("opq iter %d: quantization error %.6g", it, err / n);
        if (err >= prevErr * (1.0 - 1e-4)) break;    // converged / no longer improving
        prevErr = err;

        // Procrustes update: M = Y_hat·X^T (dim×dim), R = polar(M). Accumulate
        // in double — fp32 drifts over n=65k terms and skews the SVD.
        std::vector<float> Mm(static_cast<size_t>(dim) * dim, 0.0f);
        parallelFor(dim, [&](int r) {
            std::vector<double> Mr(dim, 0.0);
            for (int i = 0; i < n; ++i) {
                const double yr = rec[static_cast<size_t>(i) * dim + r];
                const float* x = data + static_cast<size_t>(i) * dim;
                for (int c = 0; c < dim; ++c) Mr[c] += yr * x[c];
            }
            for (int c = 0; c < dim; ++c)
                Mm[static_cast<size_t>(r) * dim + c] = static_cast<float>(Mr[c]);
        });
        std::vector<float> Rnew;
        if (!polarOrthogonal(Mm, dim, Rnew)) break;  // degenerate — keep current R
        R = std::move(Rnew);
    }
    return R;
}

}  // namespace detail
}  // namespace mflat
