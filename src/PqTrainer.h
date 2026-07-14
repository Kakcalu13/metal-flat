// SPDX-License-Identifier: Apache-2.0
// PqTrainer.h — Product Quantization codebook trainer/codec (mflat::detail).
//
// Internal, Obj-C++-only (pulls Internal.h for detail::kmeansGpu). Owns the PQ
// sub-quantizer centroids and their precomputed norms, encodes vectors to m-byte
// codes, and builds the per-query ADC lookup table. Residual PQ: train()/encode()
// take residuals (x - coarse_centroid); buildAdcTable() still takes the RAW
// query (see its comment), with the cell cross term in buildCellTable().
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "Internal.h"    // detail::{kmeansGpu, parallelFor} + Distance.h (dot, sqL2, adcLutEntry)

namespace mflat {
namespace detail {

class PqTrainer {
public:
    PqTrainer(int dim, int m, int ksub = 256)
        : dim_(dim), m_(m), ksub_(ksub), dsub_(m > 0 ? dim / m : 0) {}

    bool trained()   const { return trained_; }
    int  dim()       const { return dim_; }
    int  m()         const { return m_; }
    int  dsub()      const { return dsub_; }
    int  ksub()      const { return ksub_; }
    int  codeBytes() const { return m_; }              // 1 byte/subquantizer (8-bit)
    int  lutSize()   const { return m_ * ksub_; }
    const std::vector<float>& centroids() const { return centroids_; }  // m*ksub*dsub
    const std::vector<float>& norms()     const { return norms_; }      // m*ksub

    // Train per-subspace k-means on `training` (n×dim). If codesOut != nullptr,
    // fills n*m codes taken DIRECTLY from the k-means assignment (bit-exact —
    // a re-encode() would tie-break differently). No-op on invalid config.
    // maxPointsPerCentroid > 0 subsamples the TRAINING (faiss-style); the final
    // assignment pass still covers all n, so codesOut stays complete.
    void train(const float* training, int n, int iters,
               std::vector<uint8_t>* codesOut = nullptr,
               int maxPointsPerCentroid = 0) {
        trained_ = false;
        if (dim_ <= 0 || m_ <= 0 || dim_ % m_ != 0 || ksub_ > 256) return;
        centroids_.assign(static_cast<size_t>(m_) * ksub_ * dsub_, 0.0f);
        norms_.assign(static_cast<size_t>(m_) * ksub_, 0.0f);
        if (codesOut) codesOut->assign(static_cast<size_t>(n) * m_, 0);

        std::vector<float> sub(static_cast<size_t>(n) * dsub_);
        for (int mm = 0; mm < m_; ++mm) {
            parallelFor(n, [&](int i) {
                std::copy_n(training + static_cast<size_t>(i) * dim_ + mm * dsub_, dsub_,
                            &sub[static_cast<size_t>(i) * dsub_]);
            });
            std::vector<float> subCent;
            std::vector<int>   subAssign;
            kmeansGpu(sub.data(), n, dsub_, ksub_, iters, subCent, subAssign,
                      maxPointsPerCentroid);
            std::copy(subCent.begin(), subCent.end(),
                      &centroids_[static_cast<size_t>(mm) * ksub_ * dsub_]);
            for (int j = 0; j < ksub_; ++j) {
                const float* c = &subCent[static_cast<size_t>(j) * dsub_];
                norms_[static_cast<size_t>(mm) * ksub_ + j] = dot(c, c, dsub_);
            }
            if (codesOut)
                for (int i = 0; i < n; ++i)
                    (*codesOut)[static_cast<size_t>(i) * m_ + mm] =
                        static_cast<uint8_t>(subAssign[i]);
        }
        trained_ = true;
    }

    // Encode one vector to m bytes (nearest sub-centroid per subspace). For adds
    // / residual encoding after training; NOT used for the bit-exact build path.
    void encode(const float* vec, uint8_t* codeOut) const {
        for (int mm = 0; mm < m_; ++mm) {
            const float* qs = vec + static_cast<size_t>(mm) * dsub_;
            float best = std::numeric_limits<float>::infinity();
            int   bestJ = 0;
            for (int j = 0; j < ksub_; ++j) {
                const float d = sqL2(qs, &centroids_[(static_cast<size_t>(mm) * ksub_ + j) * dsub_], dsub_);
                if (d < best) { best = d; bestJ = j; }
            }
            codeOut[mm] = static_cast<uint8_t>(bestJ);
        }
    }
    void encode(const float* vecs, int n, uint8_t* codesOut) const {
        parallelFor(n, [&](int i) {
            encode(vecs + static_cast<size_t>(i) * dim_, codesOut + static_cast<size_t>(i) * m_);
        });
    }

    // Per-query ADC lookup table: lut[mm*ksub + j] = ||pqc||^2 - 2 q_mm·pqc.
    // Takes the RAW query in BOTH plain and residual mode — the residual
    // decomposition keeps this table cell-independent and routes the cross
    // term through buildCellTable() and the coarse ||q-c||^2 scalar instead.
    // lut must hold lutSize().
    void buildAdcTable(const float* query, float* lut) const {
        for (int mm = 0; mm < m_; ++mm)
            for (int j = 0; j < ksub_; ++j)
                lut[static_cast<size_t>(mm) * ksub_ + j] = adcLutEntry(
                    norms_[static_cast<size_t>(mm) * ksub_ + j],
                    query + static_cast<size_t>(mm) * dsub_,
                    &centroids_[(static_cast<size_t>(mm) * ksub_ + j) * dsub_], dsub_);
    }

    // Residual-ADC per-cell cross table:
    //   T[cell][mm][j] = 2 · dot(c_cell_mm, pqc[mm][j])
    // so that ||q - c - r||^2 = ||q-c||^2 + lut[mm][code] + T[cell][mm][code]
    // summed over mm. T must hold nlist * lutSize() floats (~64 MB at
    // nlist=4096, m=16 — allocated only in residual mode).
    void buildCellTable(const float* cellCentroids, int nlist, float* T) const {
        parallelFor(nlist, [&](int l) {
            const float* c = cellCentroids + static_cast<size_t>(l) * dim_;
            float* Tl = T + static_cast<size_t>(l) * m_ * ksub_;
            for (int mm = 0; mm < m_; ++mm)
                for (int j = 0; j < ksub_; ++j)
                    Tl[static_cast<size_t>(mm) * ksub_ + j] = 2.0f * dot(
                        c + static_cast<size_t>(mm) * dsub_,
                        &centroids_[(static_cast<size_t>(mm) * ksub_ + j) * dsub_], dsub_);
        });
    }

private:
    int  dim_, m_, ksub_, dsub_;
    bool trained_ = false;
    std::vector<float> centroids_;
    std::vector<float> norms_;
};

}  // namespace detail
}  // namespace mflat
