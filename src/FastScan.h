// SPDX-License-Identifier: Apache-2.0
// FastScan.h — 4-bit PQ "fast-scan" ADC (mflat::detail), CPU-only, header-only.
//
// Why this exists: metal-flat's IVFPQ loses to faiss IVFPQ+fs at recall >= 0.95
// on SIFT (up to 1.4x), and every Pareto point up there runs the CPU ADC (the
// rerank shortlist needs kRun > the GPU top-k). The 8-bit ADC's inner loop is a
// scalar gather: m_sub random loads from a 16 KB LUT per candidate. NEON has no
// gather — but it has vqtbl1q_u8, a 16-ENTRY TABLE LOOKUP IN ONE INSTRUCTION.
// That instruction is the entire reason faiss's fast-scan uses 4-bit codes: a
// 16-entry LUT fits in one 128-bit register, so the "table lookup" becomes
// register-speed and 32 candidates are scored per pass over the subquantizers.
//
// Scheme (FAISS fast-scan, adapted):
//   - PQ with ksub=16 (4-bit codes) and m4 = 2*m_sub subquantizers, so
//     bytes/vector is UNCHANGED (two codes per byte). dsub4 = dim/m4.
//   - Codes packed per IVF cell into BLOCKS of 32 vectors: block b, subq j is
//     16 bytes where byte t holds vec(32b+t)'s code in the low nibble and
//     vec(32b+16+t)'s in the high nibble. One vld1q_u8 = one subq's codes for
//     all 32 vectors.
//   - The per-(query[,cell]) float LUT (m4 x 16) is quantized to u8 with a
//     shared scale and per-subq bias:  dist ~= base + acc/scale + biasSum,
//     acc = sum_j q[j][code].  u8 entries, m4 <= 128 => acc < 32640 fits u16.
//   - Scan: per block, m4 iterations of {load, and/shift, 2x vqtbl1q_u8,
//     widening add} produce 32 u16 distances; a vminvq early-out skips whole
//     blocks that cannot beat the current worst kept distance.
//
// The shortlist this produces is ranked by QUANTIZED distances (the ~8-bit LUT
// rounding is the only approximation vs the 8-bit ADC's exact-float ADC —
// which is itself an approximation of the true distance). The exact rerank
// that every high-recall config already runs repairs both.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mflat {
namespace detail {

constexpr int kFsKsub  = 16;   // 4-bit codes
constexpr int kFsBlock = 32;   // vectors per packed block

// ---- build-time packing ---------------------------------------------------

// Blocks needed for a cell of `count` vectors.
inline int fsBlocksFor(int count) { return (count + kFsBlock - 1) / kFsBlock; }

// Pack one cell's codes (CSR-slot order, values 0..15, row stride m4) into the
// blocked nibble layout at `dst` (fsBlocksFor(count) * m4 * 16 bytes).
// Padding lanes get code 0; the caller pads the id array with -1 so they are
// never emitted.
inline void fsPackCell(const uint8_t* codes, int count, int m4, uint8_t* dst) {
    const int nBlk = fsBlocksFor(count);
    for (int b = 0; b < nBlk; ++b)
        for (int j = 0; j < m4; ++j)
            for (int t = 0; t < 16; ++t) {
                const int vLo = b * kFsBlock + t;
                const int vHi = vLo + 16;
                const uint8_t lo = (vLo < count) ? codes[static_cast<size_t>(vLo) * m4 + j] : 0;
                const uint8_t hi = (vHi < count) ? codes[static_cast<size_t>(vHi) * m4 + j] : 0;
                dst[(static_cast<size_t>(b) * m4 + j) * 16 + t] =
                    static_cast<uint8_t>(lo | (hi << 4));
            }
}

// ---- query-time LUT quantization ------------------------------------------

// Quantize the float LUT (m4 x 16) to u8: q[j][c] = (lut[j][c] - min_j) * scale,
// one shared scale so u16 accumulation is exact. Returns via out-params the
// scale and biasSum with  dist ~= base + acc/scale + biasSum.
inline void fsQuantizeLut(const float* lut, int m4,
                          uint8_t* q, float* scaleOut, float* biasSumOut) {
    float maxRange = 1e-20f, biasSum = 0.0f;
    for (int j = 0; j < m4; ++j) {
        const float* l = lut + static_cast<size_t>(j) * kFsKsub;
        float mn = l[0], mx = l[0];
        for (int c = 1; c < kFsKsub; ++c) { mn = std::min(mn, l[c]); mx = std::max(mx, l[c]); }
        biasSum += mn;
        maxRange = std::max(maxRange, mx - mn);
    }
    const float scale = 255.0f / maxRange;
    for (int j = 0; j < m4; ++j) {
        const float* l  = lut + static_cast<size_t>(j) * kFsKsub;
        float mn = l[0];
        for (int c = 1; c < kFsKsub; ++c) mn = std::min(mn, l[c]);
        uint8_t* qj = q + static_cast<size_t>(j) * kFsKsub;
        for (int c = 0; c < kFsKsub; ++c) {
            const float v = (l[c] - mn) * scale;
            qj[c] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, std::round(v))));
        }
    }
    *scaleOut   = scale;
    *biasSumOut = biasSum;
}

// ---- the scan --------------------------------------------------------------

// Score one 32-vector block: blk is m4 x 16 packed bytes, lut is the quantized
// m4 x 16 table. Writes 32 u16 distances to out and returns their minimum
// (for the caller's block-level early-out).
#if defined(__aarch64__)
inline uint16_t fsScanBlock(const uint8_t* blk, const uint8_t* lut, int m4,
                            uint16_t out[kFsBlock]) {
    uint16x8_t a0 = vdupq_n_u16(0), a1 = vdupq_n_u16(0);
    uint16x8_t a2 = vdupq_n_u16(0), a3 = vdupq_n_u16(0);
    const uint8x16_t maskLo = vdupq_n_u8(0x0F);
    for (int j = 0; j < m4; ++j) {
        const uint8x16_t codes = vld1q_u8(blk + static_cast<size_t>(j) * 16);
        const uint8x16_t table = vld1q_u8(lut + static_cast<size_t>(j) * 16);
        const uint8x16_t dLo = vqtbl1q_u8(table, vandq_u8(codes, maskLo));  // vecs 0..15
        const uint8x16_t dHi = vqtbl1q_u8(table, vshrq_n_u8(codes, 4));     // vecs 16..31
        a0 = vaddw_u8(a0, vget_low_u8(dLo));
        a1 = vaddw_high_u8(a1, dLo);
        a2 = vaddw_u8(a2, vget_low_u8(dHi));
        a3 = vaddw_high_u8(a3, dHi);
    }
    vst1q_u16(out,      a0);
    vst1q_u16(out + 8,  a1);
    vst1q_u16(out + 16, a2);
    vst1q_u16(out + 24, a3);
    const uint16_t m01 = std::min(vminvq_u16(a0), vminvq_u16(a1));
    const uint16_t m23 = std::min(vminvq_u16(a2), vminvq_u16(a3));
    return std::min(m01, m23);
}
#else
inline uint16_t fsScanBlock(const uint8_t* blk, const uint8_t* lut, int m4,
                            uint16_t out[kFsBlock]) {
    uint32_t acc[kFsBlock] = {0};
    for (int j = 0; j < m4; ++j) {
        const uint8_t* c = blk + static_cast<size_t>(j) * 16;
        const uint8_t* l = lut + static_cast<size_t>(j) * kFsKsub;
        for (int t = 0; t < 16; ++t) {
            acc[t]      += l[c[t] & 0x0F];
            acc[t + 16] += l[c[t] >> 4];
        }
    }
    uint16_t mn = 0xFFFF;
    for (int t = 0; t < kFsBlock; ++t) {
        out[t] = static_cast<uint16_t>(std::min<uint32_t>(acc[t], 0xFFFF));
        mn = std::min(mn, out[t]);
    }
    return mn;
}
#endif

}  // namespace detail
}  // namespace mflat
