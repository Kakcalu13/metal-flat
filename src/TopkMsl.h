// SPDX-License-Identifier: Apache-2.0
// TopkMsl.h — the shared Metal top-k reduction snippet (mflat::detail).
//
// NOT part of the public API. Every GPU kernel that keeps a per-thread top-k and
// then reduces it across a threadgroup (FlatIndex's topk_partial /
// topk_merge_partials, IvfIndex's ivf_scan / ivf_partial_merge, IvfPqIndex's
// ivfpq_adc) prepends this source to its own kernel body, so the reduction
// exists once instead of once per index.
//
// The scheme, and why it matters: per-thread lists are ASCENDING over
// kk = nextPow2(k) slots (index 0 = worst kept; unfilled slots hold -INF/-1).
// A power-of-two length enables the bitonic merge trick — c[i] = max(a[i],
// b[kk-1-i]) holds the kk largest of the union and is a bitonic sequence, so a
// log2(kk)-stage bitonic merge re-sorts it — which lets two lanes merge lists
// through simd_shuffle_xor with NO threadgroup memory. A butterfly over
// off = 1..16 leaves every lane holding its simdgroup's merged top-kk; only the
// (tgs/32) simdgroup leaders then publish, and a small tree merge finishes it.
//
// The scratch is therefore (tgs/32) * kk entries — NOT the tgs * k of the naive
// tree merge. That distinction is the whole point: with tgs * k scratch, a
// threadgroup-memory budget of ~32 KB forces tgs down to 32 threads at k = 64
// (measured: IVF fine scan 8x under-occupied, IVFPQ 5-15x SLOWER than its own
// CPU path), so k silently throttled the GPU. With this scheme k never
// constrains the threadgroup size.
//
// Host contract: threadgroup size is a power of two >= 32; scratch buffers hold
// at least (tgs/32) * kk floats and ints; kk = nextPow2(k) <= kMaxK.

#pragma once

#import <Foundation/Foundation.h>

namespace mflat {
namespace detail {

inline NSString* const kTopkMslSrc = @R"(
#include <metal_stdlib>
using namespace metal;

constant uint kMaxK = 64;

// Insert `sc` into an ascending kk-list, evicting the worst. Callers gate on
// sc > s[0] (or a cached `worst`) so the common case never enters here.
inline void insertTopk(thread float* s, thread int* id, uint kk, float sc, int gid) {
    uint pos = 0;
    while (pos + 1u < kk && sc > s[pos + 1u]) {
        s[pos]  = s[pos + 1u];
        id[pos] = id[pos + 1u];
        ++pos;
    }
    s[pos]  = sc;
    id[pos] = gid;
}

// Merge this lane's ascending kk-list with lane^off's (via simd shuffle):
// pairwise max against the partner's REVERSED list, then bitonic re-sort.
inline void simdMergeTopk(thread float* s, thread int* id, uint kk, uint off) {
    float ns[kMaxK];
    int   ni[kMaxK];
    for (uint i = 0; i < kk; ++i) {
        const float os = simd_shuffle_xor(s[kk - 1u - i], off);
        const int   oi = simd_shuffle_xor(id[kk - 1u - i], off);
        if (os > s[i]) { ns[i] = os;   ni[i] = oi; }
        else           { ns[i] = s[i]; ni[i] = id[i]; }
    }
    for (uint i = 0; i < kk; ++i) { s[i] = ns[i]; id[i] = ni[i]; }
    for (uint st = kk >> 1; st > 0u; st >>= 1)
        for (uint i = 0; i < kk; ++i) {
            const uint j = i | st;
            if ((i & st) == 0u && j < kk && s[i] > s[j]) {
                const float ts = s[i];  s[i]  = s[j];  s[j]  = ts;
                const int   ti = id[i]; id[i] = id[j]; id[j] = ti;
            }
        }
}

// Two-pointer merge of two ascending kk-lists in threadgroup memory, keeping
// the kk largest in a.
inline void mergeListsTg(threadgroup float* aS, threadgroup int* aI,
                         threadgroup float* bS, threadgroup int* bI, uint kk) {
    float mS[kMaxK];
    int   mI[kMaxK];
    int ia = (int)kk - 1, ib = (int)kk - 1;
    for (int o = (int)kk - 1; o >= 0; --o) {
        const float av = (ia >= 0) ? aS[ia] : -INFINITY;
        const float bv = (ib >= 0) ? bS[ib] : -INFINITY;
        if (av >= bv) { mS[o] = av; mI[o] = (ia >= 0) ? aI[ia] : -1; --ia; }
        else          { mS[o] = bv; mI[o] = (ib >= 0) ? bI[ib] : -1; --ib; }
    }
    for (uint i = 0; i < kk; ++i) { aS[i] = mS[i]; aI[i] = mI[i]; }
}

// Reduce the threadgroup's per-thread lists to ONE ascending kk-list in
// redScore/redId[0..kk): simdgroup shuffle butterfly (no barriers, no scratch),
// then a tree merge over the (tgs/32) simdgroup leaders. The caller's top-k is
// redScore[kk-k .. kk).
inline void reduceTopkTg(thread float* s, thread int* id, uint kk,
                         threadgroup float* redScore, threadgroup int* redId,
                         uint tid, uint tgs, uint sgid, uint lane) {
    for (uint off = 1u; off < 32u; off <<= 1) simdMergeTopk(s, id, kk, off);
    if (lane == 0u)
        for (uint i = 0; i < kk; ++i) {
            redScore[sgid * kk + i] = s[i];
            redId[sgid * kk + i]    = id[i];
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = (tgs >> 5) >> 1; off > 0u; off >>= 1) {
        if (tid < off)
            mergeListsTg(redScore + tid * kk, redId + tid * kk,
                         redScore + (tid + off) * kk, redId + (tid + off) * kk, kk);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)";

// Host-side mirror of the kk the kernels compute: the per-thread list length.
inline uint32_t nextPow2K(int k) {
    uint32_t p = 1;
    while (p < static_cast<uint32_t>(k)) p <<= 1;
    return p;
}

// Threadgroup-memory bytes the reduction scratch needs, per scratch buffer
// (one for scores, one for ids), 16-byte aligned as Metal expects.
inline NSUInteger topkScratchBytes(NSUInteger tgSize, uint32_t kk) {
    return (((tgSize / 32) * kk * 4) + 15) & ~NSUInteger(15);
}

}  // namespace detail
}  // namespace mflat
