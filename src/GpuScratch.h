// SPDX-License-Identifier: Apache-2.0
// GpuScratch.h — persistent, grow-on-demand GPU buffers (mflat::detail).
//
// NOT part of the public API. Objective-C++ only (Metal types), header-only.
//
// Why this exists: every search() used to allocate its query / output / staging
// buffers FRESH — newBufferWithBytes + newBufferWithLength, several per call —
// and then throw them away. On a large batch that cost is amortised into
// nothing. On a SINGLE query it is most of the latency: a Metal allocation is
// not just a malloc, it maps pages and touches the driver, and we were paying
// for 4-8 of them per query. That cost was being blamed on "the GPU dispatch
// floor" when a large part of it was ours.
//
// So: keep the buffers on the index, grow them when a bigger batch arrives,
// never shrink, and memcpy into them instead of reallocating. Reuse is safe
// because search() always awaits completion before returning (and the engine is
// documented as not thread-safe per index — callers serialise).
//
// Slots are just small integers, one per logical buffer in a kernel's binding
// list; each index defines its own slot enum next to its dispatch code.

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace mflat {
namespace detail {

class GpuScratch {
public:
    void setDevice(id<MTLDevice> dev) { mDevice = dev; }

    // Buffer for `slot`, at least `bytes` long. Grown (with headroom) when the
    // request outgrows it; the same MTLBuffer is handed back otherwise.
    id<MTLBuffer> ensure(int slot, size_t bytes,
                         MTLResourceOptions opts = MTLResourceStorageModeShared) {
        if (!mDevice || bytes == 0) return nil;
        if (static_cast<int>(mSlots.size()) <= slot) {
            mSlots.resize(slot + 1, nil);
            mCaps.resize(slot + 1, 0);
            mOpts.resize(slot + 1, 0);
        }
        if (!mSlots[slot] || mCaps[slot] < bytes || mOpts[slot] != opts) {
            // Over-allocate by 25% so a slowly growing batch does not realloc
            // on every call; a shrinking batch keeps the bigger buffer.
            const size_t want = std::max<size_t>(bytes + bytes / 4, 256);
            mSlots[slot] = [mDevice newBufferWithLength:want options:opts];
            mCaps[slot]  = mSlots[slot] ? want : 0;
            mOpts[slot]  = opts;
        }
        return mSlots[slot];
    }

    // ensure() + copy `bytes` from `src` (Shared storage: CPU-writable).
    id<MTLBuffer> upload(int slot, const void* src, size_t bytes) {
        id<MTLBuffer> b = ensure(slot, bytes);
        if (b && src && bytes) std::memcpy([b contents], src, bytes);
        return b;
    }

    void reset() { mSlots.clear(); mCaps.clear(); mOpts.clear(); }

private:
    id<MTLDevice>              mDevice = nil;
    std::vector<id<MTLBuffer>> mSlots;
    std::vector<size_t>        mCaps;
    std::vector<MTLResourceOptions> mOpts;
};

}  // namespace detail
}  // namespace mflat
