// SPDX-License-Identifier: Apache-2.0
// mflat.cpp — C ABI implementation over the C++ engine.
//
// Each opaque handle wraps the C++ index plus a std::mutex: the v1 engine is
// not thread-safe (the GPU buffer syncs lazily), so we serialize mutating /
// search calls per handle. The lock is cheap (the GPU already serializes work)
// and makes the ABI safe-by-default. Every entry point is a try/catch shell
// mapping C++ exceptions to status codes; search results are memcpy'd into the
// caller's buffers. This file is plain C++ (no Metal/ObjC) — the public headers
// hide Metal behind the pimpl — and links the metalflat static library.

#include "mflat.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <new>

#include "metalflat/FlatIndex.h"
#include "metalflat/IvfIndex.h"
#include "metalflat/IvfPqIndex.h"
#include "metalflat/Log.h"

using mflat::FlatIndex;
using mflat::IvfIndex;
using mflat::IvfPqIndex;
using mflat::Metric;
using mflat::SearchResult;

namespace {

Metric conv(mflat_metric_t m) {
    switch (m) {
        case MFLAT_METRIC_INNER_PRODUCT: return Metric::InnerProduct;
        case MFLAT_METRIC_COSINE:        return Metric::Cosine;
        default:                         return Metric::L2;
    }
}

// Copy a SearchResult into caller buffers. k_used = columns actually produced
// (= result.size()/m); caller's buffers are sized for their requested k >= k_used.
mflat_status_t emit(const SearchResult& r, int m,
                    int32_t* out_ids, float* out_distances, int* out_k_used) {
    if (m <= 0 || r.ids.empty()) { if (out_k_used) *out_k_used = 0; return MFLAT_OK; }
    const int kUsed = static_cast<int>(r.ids.size() / static_cast<size_t>(m));
    if (out_ids)       std::memcpy(out_ids,       r.ids.data(),       r.ids.size() * sizeof(int32_t));
    if (out_distances) std::memcpy(out_distances, r.distances.data(), r.distances.size() * sizeof(float));
    if (out_k_used)    *out_k_used = kUsed;
    return MFLAT_OK;
}

// Log trampoline: the C handler type (C enum) differs from mflat::LogHandler
// (enum class), so we trampoline through this — never cast the function pointer.
std::atomic<mflat_log_handler_t> gCHandler{nullptr};
std::atomic<void*>               gCUser{nullptr};
void cLogTrampoline(mflat::LogLevel lvl, const char* msg, void*) {
    if (auto h = gCHandler.load(std::memory_order_acquire))
        h(static_cast<mflat_log_level_t>(lvl), msg, gCUser.load(std::memory_order_acquire));
}

}  // namespace

struct mflat_flat_index  { FlatIndex   idx; std::mutex mu;
    mflat_flat_index(int d, Metric m) : idx(d, m) {} };
struct mflat_ivf_index   { IvfIndex    idx; std::mutex mu;
    mflat_ivf_index(int d, Metric m, int nl) : idx(d, m, nl) {} };
struct mflat_ivfpq_index { IvfPqIndex  idx; std::mutex mu;
    mflat_ivfpq_index(int d, Metric m, int nl, int sub) : idx(d, m, nl, sub) {} };

extern "C" {

const char* mflat_version(void) { return "0.1.0"; }
int         mflat_max_k(void)   { return MFLAT_MAX_K; }

void mflat_set_log_handler(mflat_log_handler_t h, void* user) {
    gCUser.store(user, std::memory_order_release);
    gCHandler.store(h, std::memory_order_release);
    mflat::setLogHandler(h ? &cLogTrampoline : nullptr, nullptr);  // NULL => C++ default sink
}
void mflat_set_log_level(mflat_log_level_t l) {
    mflat::setLogLevel(static_cast<mflat::LogLevel>(l));
}
mflat_log_level_t mflat_log_level(void) {
    return static_cast<mflat_log_level_t>(mflat::logLevel());
}
const char* mflat_status_str(mflat_status_t s) {
    switch (s) {
        case MFLAT_OK:            return "ok";
        case MFLAT_ERR_NULL_ARG:  return "null argument";
        case MFLAT_ERR_BAD_ARG:   return "bad argument";
        case MFLAT_ERR_ALLOC:     return "allocation failed";
        case MFLAT_ERR_NOT_READY: return "index not built";
        case MFLAT_ERR_INTERNAL:  return "internal error";
    }
    return "unknown";
}

/* ---------------- FlatIndex ------------------------------------------- */
mflat_flat_index_t* mflat_flat_create(int dim, mflat_metric_t metric, mflat_status_t* st) {
    if (dim <= 0) { if (st) *st = MFLAT_ERR_BAD_ARG; return nullptr; }
    try { auto* h = new mflat_flat_index(dim, conv(metric)); if (st) *st = MFLAT_OK; return h; }
    catch (const std::bad_alloc&) { if (st) *st = MFLAT_ERR_ALLOC; return nullptr; }
    catch (...)                   { if (st) *st = MFLAT_ERR_INTERNAL; return nullptr; }
}
void mflat_flat_free(mflat_flat_index_t* h) { delete h; }
int  mflat_flat_ready(const mflat_flat_index_t* h) { return (h && h->idx.ready()) ? 1 : 0; }
int  mflat_flat_size (const mflat_flat_index_t* h) { return h ? h->idx.size() : 0; }
int  mflat_flat_dim  (const mflat_flat_index_t* h) { return h ? h->idx.dim()  : 0; }

mflat_status_t mflat_flat_add(mflat_flat_index_t* h, const float* v, int n) {
    if (!h || !v) return MFLAT_ERR_NULL_ARG;
    if (n < 0)    return MFLAT_ERR_BAD_ARG;
    try { std::lock_guard<std::mutex> lk(h->mu); h->idx.add(v, n); return MFLAT_OK; }
    catch (const std::bad_alloc&) { return MFLAT_ERR_ALLOC; }
    catch (...)                   { return MFLAT_ERR_INTERNAL; }
}
mflat_status_t mflat_flat_reset(mflat_flat_index_t* h) {
    if (!h) return MFLAT_ERR_NULL_ARG;
    try { std::lock_guard<std::mutex> lk(h->mu); h->idx.reset(); return MFLAT_OK; }
    catch (...) { return MFLAT_ERR_INTERNAL; }
}
mflat_status_t mflat_flat_search(mflat_flat_index_t* h, const float* q, int m, int k,
                                 int32_t* oi, float* od, int* ok) {
    if (!h || !q) return MFLAT_ERR_NULL_ARG;
    if (m < 0 || k < 1) return MFLAT_ERR_BAD_ARG;
    try { std::lock_guard<std::mutex> lk(h->mu); return emit(h->idx.search(q, m, k), m, oi, od, ok); }
    catch (const std::bad_alloc&) { return MFLAT_ERR_ALLOC; }
    catch (...)                   { return MFLAT_ERR_INTERNAL; }
}

/* ---------------- IvfIndex -------------------------------------------- */
mflat_ivf_index_t* mflat_ivf_create(int dim, mflat_metric_t metric, int nlist, mflat_status_t* st) {
    if (dim <= 0 || nlist <= 0) { if (st) *st = MFLAT_ERR_BAD_ARG; return nullptr; }
    try { auto* h = new mflat_ivf_index(dim, conv(metric), nlist); if (st) *st = MFLAT_OK; return h; }
    catch (const std::bad_alloc&) { if (st) *st = MFLAT_ERR_ALLOC; return nullptr; }
    catch (...)                   { if (st) *st = MFLAT_ERR_INTERNAL; return nullptr; }
}
void mflat_ivf_free(mflat_ivf_index_t* h) { delete h; }
int  mflat_ivf_ready(const mflat_ivf_index_t* h) { return (h && h->idx.ready()) ? 1 : 0; }
int  mflat_ivf_size (const mflat_ivf_index_t* h) { return h ? h->idx.size()  : 0; }
int  mflat_ivf_dim  (const mflat_ivf_index_t* h) { return h ? h->idx.dim()   : 0; }
int  mflat_ivf_nlist(const mflat_ivf_index_t* h) { return h ? h->idx.nlist() : 0; }

mflat_status_t mflat_ivf_build(mflat_ivf_index_t* h, const float* v, int n) {
    if (!h || !v) return MFLAT_ERR_NULL_ARG;
    if (n <= 0)   return MFLAT_ERR_BAD_ARG;
    try { std::lock_guard<std::mutex> lk(h->mu); h->idx.build(v, n); return MFLAT_OK; }
    catch (const std::bad_alloc&) { return MFLAT_ERR_ALLOC; }
    catch (...)                   { return MFLAT_ERR_INTERNAL; }
}
mflat_status_t mflat_ivf_search(mflat_ivf_index_t* h, const float* q, int m, int k, int nprobe,
                                int32_t* oi, float* od, int* ok) {
    if (!h || !q) return MFLAT_ERR_NULL_ARG;
    if (m < 0 || k < 1 || nprobe < 1) return MFLAT_ERR_BAD_ARG;
    if (!h->idx.ready()) return MFLAT_ERR_NOT_READY;
    try { std::lock_guard<std::mutex> lk(h->mu); return emit(h->idx.search(q, m, k, nprobe), m, oi, od, ok); }
    catch (const std::bad_alloc&) { return MFLAT_ERR_ALLOC; }
    catch (...)                   { return MFLAT_ERR_INTERNAL; }
}

/* ---------------- IvfPqIndex ------------------------------------------ */
mflat_ivfpq_index_t* mflat_ivfpq_create(int dim, mflat_metric_t metric, int nlist, int m_sub,
                                        mflat_status_t* st) {
    if (dim <= 0 || nlist <= 0 || m_sub <= 0 || dim % m_sub != 0 ||
        metric == MFLAT_METRIC_INNER_PRODUCT) {
        if (st) *st = MFLAT_ERR_BAD_ARG; return nullptr;   /* IP unsupported on IVFPQ */
    }
    try { auto* h = new mflat_ivfpq_index(dim, conv(metric), nlist, m_sub); if (st) *st = MFLAT_OK; return h; }
    catch (const std::bad_alloc&) { if (st) *st = MFLAT_ERR_ALLOC; return nullptr; }
    catch (...)                   { if (st) *st = MFLAT_ERR_INTERNAL; return nullptr; }
}
void mflat_ivfpq_free(mflat_ivfpq_index_t* h) { delete h; }
void mflat_ivfpq_set_rerank(mflat_ivfpq_index_t* h, int enable) {
    if (h) { std::lock_guard<std::mutex> lk(h->mu); h->idx.setRerank(enable != 0); }
}
int mflat_ivfpq_ready(const mflat_ivfpq_index_t* h) { return (h && h->idx.ready()) ? 1 : 0; }
int mflat_ivfpq_size (const mflat_ivfpq_index_t* h) { return h ? h->idx.size()  : 0; }
int mflat_ivfpq_dim  (const mflat_ivfpq_index_t* h) { return h ? h->idx.dim()   : 0; }
int mflat_ivfpq_nlist(const mflat_ivfpq_index_t* h) { return h ? h->idx.nlist() : 0; }
int mflat_ivfpq_subquantizers(const mflat_ivfpq_index_t* h) { return h ? h->idx.subquantizers() : 0; }

mflat_status_t mflat_ivfpq_build(mflat_ivfpq_index_t* h, const float* v, int n) {
    if (!h || !v) return MFLAT_ERR_NULL_ARG;
    if (n <= 0)   return MFLAT_ERR_BAD_ARG;
    try { std::lock_guard<std::mutex> lk(h->mu); h->idx.build(v, n); return MFLAT_OK; }
    catch (const std::bad_alloc&) { return MFLAT_ERR_ALLOC; }
    catch (...)                   { return MFLAT_ERR_INTERNAL; }
}
mflat_status_t mflat_ivfpq_search(mflat_ivfpq_index_t* h, const float* q, int m, int k, int nprobe,
                                  int rerank, int32_t* oi, float* od, int* ok) {
    if (!h || !q) return MFLAT_ERR_NULL_ARG;
    if (m < 0 || k < 1 || nprobe < 1 || rerank < 0) return MFLAT_ERR_BAD_ARG;
    if (!h->idx.ready()) return MFLAT_ERR_NOT_READY;
    try { std::lock_guard<std::mutex> lk(h->mu); return emit(h->idx.search(q, m, k, nprobe, rerank), m, oi, od, ok); }
    catch (const std::bad_alloc&) { return MFLAT_ERR_ALLOC; }
    catch (...)                   { return MFLAT_ERR_INTERNAL; }
}

}  // extern "C"
