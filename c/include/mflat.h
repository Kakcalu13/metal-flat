/* SPDX-License-Identifier: Apache-2.0 */
/* mflat.h — stable C ABI for metal-flat.
 *
 * One C boundary over the C++ engine (FlatIndex / IvfIndex / IvfPqIndex) so any
 * language (Python, Swift, Rust, ...) can drive it. Opaque handles, caller-owned
 * output buffers, explicit status codes; SemVer, additive-only.
 *
 * Buffer contract for every *_search: out_ids and out_distances are caller-
 * allocated and must each hold at least m * k elements, row-major (row r, col j
 * at r*k + j), nearest-first. *out_k_used (nullable) receives how many columns
 * per row were actually written (the GPU exact path caps k at MFLAT_MAX_K; the
 * IVF/IVFPQ exact-CPU path serves k > MFLAT_MAX_K). ids are -1 padded.
 *
 * Apple Silicon only; without a Metal device the engine falls back to exact CPU.
 */
#ifndef MFLAT_H
#define MFLAT_H

#include <stdint.h>

#if defined(_WIN32)
#  define MFLAT_API
#else
#  define MFLAT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MFLAT_VERSION_MAJOR 0
#define MFLAT_VERSION_MINOR 1
#define MFLAT_VERSION_PATCH 0
#define MFLAT_MAX_K 64          /* GPU per-thread top-k limit; mirrors kMaxK */

typedef enum {                  /* value-stable; mirrors mflat::Metric */
    MFLAT_METRIC_L2            = 0,
    MFLAT_METRIC_INNER_PRODUCT = 1,
    MFLAT_METRIC_COSINE        = 2
} mflat_metric_t;

typedef enum {
    MFLAT_OK            = 0,
    MFLAT_ERR_NULL_ARG  = 1,    /* a required pointer was NULL                 */
    MFLAT_ERR_BAD_ARG   = 2,    /* dim/n/m/k/nlist/m_sub/metric out of range   */
    MFLAT_ERR_ALLOC     = 3,    /* host allocation / std::bad_alloc            */
    MFLAT_ERR_NOT_READY = 4,    /* IVF/IVFPQ searched before build()           */
    MFLAT_ERR_INTERNAL  = 5     /* unexpected C++ exception                    */
} mflat_status_t;

MFLAT_API const char* mflat_version(void);             /* "0.1.0"             */
MFLAT_API const char* mflat_status_str(mflat_status_t);/* static; do not free */
MFLAT_API int         mflat_max_k(void);               /* MFLAT_MAX_K         */

/* ---------------- Logging -------------------------------------------- */
/* By default the library writes diagnostics to stderr ("[metalflat] ..."). */
/* Consumers can redirect or silence that. Process-global (like mflat_version). */
typedef enum {                    /* value-stable; mirrors mflat::LogLevel */
    MFLAT_LOG_INFO  = 0,
    MFLAT_LOG_WARN  = 1,
    MFLAT_LOG_ERROR = 2,
    MFLAT_LOG_OFF   = 3            /* threshold only; never delivered to a handler */
} mflat_log_level_t;

/* msg: NUL-terminated UTF-8, no "[metalflat]" prefix and no newline. user is
   round-tripped untouched. Called synchronously on the logging thread. */
typedef void (*mflat_log_handler_t)(mflat_log_level_t level, const char* msg, void* user);

/* handler == NULL RESTORES the built-in stderr sink; use
   mflat_set_log_level(MFLAT_LOG_OFF) to silence. Configure once before
   concurrent searches; a custom handler must be thread-safe and must not throw. */
MFLAT_API void              mflat_set_log_handler(mflat_log_handler_t handler, void* user);
MFLAT_API void              mflat_set_log_level(mflat_log_level_t threshold);
MFLAT_API mflat_log_level_t mflat_log_level(void);

/* ---------------- FlatIndex (exact) ----------------------------------- */
typedef struct mflat_flat_index mflat_flat_index_t;    /* opaque */

MFLAT_API mflat_flat_index_t*
mflat_flat_create(int dim, mflat_metric_t metric, mflat_status_t* out_status);
MFLAT_API void mflat_flat_free(mflat_flat_index_t*);

MFLAT_API int mflat_flat_ready(const mflat_flat_index_t*);  /* 1=GPU, 0=CPU fallback */
MFLAT_API int mflat_flat_size (const mflat_flat_index_t*);
MFLAT_API int mflat_flat_dim  (const mflat_flat_index_t*);

MFLAT_API mflat_status_t mflat_flat_add(mflat_flat_index_t*, const float* vectors, int n);
MFLAT_API mflat_status_t mflat_flat_reset(mflat_flat_index_t*);
MFLAT_API mflat_status_t mflat_flat_search(mflat_flat_index_t*, const float* queries,
        int m, int k, int32_t* out_ids, float* out_distances, int* out_k_used);

/* ---------------- IvfIndex (approximate) ------------------------------ */
typedef struct mflat_ivf_index mflat_ivf_index_t;      /* opaque */

MFLAT_API mflat_ivf_index_t*
mflat_ivf_create(int dim, mflat_metric_t metric, int nlist, mflat_status_t* out_status);
MFLAT_API void mflat_ivf_free(mflat_ivf_index_t*);

MFLAT_API int mflat_ivf_ready(const mflat_ivf_index_t*);  /* 1 after build() */
MFLAT_API int mflat_ivf_size (const mflat_ivf_index_t*);
MFLAT_API int mflat_ivf_dim  (const mflat_ivf_index_t*);
MFLAT_API int mflat_ivf_nlist(const mflat_ivf_index_t*);

MFLAT_API mflat_status_t mflat_ivf_build(mflat_ivf_index_t*, const float* vectors, int n);
MFLAT_API mflat_status_t mflat_ivf_search(mflat_ivf_index_t*, const float* queries,
        int m, int k, int nprobe, int32_t* out_ids, float* out_distances, int* out_k_used);

/* ---------------- IvfPqIndex (compressed, ~32x smaller) --------------- */
/* metric must be L2 or COSINE (pure InnerProduct is not supported here).  */
typedef struct mflat_ivfpq_index mflat_ivfpq_index_t;  /* opaque */

MFLAT_API mflat_ivfpq_index_t*
mflat_ivfpq_create(int dim, mflat_metric_t metric, int nlist, int m_sub, mflat_status_t* out_status);
MFLAT_API void mflat_ivfpq_free(mflat_ivfpq_index_t*);

/* Enable exact reranking BEFORE build() (keeps full vectors; see search). */
MFLAT_API void mflat_ivfpq_set_rerank(mflat_ivfpq_index_t*, int enable);

MFLAT_API int mflat_ivfpq_ready(const mflat_ivfpq_index_t*);  /* 1 after build() */
MFLAT_API int mflat_ivfpq_size (const mflat_ivfpq_index_t*);
MFLAT_API int mflat_ivfpq_dim  (const mflat_ivfpq_index_t*);
MFLAT_API int mflat_ivfpq_nlist(const mflat_ivfpq_index_t*);
MFLAT_API int mflat_ivfpq_subquantizers(const mflat_ivfpq_index_t*);  /* bytes/code */

MFLAT_API mflat_status_t mflat_ivfpq_build(mflat_ivfpq_index_t*, const float* vectors, int n);
/* rerank: 0/1 = off; >k = exact-rerank a top-`rerank` PQ shortlist -> top-k.   */
MFLAT_API mflat_status_t mflat_ivfpq_search(mflat_ivfpq_index_t*, const float* queries,
        int m, int k, int nprobe, int rerank,
        int32_t* out_ids, float* out_distances, int* out_k_used);

#ifdef __cplusplus
}
#endif
#endif /* MFLAT_H */
