/* SPDX-License-Identifier: Apache-2.0 */
/* mflat.h — stable C ABI for metal-flat.
 *
 * One C boundary over the C++ engine (FlatIndex / IvfIndex / IvfPqIndex /
 * GraphIndex) so any language (Python, Swift, Rust, ...) can drive it. Opaque
 * handles, caller-owned output buffers, explicit status codes; SemVer,
 * additive-only.
 *
 * Buffer contract for every *_search: out_ids and out_distances are caller-
 * allocated and must each hold at least m * k elements, row-major (row r, col j
 * at r*k + j), nearest-first. *out_k_used (nullable) receives how many columns
 * per row were actually written (FlatIndex serves any k; the IVF/IVFPQ GPU
 * paths cap k at MFLAT_MAX_K and serve larger k on the exact-CPU path; the
 * graph index caps k at MFLAT_MAX_K). ids are -1 padded.
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
#define MFLAT_VERSION_MINOR 2
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
    MFLAT_ERR_NOT_READY = 4,    /* IVF/IVFPQ/graph searched before build()     */
    MFLAT_ERR_INTERNAL  = 5,    /* unexpected C++ exception                    */
    MFLAT_ERR_IO        = 6     /* graph save/load file error (v0.2+)          */
} mflat_status_t;

MFLAT_API const char* mflat_version(void);             /* "0.2.0"             */
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
/* Build-mode knobs, latched at build() (calling after build() affects only a
   later rebuild). residual (default ON) PQ-encodes x - coarse_centroid;
   opq (default OFF, v0.2+) learns a rotation that lowers quantization error
   (slower build, higher recall at the same code size). */
MFLAT_API void mflat_ivfpq_set_residual(mflat_ivfpq_index_t*, int enable);
MFLAT_API void mflat_ivfpq_set_opq(mflat_ivfpq_index_t*, int enable);

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

/* ---------------- GraphIndex (CAGRA-style graph ANN, v0.2+) ----------- */
/* High-recall / low-latency regime (what HNSW owns on CPU). Build once    */
/* (slow: k-NN graph construction), then beam-search on the GPU.           */
typedef struct mflat_graph_index mflat_graph_index_t;  /* opaque */

/* R = graph out-degree (default 32 when <= 0); R+1 must fit MFLAT_MAX_K.  */
MFLAT_API mflat_graph_index_t*
mflat_graph_create(int dim, mflat_metric_t metric, int R, mflat_status_t* out_status);
MFLAT_API void mflat_graph_free(mflat_graph_index_t*);

MFLAT_API int mflat_graph_ready(const mflat_graph_index_t*);  /* 1 after build/load */
MFLAT_API int mflat_graph_size  (const mflat_graph_index_t*);
MFLAT_API int mflat_graph_dim   (const mflat_graph_index_t*);
MFLAT_API int mflat_graph_degree(const mflat_graph_index_t*); /* R */

/* nprobe = build-quality knob for the internal IVF self-search (<= 0 => 64:
   higher = better graph, slower build). */
MFLAT_API mflat_status_t mflat_graph_build(mflat_graph_index_t*, const float* vectors,
        int n, int nprobe);
/* Beam search. L = beam width, the recall/speed knob (<= 0 => 64; >= k).
   max_iter <= 0 => auto. num_start = restart seeds (<= 0 => 32).
   search_width = nodes expanded per iteration (<= 0 => 1, capped at 16).
   k is capped at MFLAT_MAX_K (see *out_k_used). */
MFLAT_API mflat_status_t mflat_graph_search(mflat_graph_index_t*, const float* queries,
        int m, int k, int L, int max_iter, int num_start, int search_width,
        int32_t* out_ids, float* out_distances, int* out_k_used);

/* Persist / restore a built graph (the slow build becomes a one-time cost). */
MFLAT_API mflat_status_t mflat_graph_save(const mflat_graph_index_t*, const char* path);
MFLAT_API mflat_status_t mflat_graph_load(mflat_graph_index_t*, const char* path);

#ifdef __cplusplus
}
#endif
#endif /* MFLAT_H */
