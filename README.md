# metal-flat

GPU-accelerated vector search for **Apple Silicon**, built on Metal.

Two index types behind one Faiss-shaped C++ API:

- **`FlatIndex`** — *exact* brute-force search (100% recall), accelerated
  with MetalPerformanceShaders GEMM + a tiled top-k. The production-ready
  fast path.
- **`IvfIndex`** — *approximate* search (IVF / k-means), trading tunable
  recall for scanning a small fraction of the database.

Metrics: **L2** (squared Euclidean), **Inner Product**, **Cosine**.

> Status: early but real. Numbers below are measured on an M2 Pro and
> reproducible with the bundled benchmark. Apple Silicon only. The
> public headers are plain C++17 (Metal is hidden behind a pimpl);
> the C++ namespace is `mflat`.

---

## Quick start

```sh
cmake -S . -B build -G Ninja
cmake --build build

# FLAT (exact) + IVF (approximate), validated against a CPU reference:
./build/metalflat_bench                       # defaults: N=32768 D=96 M=2048 k=10
./build/metalflat_bench 65536 128 4096 10     # N D M k
./build/metalflat_bench 65536 128 4096 10 256 8   # ... nlist nprobe (IVF knobs)
```

The bench builds both indexes, checks results against an independent
multi-threaded CPU brute-force reference, and reports recall + timing.

---

## API

Exact search:

```cpp
#include "metalflat/FlatIndex.h"

mflat::FlatIndex index(/*dim=*/128, mflat::Metric::L2);
index.add(database, n);                        // n * dim row-major floats
mflat::SearchResult r = index.search(queries, m, /*k=*/10);
// r.ids[q*k + j], r.distances[q*k + j] — j-th neighbour of query q (nearest first)
```

Approximate search:

```cpp
#include "metalflat/IvfIndex.h"

mflat::IvfIndex ivf(/*dim=*/128, mflat::Metric::L2, /*nlist=*/1024);
ivf.build(database, n);                          // k-means + inverted lists (one-time)
auto r = ivf.search(queries, m, /*k=*/10, /*nprobe=*/16);
// nprobe is the recall/speed dial: higher = more recall, less speed.
```

If no Metal device is available, both `search()` paths fall back to an
exact CPU implementation — results are always correct, just slower.

---

## Performance (measured, M2 Pro)

`N=65536, D=128, M=4096, k=10`, clustered data:

| Path | Latency | Recall | Notes |
|------|---------|--------|-------|
| FLAT — GPU | **~54 ms** | 1.00 | exact |
| FLAT — CPU (10 threads) | ~2460 ms | 1.00 | fair baseline |
| **FLAT speedup** | **~48x** | | vs all-core CPU |
| IVF — nprobe=1 (scans 0.4%) | ~33 ms | 0.95 | approximate |
| IVF — nprobe=8 (scans 3%) | — | 1.00 | approximate |

**FlatIndex** is the strong path: ~48x over a *fair* multi-threaded CPU
baseline, exact recall, and it **tiles over the database** so memory
stays bounded (~128 MB) at any N — verified at N=500k where a naive
full-matrix approach would need gigabytes.

**IvfIndex** is correct and tunable (recall climbs with `nprobe`). One
honest caveat on *speed*: IVF deliberately removes almost all the
compute (it scans a tiny fraction of the data), so at small workloads
the **fixed GPU overhead** — dispatch, sync, allocation — dominates the
sub-millisecond scan, and CPU IVF is competitive. IVF's GPU advantage
grows with workload size; flat is the fast path to reach for today. (A
fused, allocation-free IVF pipeline is on the roadmap.)

---

## How it works

**Flat** reformulates exact search as a matrix multiply: the
query×database distance matrix is `G = Q·Dᵀ` (MPS GEMM, near-peak FP32),
with L2 reconstructed from `‖q−d‖² = ‖q‖² + ‖d‖² − 2·(q·d)`. G is
computed in **column-tiles**, each folded straight into a running
per-query top-k, so the full M×N matrix is never materialized.

**IVF** clusters the database into `nlist` cells (k-means), then a query
scans only its `nprobe` nearest cells instead of everything — doing
~`nlist/nprobe` times less work, at the cost of missing neighbours in
unprobed cells (recall < 1, tunable).

---

## Benchmarking honestly

Two things the bundled bench gets right, and you should insist on
anywhere:

1. **Fair baselines.** Speedups are measured against a **multi-threaded**
   CPU reference (all cores), not a single-thread strawman.
2. **Realistic data.** The bench generates **clustered** vectors
   (Gaussian blobs), which mimic real embeddings. *Uniform-random data
   is the worst case for any ANN index* — it has no structure to exploit,
   so IVF recall collapses on it. (On random data IVF needed nprobe=64
   for 0.63 recall; on clustered data nprobe=1 gives 0.95.) For
   publishable numbers, use a standard dataset (SIFT1M / GloVe).

---

## Limitations (current)

- **Apple Silicon only** (Metal). Kernels compile from source at runtime,
  so they adapt to the present GPU (M1–M4+); no per-chip code.
- **`k ≤ 64`** on the GPU path (fixed per-thread top-k buffers); larger
  `k` is clamped.
- Flat L2 uses the `‖q‖²+‖d‖²−2q·d` identity — exact up to floating-point
  ordering, not bit-exact (recall ~1.0, occasionally 0.999x on ties).
- IVF coarse quantizer clusters by L2; Inner-Product recall is therefore
  suboptimal (fine for L2 / Cosine).
- IVF `nprobe > 64` falls back to the CPU path.
- GPU IVF is overhead-bound at small workloads (see Performance).

---

## Roadmap

- Flat: warp-select top-k + fp16 (more exact-search speed).
- IVF: fused, allocation-free single-dispatch GPU pipeline + parallel
  fine-scan kernel (to realize the GPU speedup at all workload sizes).
- Quantization (SQ / PQ) for memory-bound datasets.
- Graph index (HNSW-style).
- Bindings (C API → Python/Swift) and a Faiss-CPU head-to-head on
  standard datasets.

---

## License

See [`LICENSE`](LICENSE). **Not** open-source yet — the model
(proprietary / open-core / source-available) is undecided. Do not
distribute until that's settled. Developed independently of, and not
covered by, any surrounding repository's license.
