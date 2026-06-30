// SPDX-License-Identifier: Apache-2.0
// MetalFlat — standalone bench + correctness harness.
//
// Generates random data, runs the GPU FlatIndex against an independent
// CPU brute-force reference, and reports recall + timing. This is the
// "prove it's correct AND fast" artifact — runnable on any M-series Mac
// with no third-party dependencies.
//
//   ./metalflat_bench                  # defaults
//   ./metalflat_bench N D M k          # custom sizes

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "metalflat/FlatIndex.h"
#include "metalflat/IvfIndex.h"

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Recall@k of `got` vs the exact ground truth `truth` (both m×k id
// lists): fraction of returned ids that are in the true top-k.
double recallVsTruth(const std::vector<int32_t>& got,
                     const std::vector<int>& truth, int m, int k) {
    long hits = 0, total = 0;
    for (int qi = 0; qi < m; ++qi) {
        for (int i = 0; i < k; ++i) {
            const int id = got[static_cast<size_t>(qi) * k + i];
            ++total;
            for (int j = 0; j < k; ++j) {
                if (truth[static_cast<size_t>(qi) * k + j] == id) { ++hits; break; }
            }
        }
    }
    return total ? double(hits) / double(total) : 0.0;
}

// Independent CPU brute-force top-k (L2), used only to validate the GPU
// path — deliberately not shared with the library's internals.
void cpuTopkL2(const std::vector<float>& db, int n, int dim,
               const float* q, int k, std::vector<int>& ids) {
    std::vector<std::pair<float, int>> all(n);
    for (int j = 0; j < n; ++j) {
        const float* d = &db[static_cast<size_t>(j) * dim];
        float acc = 0.0f;
        for (int c = 0; c < dim; ++c) { float e = q[c] - d[c]; acc += e * e; }
        all[j] = {acc, j};
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    ids.resize(k);
    for (int i = 0; i < k; ++i) ids[i] = all[i].second;
}

// Generate `n` clustered vectors: pick a random cluster center, add
// Gaussian noise around it. This mimics real embeddings (which cluster
// — similar items land near each other) far better than uniform-random
// data, which has no structure and is the pathological worst case for
// any ANN index. Drawing queries from the same centers models a
// realistic workload (queries resemble database items).
void makeClustered(std::vector<float>& out, int n, int dim,
                   const std::vector<float>& centers, int numClusters,
                   float spread, std::mt19937& rng) {
    std::uniform_int_distribution<int> pickC(0, numClusters - 1);
    std::normal_distribution<float>    noise(0.0f, spread);
    out.assign(static_cast<size_t>(n) * dim, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float* ce = &centers[static_cast<size_t>(pickC(rng)) * dim];
        float* v = &out[static_cast<size_t>(i) * dim];
        for (int d = 0; d < dim; ++d) v[d] = ce[d] + noise(rng);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int N = 32768;   // database vectors
    int D = 96;      // dimension
    int M = 2048;    // queries
    int k = 10;      // neighbours
    if (argc >= 5) {
        N = std::atoi(argv[1]); D = std::atoi(argv[2]);
        M = std::atoi(argv[3]); k = std::atoi(argv[4]);
    }
    // IVF knobs (optional): nlist cells, nprobe cells scanned per query.
    int nlist  = (argc >= 6) ? std::atoi(argv[5])
                             : std::max(64, static_cast<int>(std::sqrt((double)N)));
    int nprobe = (argc >= 7) ? std::atoi(argv[6])
                             : std::max(1, nlist / 16);
    printf("MetalFlat bench — N=%d  D=%d  M=%d  k=%d  metric=L2\n",
           N, D, M, k);
    printf("IVF — nlist=%d  nprobe=%d\n", nlist, nprobe);

    std::mt19937 rng(42);
    // Clustered data (Gaussian blobs) — mimics real embeddings. ANN
    // indexes exploit this structure; uniform-random data has none and
    // makes any index look bad (see the earlier 0.63-recall runs).
    const int   numClusters = std::max(16, static_cast<int>(std::sqrt((double)N)));
    const float spread      = 0.10f;
    std::uniform_real_distribution<float> centerDist(-1.0f, 1.0f);
    std::vector<float> centers(static_cast<size_t>(numClusters) * D);
    for (auto& x : centers) x = centerDist(rng);
    std::vector<float> db, queries;
    makeClustered(db, N, D, centers, numClusters, spread, rng);
    makeClustered(queries, M, D, centers, numClusters, spread, rng);
    printf("data: %d Gaussian clusters, spread %.2f (mimics real embeddings)\n",
           numClusters, spread);

    mflat::FlatIndex index(D, mflat::Metric::L2);
    printf("Metal device: %s\n", index.ready() ? "yes" : "NO (CPU fallback)");
    index.add(db.data(), N);

    // GPU search (one warm-up run, then timed).
    mflat::SearchResult warm = index.search(queries.data(), M, k);
    (void)warm;
    auto t0 = Clock::now();
    mflat::SearchResult res = index.search(queries.data(), M, k);
    const double gpuMs = msSince(t0);

    // CPU reference + timing — MULTI-THREADED so the speedup is measured
    // against a fair baseline (all cores busy), not a single-thread
    // strawman. This is the number that holds up under scrutiny.
    std::vector<int> cpuIds(static_cast<size_t>(M) * k);
    const unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
    auto worker = [&](int lo, int hi) {
        std::vector<int> ids;
        for (int qi = lo; qi < hi; ++qi) {
            cpuTopkL2(db, N, D, &queries[static_cast<size_t>(qi) * D], k, ids);
            for (int i = 0; i < k; ++i)
                cpuIds[static_cast<size_t>(qi) * k + i] = ids[i];
        }
    };
    auto c0 = Clock::now();
    {
        std::vector<std::thread> pool;
        const int chunk = (M + static_cast<int>(nthreads) - 1)
                        / static_cast<int>(nthreads);
        for (unsigned t = 0; t < nthreads; ++t) {
            int lo = static_cast<int>(t) * chunk;
            int hi = std::min(M, lo + chunk);
            if (lo < hi) pool.emplace_back(worker, lo, hi);
        }
        for (auto& th : pool) th.join();
    }
    const double cpuMs = msSince(c0);

    const double flatRecall = recallVsTruth(res.ids, cpuIds, M, k);

    printf("\n== FLAT (exact) ==\n");
    printf("  GPU search : %8.2f ms  (%.0f queries/s)\n",
           gpuMs, M / (gpuMs / 1000.0));
    printf("  CPU search : %8.2f ms  (%u threads, fair baseline)\n",
           cpuMs, nthreads);
    printf("  speedup    : %8.2fx  vs %u-thread CPU\n",
           cpuMs / gpuMs, nthreads);
    printf("  recall@%-2d  : %8.4f\n", k, flatRecall);

    // ---- IVF (approximate) — GPU two-stage: coarse (nprobe nearest
    // centroids) + fine (scan those cells on the GPU). Scans only
    // ~nprobe/nlist of the database; recall < 1 is the tradeoff (raise
    // nprobe to recover it). ---------------------------------------
    mflat::IvfIndex ivf(D, mflat::Metric::L2, nlist);
    auto b0 = Clock::now();
    ivf.build(db.data(), N);
    const double ivfBuildMs = msSince(b0);

    mflat::SearchResult ivfWarm = ivf.search(queries.data(), M, k, nprobe);
    (void)ivfWarm;
    auto i0 = Clock::now();
    mflat::SearchResult ivfRes = ivf.search(queries.data(), M, k, nprobe);
    const double ivfMs = msSince(i0);
    const double ivfRecall = recallVsTruth(ivfRes.ids, cpuIds, M, k);

    printf("\n== IVF (approximate, GPU) ==\n");
    printf("  build      : %8.2f ms  (one-time, k-means)\n", ivfBuildMs);
    printf("  search     : %8.2f ms  (%.0f queries/s)\n",
           ivfMs, M / (ivfMs / 1000.0));
    printf("  recall@%-2d  : %8.4f   <- approximation tradeoff (raise nprobe)\n",
           k, ivfRecall);
    printf("  vs GPU flat: %8.2fx  (exact GPU flat -> approximate GPU IVF)\n",
           gpuMs / ivfMs);
    printf("  vs CPU flat: %8.2fx  (vs %u-thread exact CPU)\n",
           cpuMs / ivfMs, nthreads);

    printf("\nnotes\n");
    printf("  - FLAT recall <1.0 = fp ordering in the GEMM L2 identity, not a bug.\n");
    printf("  - IVF recall is tunable: nprobe up -> recall up, speed down.\n");
    if (flatRecall < 0.99) {
        printf("\nWARNING: FLAT recall %.4f too low — investigate.\n", flatRecall);
        return 1;
    }
    return 0;
}
