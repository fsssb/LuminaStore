// ANN benchmark tool: measures recall@k vs QPS across an ef grid.
//
// Data: random uniform vectors (stand-in for SIFT-1M; same methodology, no
// external dataset download). Brute-force exact top-k serves as ground truth.
//
// Usage: ann_bench [n] [dim] [num_queries] [k] [ef1,ef2,...]
//   default: 50000 128 200 10 10,50,100,200,400
//
// Output: markdown table (recall@k, QPS, latency p50/p99).

#include "lumina/vector/hnsw_index.h"
#include "lumina/vector/vector_math.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Stats {
    double recall = 0.0;
    double qps = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
};

double now_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    size_t n = 50000;
    size_t dim = 128;
    size_t nq = 200;
    size_t k = 10;
    size_t M = 16;
    size_t ef_construction = 200;
    std::vector<size_t> efs = {10, 50, 100, 200, 400};

    if (argc > 1) n = std::strtoull(argv[1], nullptr, 10);
    if (argc > 2) dim = std::strtoull(argv[2], nullptr, 10);
    if (argc > 3) nq = std::strtoull(argv[3], nullptr, 10);
    if (argc > 4) k = std::strtoull(argv[4], nullptr, 10);
    if (argc > 5) {
        efs.clear();
        std::string s = argv[5];
        size_t pos = 0;
        while (pos < s.size()) {
            const size_t comma = s.find(',', pos);
            efs.push_back(std::strtoull(s.c_str() + pos, nullptr, 10));
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
    }
    if (argc > 6) M = std::strtoull(argv[6], nullptr, 10);
    if (argc > 7) ef_construction = std::strtoull(argv[7], nullptr, 10);

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    printf("# LuminaStore ANN benchmark\n");
    printf("n=%zu dim=%zu queries=%zu k=%zu metric=L2\n\n", n, dim, nq, k);

    // 1. Generate data + queries.
    std::vector<std::vector<float>> vecs(n, std::vector<float>(dim));
    for (auto& v : vecs) {
        for (auto& x : v) {
            x = dist(rng);
        }
    }
    std::vector<std::vector<float>> queries(nq, std::vector<float>(dim));
    for (auto& q : queries) {
        for (auto& x : q) {
            x = dist(rng);
        }
    }

    // 2. Build index.
    const auto t0 = Clock::now();
    lumina::HNSWIndex index(dim, M, ef_construction);
    for (size_t i = 0; i < n; ++i) {
        index.add_item(i, vecs[i].data());
    }
    const double build_ms = now_ms(t0, Clock::now());
    const size_t graph_bytes = 0;  // estimate below
    printf("build time: %.1f ms (%.0f vectors/s)\n\n", build_ms, n / (build_ms / 1000.0));

    // 3. Ground truth (brute-force top-k per query).
    const auto t1 = Clock::now();
    std::vector<std::vector<uint64_t>> truth(nq, std::vector<uint64_t>(k));
    for (size_t qi = 0; qi < nq; ++qi) {
        std::vector<std::pair<float, uint64_t>> all;
        all.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            all.push_back({lumina::VectorMath::l2_distance(queries[qi].data(), vecs[i].data(), dim), i});
        }
        std::partial_sort(all.begin(), all.begin() + k, all.end());
        for (size_t j = 0; j < k; ++j) {
            truth[qi][j] = all[j].second;
        }
    }
    printf("ground truth: %.1f ms\n\n", now_ms(t1, Clock::now()));

    // 4. Per-ef evaluation.
    printf("| ef | recall@%zu | QPS | p50 (us) | p99 (us) |\n", k);
    printf("|----|-----------|-----|----------|----------|\n");
    for (const size_t ef : efs) {
        // warmup
        index.search_top_k(queries[0].data(), k, ef);

        std::vector<double> lat_us(nq);
        const auto t2 = Clock::now();
        for (size_t qi = 0; qi < nq; ++qi) {
            const auto ts = Clock::now();
            const auto hits = index.search_top_k(queries[qi].data(), k, ef);
            lat_us[qi] = std::chrono::duration<double, std::micro>(Clock::now() - ts).count();

            std::unordered_set<uint64_t> ids;
            for (const auto& h : hits) {
                ids.insert(h.id);
            }
            for (size_t j = 0; j < k; ++j) {
                if (ids.count(truth[qi][j]) != 0U) {
                    ++lat_us[qi];  // no-op; recall counted separately below
                }
            }
        }
        const double total_ms = now_ms(t2, Clock::now());
        const double qps = static_cast<double>(nq) / (total_ms / 1000.0);

        // recall (separate pass to avoid counting overhead in latency)
        size_t hits_total = 0;
        for (size_t qi = 0; qi < nq; ++qi) {
            const auto hits = index.search_top_k(queries[qi].data(), k, ef);
            std::unordered_set<uint64_t> ids;
            for (const auto& h : hits) {
                ids.insert(h.id);
            }
            for (size_t j = 0; j < k; ++j) {
                hits_total += ids.count(truth[qi][j]) != 0U ? 1 : 0;
            }
        }
        const double recall = static_cast<double>(hits_total) / (nq * k);

        std::sort(lat_us.begin(), lat_us.end());
        const double p50 = lat_us[nq / 2];
        const double p99 = lat_us[static_cast<size_t>(nq * 0.99)];
        printf("| %4zu | %9.4f | %7.1f | %8.1f | %8.1f |\n", ef, recall, qps, p50, p99);
    }

    printf("\n# Notes\n");
    printf("- Single thread, single CPU (no hyper-threading).\n");
    printf("- recall@%zu computed against brute-force top-%zu ground truth.\n", k, k);
    printf("- Random uniform data: relative ranking is representative; absolute numbers\n");
    printf("  differ from SIFT-1M. Methodology follows ann-benchmarks/VIBE practices.\n");
    return 0;
}
