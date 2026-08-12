#include <benchmark/benchmark.h>

#include "lumina/engine/collection.h"
#include "lumina/vector/vector_math.h"

#include <filesystem>
#include <random>
#include <unordered_set>
#include <vector>

namespace {

constexpr size_t kDim = 64;
constexpr size_t kN = 10000;

struct Env {
    std::string dir;
    lumina::Collection col;
    std::vector<std::vector<float>> vecs;
    std::vector<lumina::FilterExpr> filters;   // selectivity 1%, 10%, 50%
    std::vector<std::vector<uint64_t>> truth;  // filtered brute-force top-10 per filter

    Env() : dir("/tmp/lumina_filter_bench"), col(dir, kDim) {
        // Clear previous data but keep the directory that col created.
        std::filesystem::remove_all(dir + "/data.wal");
        std::filesystem::remove_all(dir + "/snap");
        col.open();
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
        vecs.resize(kN);
        for (uint64_t i = 0; i < kN; ++i) {
            vecs[i].resize(kDim);
            for (auto& x : vecs[i]) {
                x = dist(rng);
            }
            // cat = i % 100 -> each value covers exactly 1% of the ids.
            const int64_t cat = static_cast<int64_t>(i % 100);
            std::vector<lumina::ScalarField> scalars;
            scalars.push_back({"cat", cat});
            col.add(i, vecs[i].data(), "p", scalars);
        }

        // Selectivities: 1%, 20%, 50%.
        const int64_t targets[3] = {3, 20, 50};
        for (size_t s = 0; s < 3; ++s) {
            lumina::FilterExpr f;
            f.clauses.push_back({"cat", lumina::FilterOp::kEq, targets[s]});
            filters.push_back(f);
            // brute-force filtered top-10
            std::vector<std::pair<float, uint64_t>> all;
            for (uint64_t i = 0; i < kN; ++i) {
                if (i % 100 == static_cast<uint64_t>(targets[s])) {
                    all.push_back({lumina::VectorMath::l2_distance(vecs[0].data(), vecs[i].data(), kDim), i});
                }
            }
            std::partial_sort(all.begin(), all.begin() + 10, all.end());
            all.resize(10);
            truth.emplace_back();
            for (const auto& t : all) {
                truth.back().push_back(t.second);
            }
        }
    }

    double recall(const std::vector<lumina::SearchResult>& hits, size_t s) const {
        std::unordered_set<uint64_t> ids;
        for (const auto& h : hits) {
            ids.insert(h.id);
        }
        size_t hit = 0;
        for (const auto t : truth[s]) {
            hit += ids.count(t) != 0U ? 1 : 0;
        }
        return static_cast<double>(hit) / 10.0;
    }
};

Env& env() {
    static Env e;
    return e;
}

static void BM_InFilter(benchmark::State& state) {
    auto& e = env();
    lumina::SearchOptions opts;
    opts.ef_search = 500;
    opts.filter_mode = lumina::FilterMode::kInFilter;
    double r = 0.0;
    for (auto _ : state) {
        const auto hits = e.col.search_filtered(e.vecs[0].data(), 10, e.filters[state.range(0)], opts);
        benchmark::DoNotOptimize(hits);
        r = e.recall(hits, state.range(0));
    }
    state.counters["recall@10"] = r;
    state.SetItemsProcessed(state.iterations() * 100);  // rough traversal cost
}
BENCHMARK(BM_InFilter)->DenseRange(0, 2);

static void BM_PostFilter(benchmark::State& state) {
    auto& e = env();
    lumina::SearchOptions opts;
    opts.ef_search = 500;
    opts.filter_mode = lumina::FilterMode::kPostFilter;
    double r = 0.0;
    for (auto _ : state) {
        const auto hits = e.col.search_filtered(e.vecs[0].data(), 10, e.filters[state.range(0)], opts);
        benchmark::DoNotOptimize(hits);
        r = e.recall(hits, state.range(0));
    }
    state.counters["recall@10"] = r;
    state.SetItemsProcessed(state.iterations() * 100);
}
BENCHMARK(BM_PostFilter)->DenseRange(0, 2);

}  // namespace

BENCHMARK_MAIN();
