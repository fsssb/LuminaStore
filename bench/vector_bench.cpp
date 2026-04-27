#include <benchmark/benchmark.h>

#include "lumina/vector/hnsw_index.h"
#include "lumina/vector/vector_math.h"

#include <random>
#include <vector>

namespace {

constexpr size_t kDim = 128;
constexpr size_t kN = 10000;

std::vector<float> random_vector(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> v(kDim);
    for (size_t i = 0; i < kDim; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

static void BM_L2Naive(benchmark::State& state) {
    std::mt19937 rng(7);
    auto a = random_vector(rng);
    auto b = random_vector(rng);

    for (auto _ : state) {
        benchmark::DoNotOptimize(lumina::VectorMath::l2_distance_naive(a.data(), b.data(), kDim));
    }
}

static void BM_L2Dispatched(benchmark::State& state) {
    std::mt19937 rng(8);
    auto a = random_vector(rng);
    auto b = random_vector(rng);
    lumina::VectorMath::init();

    for (auto _ : state) {
        benchmark::DoNotOptimize(lumina::VectorMath::l2_distance(a.data(), b.data(), kDim));
    }
}

static void BM_HNSWSearchTopK(benchmark::State& state) {
    std::mt19937 rng(9);
    lumina::HNSWIndex index(kDim, 16, 200);

    std::vector<std::vector<float>> dataset;
    dataset.reserve(kN);
    for (size_t i = 0; i < kN; ++i) {
        dataset.push_back(random_vector(rng));
        auto s = index.add_item(static_cast<uint64_t>(i), dataset.back().data());
        if (!s.ok()) {
            state.SkipWithError(s.ToString().c_str());
            return;
        }
    }

    auto query = random_vector(rng);

    for (auto _ : state) {
        auto result = index.search_top_k(query.data(), 10, 100);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(BM_L2Naive);
BENCHMARK(BM_L2Dispatched);
BENCHMARK(BM_HNSWSearchTopK);

}  // namespace

BENCHMARK_MAIN();
