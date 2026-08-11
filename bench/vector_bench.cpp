#include <benchmark/benchmark.h>

#include "lumina/vector/aligned_alloc.h"
#include "lumina/vector/hnsw_index.h"
#include "lumina/vector/vector_math.h"

#include <cstdint>
#include <random>
#include <vector>

namespace {

using lumina::AlignedFloatVector;

void fill_random(float* a, size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (size_t i = 0; i < n; ++i) {
        a[static_cast<size_t>(i)] = dist(rng);
    }
}

void fill_random(AlignedFloatVector& v, std::mt19937& rng) {
    fill_random(v.data(), v.size(), rng);
}

// 在 float 对象数组上使用非首元素作为视图起点，保证「float 对象」与生存期，避免 char 强转
struct MisalignedBlock {
    std::vector<float> storage_a;
    std::vector<float> storage_b;
    float* a = nullptr;
    float* b = nullptr;

    void reset(size_t dim, std::mt19937& rng) {
        storage_a.assign(dim + 8U, 0.0F);
        storage_b.assign(dim + 8U, 0.0F);
        // 使用 vector 内合法 float 子区间（从第二个元素起），+4B 通常可破坏 32B/64B 块对齐
        a = &storage_a[1];
        b = &storage_b[1];
        fill_random(a, dim, rng);
        fill_random(b, dim, rng);
    }
};

static void BM_L2Naive(benchmark::State& state) {
    const size_t dim = static_cast<size_t>(state.range(0));
    std::mt19937 rng(7U + static_cast<unsigned int>(dim));
    AlignedFloatVector a(dim);
    AlignedFloatVector b(dim);
    fill_random(a, rng);
    fill_random(b, rng);
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            lumina::VectorMath::l2_distance_naive(a.data(), b.data(), dim));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim));
}

static void BM_L2DispatchedAligned(benchmark::State& state) {
    const size_t dim = static_cast<size_t>(state.range(0));
    std::mt19937 rng(8U + static_cast<unsigned int>(dim));
    AlignedFloatVector a(dim);
    AlignedFloatVector b(dim);
    fill_random(a, rng);
    fill_random(b, rng);
    lumina::VectorMath::init();
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            lumina::VectorMath::l2_distance(a.data(), b.data(), dim));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim));
}

static void BM_L2DispatchedMisaligned(benchmark::State& state) {
    const size_t dim = static_cast<size_t>(state.range(0));
    std::mt19937 rng(9U + static_cast<unsigned int>(dim));
    MisalignedBlock blk;
    blk.reset(dim, rng);
    lumina::VectorMath::init();
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            lumina::VectorMath::l2_distance(blk.a, blk.b, dim));
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim));
}

BENCHMARK(BM_L2Naive)->Arg(127)->Arg(128)->Arg(129)->Arg(512)->Arg(1024);
BENCHMARK(BM_L2DispatchedAligned)->Arg(127)->Arg(128)->Arg(129)->Arg(512)->Arg(1024);
BENCHMARK(BM_L2DispatchedMisaligned)->Arg(127)->Arg(128)->Arg(129)->Arg(512)->Arg(1024);

constexpr size_t kHnswDim = 128;
constexpr size_t kHnswN = 10000U;

static void BM_HNSWSearchTopK(benchmark::State& state) {
    std::mt19937 rng(9);
    lumina::HNSWIndex index(kHnswDim, 16, 200);
    std::vector<AlignedFloatVector> dataset;
    dataset.reserve(kHnswN);
    for (size_t i = 0; i < kHnswN; ++i) {
        AlignedFloatVector v(kHnswDim);
        fill_random(v, rng);
        dataset.push_back(std::move(v));
        const auto s = index.add_item(static_cast<uint64_t>(i), dataset.back().data());
        if (!s.ok()) {
            state.SkipWithError(s.ToString().c_str());
            return;
        }
    }
    AlignedFloatVector query(kHnswDim);
    fill_random(query, rng);
    for (auto _ : state) {
        const auto result = index.search_top_k(query.data(), 10, 100);
        float sink = 0.0F;
        if (!result.empty()) {
            sink = result.front().distance;
        }
        benchmark::DoNotOptimize(sink);
    }
}

BENCHMARK(BM_HNSWSearchTopK);

}  // namespace

BENCHMARK_MAIN();
