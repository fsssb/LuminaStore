#include <benchmark/benchmark.h>

#include "lumina/index/quantizer.h"
#include "lumina/vector/vector_math.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

namespace {

constexpr size_t kDim = 128;
constexpr size_t kVecs = 20000;

struct Data {
    std::vector<std::vector<float>> vecs;
    std::vector<const float*> ptrs;
    lumina::QuantConfig cfg_sq8, cfg_bin, cfg_pq;
    std::unique_ptr<lumina::Quantizer> sq8, bin, pq;
    std::vector<lumina::QuantCode> sq8_codes, bin_codes, pq_codes;

    Data() : cfg_sq8{lumina::QuantConfig::Mode::kSQ8, 0, 0},
             cfg_bin{lumina::QuantConfig::Mode::kBinary, 0, 0},
             cfg_pq{lumina::QuantConfig::Mode::kPQ, 16, 256} {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
        vecs.resize(kVecs, std::vector<float>(kDim));
        for (auto& v : vecs) {
            for (auto& x : v) {
                x = dist(rng);
            }
        }
        for (auto& v : vecs) {
            ptrs.push_back(v.data());
        }

        sq8 = lumina::make_quantizer(cfg_sq8);
        bin = lumina::make_quantizer(cfg_bin);
        pq = lumina::make_quantizer(cfg_pq);
        sq8->train(ptrs, kVecs, kDim);
        bin->train(ptrs, kVecs, kDim);
        pq->train(ptrs, kVecs, kDim);

        sq8_codes.resize(kVecs);
        bin_codes.resize(kVecs);
        pq_codes.resize(kVecs);
        for (size_t i = 0; i < kVecs; ++i) {
            sq8->encode(vecs[i].data(), &sq8_codes[i]);
            bin->encode(vecs[i].data(), &bin_codes[i]);
            pq->encode(vecs[i].data(), &pq_codes[i]);
        }
    }
};

Data& data() {
    static Data d;
    return d;
}

void exact_l2(const float* query, const Data& d, std::vector<std::pair<float, size_t>>* out) {
    out->clear();
    out->reserve(kVecs);
    for (size_t i = 0; i < kVecs; ++i) {
        out->push_back({lumina::VectorMath::l2_distance(query, d.vecs[i].data(), kDim), i});
    }
    std::partial_sort(out->begin(), out->begin() + 100, out->end());
    out->resize(100);
}

template <typename Dist>
void quant_topk(const Data& d, const std::vector<lumina::QuantCode>& codes, Dist dist,
                const float* query, size_t k, std::vector<size_t>* out) {
    std::vector<std::pair<float, size_t>> all;
    all.reserve(kVecs);
    for (size_t i = 0; i < kVecs; ++i) {
        all.push_back({dist(codes[i], query), i});
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    out->clear();
    for (size_t i = 0; i < k; ++i) {
        out->push_back(all[i].second);
    }
}

}  // namespace

static void BM_ExactL2(benchmark::State& state) {
    const auto& d = data();
    std::vector<std::pair<float, size_t>> out;
    for (auto _ : state) {
        exact_l2(d.vecs[0].data(), d, &out);
    }
    state.SetItemsProcessed(state.iterations() * kVecs);
}
BENCHMARK(BM_ExactL2);

static void BM_Sq8Distance(benchmark::State& state) {
    const auto& d = data();
    float sink = 0.0F;
    for (auto _ : state) {
        for (size_t i = 0; i < 1000; ++i) {
            sink += d.sq8->distance_to_query(d.sq8_codes[i], d.vecs[i].data(), kDim);
        }
    }
    benchmark::DoNotOptimize(sink);
    state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_Sq8Distance);

// Code-to-code distance: the hot path once queries are pre-quantized (graph
// traversal over stored codes). Pure integer SIMD, no per-call quantization.
static void BM_Sq8CodeCode(benchmark::State& state) {
    const auto& d = data();
    float sink = 0.0F;
    for (auto _ : state) {
        for (size_t i = 1; i <= 1000; ++i) {
            sink += d.sq8->distance(d.sq8_codes[i - 1], d.sq8_codes[i % 1000]);
        }
    }
    benchmark::DoNotOptimize(sink);
    state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_Sq8CodeCode);

static void BM_BinaryDistance(benchmark::State& state) {
    const auto& d = data();
    float sink = 0.0F;
    for (auto _ : state) {
        for (size_t i = 0; i < 1000; ++i) {
            sink += d.bin->distance_to_query(d.bin_codes[i], d.vecs[i].data(), kDim);
        }
    }
    benchmark::DoNotOptimize(sink);
    state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_BinaryDistance);

static void BM_BinaryCodeCode(benchmark::State& state) {
    const auto& d = data();
    float sink = 0.0F;
    for (auto _ : state) {
        for (size_t i = 1; i <= 1000; ++i) {
            sink += d.bin->distance(d.bin_codes[i - 1], d.bin_codes[i % 1000]);
        }
    }
    benchmark::DoNotOptimize(sink);
    state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_BinaryCodeCode);

static void BM_PqDistance(benchmark::State& state) {
    const auto& d = data();
    float sink = 0.0F;
    for (auto _ : state) {
        for (size_t i = 0; i < 1000; ++i) {
            sink += d.pq->distance_to_query(d.pq_codes[i], d.vecs[i].data(), kDim);
        }
    }
    benchmark::DoNotOptimize(sink);
    state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_PqDistance);

static void BM_QuantRecall(benchmark::State& state) {
    const auto& d = data();
    std::vector<std::pair<float, size_t>> truth;
    exact_l2(d.vecs[1].data(), d, &truth);
    std::unordered_set<size_t> truth_ids;
    for (const auto& t : truth) {
        truth_ids.insert(t.second);
    }
    std::vector<size_t> hit;
    for (auto _ : state) {
        quant_topk(d, d.sq8_codes,
                   [&](const lumina::QuantCode& c, const float* q) {
                       return d.sq8->distance_to_query(c, q, kDim);
                   },
                   d.vecs[1].data(), 10, &hit);
        size_t overlap = 0;
        for (const auto h : hit) {
            overlap += truth_ids.count(h) != 0U ? 1 : 0;
        }
        state.counters["recall@10"] = static_cast<double>(overlap) / 10.0;
    }
}
BENCHMARK(BM_QuantRecall);

BENCHMARK_MAIN();
