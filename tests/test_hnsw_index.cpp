#include <gtest/gtest.h>

#include "lumina/vector/hnsw_index.h"
#include "lumina/vector/vector_math.h"

#include <algorithm>
#include <filesystem>
#include <random>
#include <unordered_set>
#include <unistd.h>
#include <vector>

namespace {

std::vector<float> random_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> out(dim);
    for (size_t i = 0; i < dim; ++i) {
        out[i] = dist(rng);
    }
    return out;
}

}  // namespace

TEST(HNSWIndexTest, BasicInsertAndSearch) {
    constexpr size_t dim = 64;
    std::mt19937 rng(123);

    lumina::HNSWIndex index(dim, 16, 200);
    std::vector<std::vector<float>> vectors;
    vectors.reserve(100);

    for (uint64_t i = 0; i < 100; ++i) {
        vectors.push_back(random_vector(dim, rng));
        ASSERT_TRUE(index.add_item(i, vectors.back().data()).ok());
    }

    for (uint64_t i = 0; i < 10; ++i) {
        const auto result = index.search_top_k(vectors[i].data(), 1, 100);
        ASSERT_FALSE(result.empty());
        EXPECT_EQ(result.front().id, i);
    }
}

TEST(HNSWIndexTest, RecallTest) {
    constexpr size_t dim = 128;
    constexpr size_t n = 1000;
    constexpr size_t k = 10;
    constexpr size_t q = 50;

    std::mt19937 rng(7);
    lumina::HNSWIndex index(dim, 16, 200);

    std::vector<std::vector<float>> vectors;
    vectors.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        vectors.push_back(random_vector(dim, rng));
        ASSERT_TRUE(index.add_item(i, vectors.back().data()).ok());
    }

    double total_recall = 0.0;
    for (size_t qi = 0; qi < q; ++qi) {
        auto query = random_vector(dim, rng);

        std::vector<std::pair<float, uint64_t>> brute;
        brute.reserve(n);
        for (uint64_t i = 0; i < n; ++i) {
            brute.push_back({lumina::VectorMath::l2_distance_naive(query.data(), vectors[i].data(), dim), i});
        }
        std::partial_sort(brute.begin(), brute.begin() + k, brute.end());

        const auto ann = index.search_top_k(query.data(), k, 120);
        std::unordered_set<uint64_t> ann_ids;
        for (const auto& item : ann) {
            ann_ids.insert(item.id);
        }

        size_t hit = 0;
        for (size_t i = 0; i < k; ++i) {
            if (ann_ids.count(brute[i].second) != 0U) {
                ++hit;
            }
        }
        total_recall += static_cast<double>(hit) / static_cast<double>(k);
    }

    const double recall = total_recall / static_cast<double>(q);
    EXPECT_GE(recall, 0.80);
}

TEST(HNSWIndexTest, SaveAndLoad) {
    constexpr size_t dim = 64;
    std::mt19937 rng(8);
    lumina::HNSWIndex index(dim, 16, 200);

    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < 500; ++i) {
        vecs.push_back(random_vector(dim, rng));
        ASSERT_TRUE(index.add_item(i, vecs.back().data()).ok());
    }

    const auto path = (std::filesystem::temp_directory_path() /
                      ("lumina_hnsw_" + std::to_string(::getpid()) + ".bin"));

    ASSERT_TRUE(index.save(path.string()).ok());

    lumina::HNSWIndex loaded(dim, 16, 200);
    ASSERT_TRUE(loaded.load(path.string()).ok());

    auto qv = vecs[42];
    const auto a = index.search_top_k(qv.data(), 5, 100);
    const auto b = loaded.search_top_k(qv.data(), 5, 100);

    ASSERT_EQ(a.size(), b.size());
    ASSERT_FALSE(a.empty());
    EXPECT_EQ(a[0].id, b[0].id);

    std::filesystem::remove(path);
}

TEST(HNSWIndexTest, EmptyIndex) {
    lumina::HNSWIndex index(32, 16, 200);
    std::vector<float> q(32, 0.1F);
    const auto res = index.search_top_k(q.data(), 10, 50);
    EXPECT_TRUE(res.empty());
    EXPECT_EQ(index.size(), 0U);
}
