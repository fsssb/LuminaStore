#include <gtest/gtest.h>

#include "lumina/vector/hnsw_index.h"
#include "lumina/vector/vector_math.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>
#include <unistd.h>
#include <unordered_set>
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

// Negative inner product: smaller = better (larger dot product).
float ip_neg(const float* a, const float* b, size_t dim) {
    float s = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return -s;
}

std::string temp_path(const std::string& name) {
    const auto p = std::filesystem::temp_directory_path() /
                   ("lumina_hnswv2_" + name + "_" + std::to_string(::getpid()) + ".bin");
    return p.string();
}

std::vector<std::pair<float, uint64_t>> brute_topk(const float* query,
                                                   const std::vector<std::vector<float>>& vecs,
                                                   size_t k, lumina::DistanceFn dist) {
    std::vector<std::pair<float, uint64_t>> all;
    all.reserve(vecs.size());
    for (size_t i = 0; i < vecs.size(); ++i) {
        all.push_back({dist(query, vecs[i].data(), vecs[i].size()), i});
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    all.resize(k);
    return all;
}

double recall_at_k(const lumina::HNSWIndex& index,
                   const std::vector<std::vector<float>>& vecs,
                   const std::vector<std::vector<float>>& queries,
                   size_t k, size_t ef, lumina::DistanceFn dist,
                   const std::unordered_set<uint64_t>& excluded = {}) {
    size_t hit = 0, total = 0;
    for (const auto& q : queries) {
        const auto truth = brute_topk(q.data(), vecs, k, dist);
        const auto ann = index.search_top_k(q.data(), k, ef);
        std::unordered_set<uint64_t> ann_ids;
        for (const auto& r : ann) {
            ann_ids.insert(r.id);
        }
        for (size_t i = 0; i < k; ++i) {
            if (excluded.count(truth[i].second) != 0U) {
                continue;  // skip truth entries that were excluded (e.g. deleted)
            }
            ++total;
            if (ann_ids.count(truth[i].second) != 0U) {
                ++hit;
            }
        }
    }
    return total == 0 ? 1.0 : static_cast<double>(hit) / static_cast<double>(total);
}

}  // namespace

TEST(HNSWIndexV2Test, HeuristicRecallHigh) {
    constexpr size_t dim = 64;
    constexpr size_t n = 2000;
    constexpr size_t q = 100;
    constexpr size_t k = 10;

    std::mt19937 rng(11);
    lumina::HNSWIndex index(dim, 16, 200);
    std::vector<std::vector<float>> vecs;
    vecs.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        vecs.push_back(random_vector(dim, rng));
        ASSERT_TRUE(index.add_item(i, vecs.back().data()).ok());
    }

    std::vector<std::vector<float>> queries;
    for (size_t i = 0; i < q; ++i) {
        queries.push_back(random_vector(dim, rng));
    }

    // With ef=200 heuristic selection should comfortably exceed 0.95 recall@10.
    const double recall = recall_at_k(index, vecs, queries, k, 200, lumina::VectorMath::l2_distance);
    EXPECT_GE(recall, 0.95) << "heuristic neighbor selection recall too low";
}

TEST(HNSWIndexV2Test, DeleteExcludesFromSearch) {
    constexpr size_t dim = 32;
    constexpr size_t n = 500;

    std::mt19937 rng(3);
    lumina::HNSWIndex index(dim, 16, 200);
    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < n; ++i) {
        vecs.push_back(random_vector(dim, rng));
        ASSERT_TRUE(index.add_item(i, vecs.back().data()).ok());
    }
    ASSERT_EQ(index.size(), n);

    // Delete ids [0, 100).
    for (uint64_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(index.remove(i).ok());
    }
    EXPECT_EQ(index.size(), n - 100);

    // Removing again is idempotent; removing unknown id is NotFound.
    EXPECT_TRUE(index.remove(0).ok());
    EXPECT_TRUE(index.remove(99999).IsNotFound());

    std::unordered_set<uint64_t> excluded;
    for (uint64_t i = 0; i < 100; ++i) {
        excluded.insert(i);
    }
    std::vector<std::vector<float>> queries;
    for (size_t i = 0; i < 50; ++i) {
        queries.push_back(random_vector(dim, rng));
    }
    const double recall = recall_at_k(index, vecs, queries, 10, 200, lumina::VectorMath::l2_distance,
                                      excluded);
    EXPECT_GE(recall, 0.95);

    // Deleted ids must never appear in results.
    for (const auto& q : queries) {
        for (const auto& r : index.search_top_k(q.data(), 20, 200)) {
            EXPECT_GE(r.id, 100U) << "deleted id returned in search results";
        }
    }
}

TEST(HNSWIndexV2Test, UpdateMovesVector) {
    constexpr size_t dim = 32;

    std::mt19937 rng(5);
    lumina::HNSWIndex index(dim, 16, 200);
    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < 300; ++i) {
        vecs.push_back(random_vector(dim, rng));
        ASSERT_TRUE(index.add_item(i, vecs.back().data()).ok());
    }

    // Update id 0 to be exactly query target; it must become the top-1 result.
    const auto target = random_vector(dim, rng);
    ASSERT_TRUE(index.update_item(0, target.data()).ok());
    const auto res = index.search_top_k(target.data(), 1, 100);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res.front().id, 0U);
    EXPECT_NEAR(res.front().distance, 0.0F, 1e-4F);

    // Updating an unknown id is NotFound.
    EXPECT_TRUE(index.update_item(99999, target.data()).IsNotFound());
}

TEST(HNSWIndexV2Test, UpdateRevivesDeleted) {
    constexpr size_t dim = 16;
    lumina::HNSWIndex index(dim, 16, 200);
    const std::vector<float> v(dim, 0.5F);
    ASSERT_TRUE(index.add_item(1, v.data()).ok());
    ASSERT_TRUE(index.remove(1).ok());
    EXPECT_EQ(index.size(), 0U);

    ASSERT_TRUE(index.update_item(1, v.data()).ok());
    EXPECT_EQ(index.size(), 1U);
    const auto res = index.search_top_k(v.data(), 1, 100);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res.front().id, 1U);
}

TEST(HNSWIndexV2Test, CustomDistanceIp) {
    constexpr size_t dim = 32;
    constexpr size_t n = 500;
    constexpr size_t k = 5;

    std::mt19937 rng(9);
    std::uniform_real_distribution<float> pos(0.0F, 1.0F);  // non-negative for stable MIPS
    lumina::HNSWIndex index(dim, 16, 200, ip_neg);
    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < n; ++i) {
        vecs.emplace_back(dim);
        for (auto& x : vecs.back()) {
            x = pos(rng);
        }
        ASSERT_TRUE(index.add_item(i, vecs.back().data()).ok());
    }

    size_t hit = 0, total = 0;
    for (int qi = 0; qi < 20; ++qi) {
        std::vector<float> q(dim);
        for (auto& x : q) {
            x = pos(rng);
        }
        const auto truth = brute_topk(q.data(), vecs, k, ip_neg);
        const auto ann = index.search_top_k(q.data(), k, 200);
        ASSERT_EQ(ann.size(), k);
        std::unordered_set<uint64_t> ids;
        for (const auto& r : ann) {
            ids.insert(r.id);
        }
        for (size_t i = 0; i < k; ++i) {
            ++total;
            hit += ids.count(truth[i].second) != 0U ? 1 : 0;
        }
    }
    // MIPS on raw vectors: IP is not a metric, so a small recall loss vs L2 is
    // expected; assert a healthy majority of the brute-force top-k is found.
    EXPECT_GE(static_cast<double>(hit) / static_cast<double>(total), 0.9);
}

TEST(HNSWIndexV2Test, SaveLoadPreservesDeletes) {
    constexpr size_t dim = 32;
    constexpr size_t n = 400;

    std::mt19937 rng(21);
    lumina::HNSWIndex index(dim, 16, 200);
    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < n; ++i) {
        vecs.push_back(random_vector(dim, rng));
        ASSERT_TRUE(index.add_item(i, vecs.back().data()).ok());
    }
    for (uint64_t i = 0; i < 50; ++i) {
        ASSERT_TRUE(index.remove(i).ok());
    }

    const std::string path = temp_path("save_delete");
    ASSERT_TRUE(index.save(path).ok());

    lumina::HNSWIndex loaded(dim, 16, 200);
    ASSERT_TRUE(loaded.load(path).ok());
    EXPECT_EQ(loaded.size(), n - 50);

    auto qv = vecs[100];
    const auto res = loaded.search_top_k(qv.data(), 10, 100);
    for (const auto& r : res) {
        EXPECT_GE(r.id, 50U) << "deleted id loaded back and returned";
    }
    std::filesystem::remove(path);
}

TEST(HNSWIndexV2Test, RejectsBadFile) {
    const std::string path = temp_path("bad_file");
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs << "this is not a valid hnsw file at all....";
    }
    lumina::HNSWIndex index(16, 16, 200);
    EXPECT_TRUE(index.load(path).IsCorruption());
    std::filesystem::remove(path);
}

TEST(HNSWIndexV2Test, ConcurrentAddsAllVisible) {
    constexpr size_t dim = 32;
    constexpr size_t n = 4000;
    constexpr int nthreads = 4;

    std::mt19937 rng(77);
    std::vector<std::vector<float>> vecs(n);
    for (size_t i = 0; i < n; ++i) {
        vecs[i] = random_vector(dim, rng);
    }

    lumina::HNSWIndex index(dim, 16, 200);
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = static_cast<size_t>(t); i < n; i += nthreads) {
                ASSERT_TRUE(index.add_item(i, vecs[i].data()).ok());
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_EQ(index.size(), n);

    // Post-condition: queries still find their true nearest neighbours.
    double hit = 0;
    for (size_t qi = 0; qi < 50; ++qi) {
        auto q = random_vector(dim, rng);
        const auto truth = brute_topk(q.data(), vecs, 5, lumina::VectorMath::l2_distance);
        const auto ann = index.search_top_k(q.data(), 5, 200);
        std::unordered_set<uint64_t> ids;
        for (const auto& r : ann) {
            ids.insert(r.id);
        }
        for (size_t i = 0; i < 5; ++i) {
            hit += ids.count(truth[i].second) != 0U ? 1.0 : 0.0;
        }
    }
    EXPECT_GE(hit / (50.0 * 5.0), 0.9);
}

TEST(HNSWIndexV2Test, ConcurrentSearchSafe) {
    constexpr size_t dim = 16;
    constexpr size_t n = 1000;

    std::mt19937 rng(31);
    lumina::HNSWIndex index(dim, 16, 200);
    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < n; ++i) {
        vecs.push_back(random_vector(dim, rng));
        ASSERT_TRUE(index.add_item(i, vecs.back().data()).ok());
    }

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> total_results{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 qrng(100 + t);
            while (!stop.load()) {
                auto q = random_vector(dim, qrng);
                const auto res = index.search_top_k(q.data(), 10, 100);
                total_results.fetch_add(res.size());
            }
        });
    }

    // Concurrently add/remove while searches run.
    for (uint64_t i = 0; i < 2000; ++i) {
        auto v = random_vector(dim, rng);
        ASSERT_TRUE(index.update_item(i % n, v.data()).ok());  // exercise write path
        if (i % 5 == 0) {
            index.remove(i % n);
            ASSERT_TRUE(index.update_item(i % n, v.data()).ok());  // revive
        }
    }
    stop.store(true);
    for (auto& th : threads) {
        th.join();
    }
    EXPECT_GT(total_results.load(), 0U);
    EXPECT_EQ(index.size(), n);
}
