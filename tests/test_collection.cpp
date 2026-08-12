#include <gtest/gtest.h>

#include "lumina/engine/collection.h"
#include "lumina/vector/vector_math.h"

#include <filesystem>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string temp_dir(const std::string& name) {
    const auto p = std::filesystem::temp_directory_path() /
                   ("lumina_collection_" + name + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(p);
    return p.string();
}

std::vector<float> random_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> out(dim);
    for (auto& x : out) {
        x = dist(rng);
    }
    return out;
}

}  // namespace

TEST(CollectionTest, AddGetSearch) {
    const std::string dir = temp_dir("basic");
    constexpr size_t dim = 32;

    lumina::Collection col(dir, dim, lumina::Metric::kL2);
    ASSERT_TRUE(col.open().ok());

    std::mt19937 rng(1);
    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < 200; ++i) {
        vecs.push_back(random_vector(dim, rng));
        ASSERT_TRUE(col.add(i, vecs.back().data(), "payload-" + std::to_string(i)).ok());
    }
    ASSERT_EQ(col.size(), 200U);

    // get returns payload.
    std::string payload;
    ASSERT_TRUE(col.get(7, &payload).ok());
    EXPECT_EQ(payload, "payload-7");

    // duplicate add rejected.
    EXPECT_TRUE(col.add(7, vecs[7].data()).IsInvalidArgument());

    // search finds the exact vector.
    const auto hits = col.search(vecs[5].data(), 1, {.ef_search = 100});
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits.front().id, 5U);
    EXPECT_NEAR(hits.front().distance, 0.0F, 1e-4F);
}

TEST(CollectionTest, RemoveAndUpdate) {
    const std::string dir = temp_dir("mutate");
    constexpr size_t dim = 16;

    lumina::Collection col(dir, dim);
    ASSERT_TRUE(col.open().ok());

    std::mt19937 rng(2);
    const auto v0 = random_vector(dim, rng);
    const auto v1 = random_vector(dim, rng);
    ASSERT_TRUE(col.add(1, v0.data(), "old").ok());
    ASSERT_TRUE(col.add(2, v1.data(), "keep").ok());

    ASSERT_TRUE(col.remove(1).ok());
    EXPECT_EQ(col.size(), 1U);
    std::string payload;
    EXPECT_TRUE(col.get(1, &payload).IsNotFound());

    // update 2 to a new vector and payload.
    ASSERT_TRUE(col.update(2, v0.data(), "new").ok());
    ASSERT_TRUE(col.get(2, &payload).ok());
    EXPECT_EQ(payload, "new");
    const auto hits = col.search(v0.data(), 1, {.ef_search = 100});
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits.front().id, 2U);
}

TEST(CollectionTest, ReopenRestoresData) {
    const std::string dir = temp_dir("reopen");
    constexpr size_t dim = 16;

    {
        lumina::Collection col(dir, dim);
        ASSERT_TRUE(col.open().ok());
        std::mt19937 rng(3);
        std::vector<std::vector<float>> vecs;
        for (uint64_t i = 0; i < 50; ++i) {
            vecs.push_back(random_vector(dim, rng));
            ASSERT_TRUE(col.add(i, vecs.back().data(), "p" + std::to_string(i)).ok());
        }
        // reopen without snapshot: full WAL replay.
    }
    {
        lumina::Collection col(dir, dim);
        ASSERT_TRUE(col.open().ok());
        EXPECT_EQ(col.size(), 50U);
        std::string payload;
        ASSERT_TRUE(col.get(3, &payload).ok());
        EXPECT_EQ(payload, "p3");
    }
}

TEST(CollectionTest, ReopenAfterSnapshot) {
    const std::string dir = temp_dir("snap");
    constexpr size_t dim = 16;

    {
        lumina::Collection col(dir, dim);
        ASSERT_TRUE(col.open().ok());
        std::mt19937 rng(4);
        for (uint64_t i = 0; i < 40; ++i) {
            auto v = random_vector(dim, rng);
            ASSERT_TRUE(col.add(i, v.data(), "s" + std::to_string(i)).ok());
        }
        ASSERT_TRUE(col.snapshot().ok());
    }
    {
        lumina::Collection col(dir, dim);
        ASSERT_TRUE(col.open().ok());
        EXPECT_EQ(col.size(), 40U);
        std::string payload;
        ASSERT_TRUE(col.get(39, &payload).ok());
        EXPECT_EQ(payload, "s39");
    }
}

TEST(CollectionTest, CosineMetricSearch) {
    const std::string dir = temp_dir("cosine");
    constexpr size_t dim = 32;

    lumina::Collection col(dir, dim, lumina::Metric::kCosine);
    ASSERT_TRUE(col.open().ok());

    std::mt19937 rng(5);
    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < 100; ++i) {
        vecs.push_back(random_vector(dim, rng));
        // normalize so cosine order == L2 order is non-trivial
        float n = 0.0F;
        for (const auto x : vecs.back()) {
            n += x * x;
        }
        n = std::sqrt(n);
        for (auto& x : vecs.back()) {
            x /= n;
        }
        ASSERT_TRUE(col.add(i, vecs.back().data()).ok());
    }

    const auto hits = col.search(vecs[0].data(), 1, {.ef_search = 100});
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits.front().id, 0U);
    EXPECT_NEAR(hits.front().distance, 0.0F, 1e-3F);
}

TEST(CollectionTest, PayloadPersistsThroughSnapshotAndReopen) {
    const std::string dir = temp_dir("payload_snap");
    constexpr size_t dim = 8;

    {
        lumina::Collection col(dir, dim);
        ASSERT_TRUE(col.open().ok());
        std::mt19937 rng(6);
        for (uint64_t i = 0; i < 25; ++i) {
            auto v = random_vector(dim, rng);
            ASSERT_TRUE(col.add(i, v.data(), "chunk-" + std::to_string(i)).ok());
        }
        ASSERT_TRUE(col.snapshot().ok());
    }
    {
        lumina::Collection col(dir, dim);
        ASSERT_TRUE(col.open().ok());
        std::string payload;
        ASSERT_TRUE(col.get(11, &payload).ok());
        EXPECT_EQ(payload, "chunk-11");
    }
}
