#include <gtest/gtest.h>

#include "lumina/index/quantizer.h"

#include <cmath>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace {

std::vector<std::vector<float>> random_vectors(size_t count, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<std::vector<float>> out(count, std::vector<float>(dim));
    for (auto& v : out) {
        for (auto& x : v) {
            x = dist(rng);
        }
    }
    return out;
}

float l2(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0F;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

}  // namespace

TEST(QuantizerTest, Sq8DistanceApproximatesL2) {
    constexpr size_t dim = 64;
    const auto vecs = random_vectors(200, dim, 1);
    std::vector<const float*> samples;
    for (const auto& v : vecs) {
        samples.push_back(v.data());
    }

    lumina::QuantConfig cfg;
    cfg.mode = lumina::QuantConfig::Mode::kSQ8;
    auto q = lumina::make_quantizer(cfg);
    ASSERT_NE(q, nullptr);
    ASSERT_TRUE(q->train(samples, samples.size(), dim).ok());
    EXPECT_EQ(q->code_bytes(), dim);

    // Quantized distance should be close to exact L2 (relative error small).
    const auto& query = vecs[0];
    lumina::QuantCode code;
    q->encode(vecs[50].data(), &code);
    const float approx = q->distance_to_query(code, query.data(), dim);
    const float exact = l2(query, vecs[50]);
    EXPECT_NEAR(approx, exact, exact * 0.2F + 1.0F);

    // code-to-code distance symmetric.
    lumina::QuantCode code2;
    q->encode(vecs[60].data(), &code2);
    const float d1 = q->distance(code, code2);
    const float d2 = q->distance(code2, code);
    EXPECT_NEAR(d1, d2, 1e-3F);
}

TEST(QuantizerTest, BinaryHamming) {
    constexpr size_t dim = 16;
    const auto vecs = random_vectors(10, dim, 2);

    lumina::QuantConfig cfg;
    cfg.mode = lumina::QuantConfig::Mode::kBinary;
    auto q = lumina::make_quantizer(cfg);
    ASSERT_NE(q, nullptr);
    ASSERT_TRUE(q->train({vecs[0].data()}, 1, dim).ok());
    EXPECT_EQ(q->code_bytes(), dim / 8);

    lumina::QuantCode code;
    q->encode(vecs[0].data(), &code);
    // distance to itself = 0.
    EXPECT_NEAR(q->distance_to_query(code, vecs[0].data(), dim), 0.0F, 1e-6F);

    // Complement vector (all sign flipped) -> full Hamming distance.
    std::vector<float> neg = vecs[0];
    for (auto& x : neg) {
        x = -x;
    }
    // Skip zeros: ensure no zero components for a clean comparison.
    EXPECT_NEAR(q->distance_to_query(code, neg.data(), dim), static_cast<float>(dim), 1e-6F);
}

TEST(QuantizerTest, PqAdcApproximatesL2) {
    constexpr size_t dim = 64;
    const auto vecs = random_vectors(300, dim, 3);
    std::vector<const float*> samples;
    for (const auto& v : vecs) {
        samples.push_back(v.data());
    }

    lumina::QuantConfig cfg;
    cfg.mode = lumina::QuantConfig::Mode::kPQ;
    cfg.pq_subspaces = 8;
    cfg.pq_centroids = 16;
    auto q = lumina::make_quantizer(cfg);
    ASSERT_NE(q, nullptr);
    ASSERT_TRUE(q->train(samples, samples.size(), dim).ok());
    EXPECT_EQ(q->code_bytes(), 8U);

    const auto& query = vecs[0];
    lumina::QuantCode code;
    q->encode(vecs[100].data(), &code);
    const float approx = q->distance_to_query(code, query.data(), dim);
    const float exact = l2(query, vecs[100]);
    EXPECT_NEAR(approx, exact, exact * 0.5F + 2.0F);
}

TEST(QuantizerTest, SaveLoadRoundTrip) {
    constexpr size_t dim = 32;
    const auto vecs = random_vectors(100, dim, 4);
    std::vector<const float*> samples;
    for (const auto& v : vecs) {
        samples.push_back(v.data());
    }

    for (auto mode : {lumina::QuantConfig::Mode::kSQ8, lumina::QuantConfig::Mode::kBinary,
                      lumina::QuantConfig::Mode::kPQ}) {
        lumina::QuantConfig cfg;
        cfg.mode = mode;
        cfg.pq_subspaces = 4;
        cfg.pq_centroids = 16;
        auto q = lumina::make_quantizer(cfg);
        ASSERT_TRUE(q->train(samples, samples.size(), dim).ok());

        std::stringstream ss;
        ASSERT_TRUE(q->save(ss).ok());

        auto q2 = lumina::make_quantizer(cfg);
        ASSERT_TRUE(q2->load(ss).ok());
        EXPECT_EQ(q2->code_bytes(), q->code_bytes());

        lumina::QuantCode c1, c2;
        q->encode(vecs[5].data(), &c1);
        q2->encode(vecs[5].data(), &c2);
        EXPECT_EQ(c1.bytes, c2.bytes);

        const float d1 = q->distance_to_query(c1, vecs[9].data(), dim);
        const float d2 = q2->distance_to_query(c2, vecs[9].data(), dim);
        EXPECT_NEAR(d1, d2, 1e-4F);
    }
}

TEST(QuantizerTest, Sq8CodeBytesAndEncodeRange) {
    constexpr size_t dim = 128;
    const auto vecs = random_vectors(50, dim, 5);
    std::vector<const float*> samples;
    for (const auto& v : vecs) {
        samples.push_back(v.data());
    }
    lumina::QuantConfig cfg;
    cfg.mode = lumina::QuantConfig::Mode::kSQ8;
    auto q = lumina::make_quantizer(cfg);
    ASSERT_TRUE(q->train(samples, samples.size(), dim).ok());
    lumina::QuantCode code;
    q->encode(vecs[0].data(), &code);
    ASSERT_EQ(code.bytes.size(), dim);
    // All bytes in [0,255] by construction (uint8).
    EXPECT_EQ(q->code_bytes(), dim);
}
