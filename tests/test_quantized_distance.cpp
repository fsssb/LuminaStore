#include <gtest/gtest.h>

#include "lumina/vector/quantized_distance.h"

#include <cmath>
#include <random>
#include <vector>

namespace {

std::vector<uint8_t> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> out(n);
    for (auto& b : out) {
        b = static_cast<uint8_t>(dist(rng));
    }
    return out;
}

// Reference scalar implementation for cross-checking.
float sq8_l2_ref(const uint8_t* a, const uint8_t* b, size_t n, float scale) {
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) {
        const int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        acc += static_cast<uint64_t>(d * d);
    }
    return static_cast<float>(acc) * scale * scale;
}

}  // namespace

TEST(QuantizedDistanceTest, Sq8MatchesScalar) {
    for (size_t n : {1U, 7U, 16U, 31U, 128U, 1000U}) {
        const auto a = random_bytes(n, 1);
        const auto b = random_bytes(n, 2);
        const float scale = 0.5F;
        const float simd = lumina::quant::sq8_l2_bytes(a.data(), b.data(), n, scale);
        const float ref = sq8_l2_ref(a.data(), b.data(), n, scale);
        EXPECT_NEAR(simd, ref, std::max(1e-2F, ref * 1e-4F)) << "n=" << n;
    }
}

TEST(QuantizedDistanceTest, Sq8QueryMatchesEncodeThenDistance) {
    constexpr size_t dim = 64;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> query(dim);
    std::vector<uint8_t> code(dim);
    for (size_t i = 0; i < dim; ++i) {
        query[i] = dist(rng);
        // Quantize like ScalarQuantizer8 with min=-1, max=1.
        code[i] = static_cast<uint8_t>(std::lround((query[i] + 1.0F) * 0.5F * 255.0F));
    }
    // distance(code, query) should be consistent with distance(code, encode(query)).
    const float d1 = lumina::quant::sq8_l2_query(code.data(), query.data(), dim, -1.0F, 1.0F);
    std::vector<uint8_t> qcode(dim);
    for (size_t i = 0; i < dim; ++i) {
        qcode[i] = static_cast<uint8_t>(std::lround((query[i] + 1.0F) * 0.5F * 255.0F));
    }
    const float d2 = lumina::quant::sq8_l2_bytes(code.data(), qcode.data(), dim, 2.0F / 255.0F);
    EXPECT_NEAR(d1, d2, 1e-3F);
}

TEST(QuantizedDistanceTest, HammingMatchesPopcount) {
    for (size_t n : {1U, 2U, 8U, 15U, 64U}) {
        const auto a = random_bytes(n, 5);
        const auto b = random_bytes(n, 6);
        uint64_t ref = 0;
        for (size_t i = 0; i < n; ++i) {
            ref += static_cast<uint64_t>(__builtin_popcount(a[i] ^ b[i]));
        }
        EXPECT_EQ(lumina::quant::binary_hamming(a.data(), b.data(), n), static_cast<float>(ref))
            << "n=" << n;
    }
}

TEST(QuantizedDistanceTest, HammingQuerySignBits) {
    constexpr size_t dim = 32;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> query(dim);
    for (auto& x : query) {
        x = dist(rng);
    }
    // All-positive query: code of all ones (sign bit set) -> distance = 0.
    for (auto& x : query) {
        x = std::abs(x) + 0.01F;  // strictly positive, avoids zero sign ambiguity
    }
    std::vector<uint8_t> code(4, 0xFF);
    EXPECT_NEAR(lumina::quant::binary_hamming_query(code.data(), query.data(), dim), 0.0F, 1e-6F);
    // All-negative query: distance = dim (all bits differ).
    for (auto& x : query) {
        x = -std::abs(x) - 0.01F;  // strictly negative
    }
    EXPECT_NEAR(lumina::quant::binary_hamming_query(code.data(), query.data(), dim),
                static_cast<float>(dim), 1e-6F);
}
