#include <chrono>
#include <cmath>
#include <gtest/gtest.h>

#include "lumina/vector/aligned_alloc.h"
#include "lumina/vector/vector_math.h"

#include <random>
#include <vector>

TEST(VectorMathTest, NaiveL2) {
    const std::vector<float> a{1.0F, 2.0F, 3.0F};
    const std::vector<float> b{2.0F, 4.0F, 6.0F};
    const float d = lumina::VectorMath::l2_distance_naive(a.data(), b.data(), a.size());
    EXPECT_NEAR(d, 14.0F, 1e-5F);
}

TEST(VectorMathTest, NaiveL2SameVector) {
    const std::vector<float> a{1.0F, -1.0F, 0.5F, 9.0F};
    const float d = lumina::VectorMath::l2_distance_naive(a.data(), a.data(), a.size());
    EXPECT_NEAR(d, 0.0F, 1e-6F);
}

TEST(VectorMathTest, NaiveCosine) {
    const std::vector<float> x{1.0F, 0.0F, 0.0F};
    const std::vector<float> y{1.0F, 0.0F, 0.0F};
    const std::vector<float> z{0.0F, 1.0F, 0.0F};

    EXPECT_NEAR(lumina::VectorMath::cosine_distance_naive(x.data(), y.data(), x.size()), 0.0F, 1e-6F);
    EXPECT_NEAR(lumina::VectorMath::cosine_distance_naive(x.data(), z.data(), x.size()), 1.0F, 1e-6F);
}

TEST(VectorMathTest, DefensiveNullL2) {
    float x = 1.0F;
    EXPECT_TRUE(std::isnan(lumina::VectorMath::l2_distance(nullptr, &x, 1U)));
    EXPECT_TRUE(std::isnan(lumina::VectorMath::l2_distance(&x, nullptr, 1U)));
}

TEST(VectorMathTest, DefensiveNullCosine) {
    float x = 1.0F;
    EXPECT_TRUE(std::isnan(lumina::VectorMath::cosine_distance(nullptr, &x, 1U)));
    EXPECT_TRUE(std::isnan(lumina::VectorMath::cosine_distance(&x, nullptr, 1U)));
}

TEST(VectorMathTest, DefensiveZeroDim) {
    const float* nullp = nullptr;
    EXPECT_EQ(lumina::VectorMath::l2_distance(nullp, nullp, 0U), 0.0F);
    EXPECT_EQ(lumina::VectorMath::cosine_distance(nullp, nullp, 0U), 1.0F);
}

void fill_random(lumina::AlignedFloatVector& a, lumina::AlignedFloatVector& b, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (size_t i = 0; i < a.size(); ++i) {
        a[static_cast<size_t>(i)] = dist(rng);
        b[static_cast<size_t>(i)] = dist(rng);
    }
}

class VectorMathManyDims : public ::testing::TestWithParam<size_t> {};

TEST_P(VectorMathManyDims, SimdMatchesNaiveL2) {
    const size_t dim = GetParam();
    std::mt19937 rng(7U + static_cast<unsigned int>(dim));
    lumina::AlignedFloatVector a(dim);
    lumina::AlignedFloatVector b(dim);
    fill_random(a, b, rng);

    lumina::VectorMath::init();
    const float d0 = lumina::VectorMath::l2_distance_naive(a.data(), b.data(), dim);
    const float d1 = lumina::VectorMath::l2_distance(a.data(), b.data(), dim);
    EXPECT_NEAR(d0, d1, 1e-2F) << "dim=" << dim;
}

TEST_P(VectorMathManyDims, SimdMatchesNaiveCosine) {
    const size_t dim = GetParam();
    std::mt19937 rng(11U + static_cast<unsigned int>(dim));
    lumina::AlignedFloatVector a(dim);
    lumina::AlignedFloatVector b(dim);
    fill_random(a, b, rng);

    lumina::VectorMath::init();
    const float c0 = lumina::VectorMath::cosine_distance_naive(a.data(), b.data(), dim);
    const float c1 = lumina::VectorMath::cosine_distance(a.data(), b.data(), dim);
    EXPECT_NEAR(c0, c1, 1e-2F) << "dim=" << dim;
}

INSTANTIATE_TEST_SUITE_P(Dims, VectorMathManyDims, ::testing::Values(127U, 128U, 129U, 512U, 1024U));

TEST(VectorMathTest, SimdDoesNotBlowUpVsNaive1024) {
    std::mt19937 rng(7);
    lumina::AlignedFloatVector a(1024);
    lumina::AlignedFloatVector b(1024);
    fill_random(a, b, rng);

    volatile float sink = 0.0F;
    const int kIter = 1000;

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIter; ++i) {
        sink += lumina::VectorMath::l2_distance_naive(a.data(), b.data(), a.size());
    }
    const auto t1 = std::chrono::steady_clock::now();

    lumina::VectorMath::init();
    const auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIter; ++i) {
        sink += lumina::VectorMath::l2_distance(a.data(), b.data(), a.size());
    }
    const auto t3 = std::chrono::steady_clock::now();

    const double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double dispatched_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double rel = dispatched_ms / std::max(naive_ms, 1e-9);

    EXPECT_GT(sink, 0.0F);
    // 若运行时退化为与 naive 相同，rel≈1；只排除灾难性回退
    EXPECT_LE(rel, 3.0);
}
