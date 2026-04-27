#include <gtest/gtest.h>

#include "lumina/vector/vector_math.h"

#include <chrono>
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

TEST(VectorMathTest, SimdMatchesNaive) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    std::vector<float> a(1024);
    std::vector<float> b(1024);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    lumina::VectorMath::init();
    const float d0 = lumina::VectorMath::l2_distance_naive(a.data(), b.data(), a.size());
    const float d1 = lumina::VectorMath::l2_distance(a.data(), b.data(), a.size());

    EXPECT_NEAR(d0, d1, 1e-3F);
}

TEST(VectorMathTest, PerformanceRatio) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);

    std::vector<float> a(1024);
    std::vector<float> b(1024);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    constexpr int kIter = 100000;

    volatile float sink = 0.0F;
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
    const double simd_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double ratio = naive_ms / std::max(simd_ms, 1e-9);

    EXPECT_GT(sink, 0.0F);
    EXPECT_GT(ratio, 1.0);
}
