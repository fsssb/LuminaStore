#pragma once

#include <cstddef>
#include <cmath>

namespace lumina {
namespace VectorMath {

// ---- Naive (scalar) implementations ----
// 与 `l2_distance` / `cosine_distance` 不同，naive 不对指针做检查：
// 当 dim>0 时 a、b 须指向至少 dim 个 float 的有效存储，否则为未定义行为。
// 当 dim==0 时循环不读取任何元素，此时 a、b 允许为 nullptr。

float l2_distance_naive(const float* a, const float* b, size_t dim);
float cosine_distance_naive(const float* a, const float* b, size_t dim);

// ---- SIMD implementations (compiled conditionally) ----

#if LUMINA_BUILD_NEON
float l2_distance_neon(const float* a, const float* b, size_t dim);
float cosine_distance_neon(const float* a, const float* b, size_t dim);
#endif

#if LUMINA_BUILD_AVX2
float l2_distance_avx2(const float* a, const float* b, size_t dim);
float cosine_distance_avx2(const float* a, const float* b, size_t dim);
#endif

#if LUMINA_BUILD_AVX512
float l2_distance_avx512(const float* a, const float* b, size_t dim);
float cosine_distance_avx512(const float* a, const float* b, size_t dim);
#endif

// ---- Runtime-dispatched entry points ----
// init() 可显式调用，也会在首次 l2/cos 距离时懒触发；多线程下安全。
// dim==0：L2 返回 0，余弦距离返回 1；否则若 a 或 b 为 nullptr 则返回 quiet NaN。

void  init();
float l2_distance(const float* a, const float* b, size_t dim);
float cosine_distance(const float* a, const float* b, size_t dim);

} // namespace VectorMath
} // namespace lumina
