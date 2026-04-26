#pragma once

#include <cstddef>
#include <cmath>

namespace lumina {
namespace VectorMath {

// ---- Naive (scalar) implementations ----

float l2_distance_naive(const float* a, const float* b, size_t dim);
float cosine_distance_naive(const float* a, const float* b, size_t dim);

// ---- SIMD implementations (compiled conditionally) ----

#if defined(LUMINA_HAS_NEON)
float l2_distance_neon(const float* a, const float* b, size_t dim);
float cosine_distance_neon(const float* a, const float* b, size_t dim);
#endif

#if defined(LUMINA_HAS_AVX2)
float l2_distance_avx2(const float* a, const float* b, size_t dim);
float cosine_distance_avx2(const float* a, const float* b, size_t dim);
#endif

#if defined(LUMINA_HAS_AVX512)
float l2_distance_avx512(const float* a, const float* b, size_t dim);
float cosine_distance_avx512(const float* a, const float* b, size_t dim);
#endif

// ---- Runtime-dispatched entry points ----
// Call init() once before using these.

void  init();
float l2_distance(const float* a, const float* b, size_t dim);
float cosine_distance(const float* a, const float* b, size_t dim);

} // namespace VectorMath
} // namespace lumina
