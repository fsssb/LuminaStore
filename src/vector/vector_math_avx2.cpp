#include "lumina/vector/vector_math.h"

#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace lumina {
namespace VectorMath {

static inline __m256 load_f32_256(const float* p, size_t i, bool ab32_aligned) {
    if (ab32_aligned) {
        return _mm256_load_ps(p + i);
    }
    return _mm256_loadu_ps(p + i);
}

float l2_distance_avx2(const float* a, const float* b, size_t dim) {
    const bool ab32 =
        ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 31U) == 0U;

    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = load_f32_256(a, i, ab32);
        const __m256 vb = load_f32_256(b, i, ab32);
        const __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    const __m128 hi = _mm256_extractf128_ps(sum, 1);
    const __m128 lo = _mm256_castps256_ps128(sum);
    const __m128 s4 = _mm_add_ps(lo, hi);
    const __m128 s2 = _mm_hadd_ps(s4, s4);
    const __m128 s1 = _mm_hadd_ps(s2, s2);
    float result = _mm_cvtss_f32(s1);

    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

float cosine_distance_avx2(const float* a, const float* b, size_t dim) {
    const bool ab32 =
        ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 31U) == 0U;

    __m256 dot_v = _mm256_setzero_ps();
    __m256 na_v = _mm256_setzero_ps();
    __m256 nb_v = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        const __m256 va = load_f32_256(a, i, ab32);
        const __m256 vb = load_f32_256(b, i, ab32);
        dot_v = _mm256_fmadd_ps(va, vb, dot_v);
        na_v = _mm256_fmadd_ps(va, va, na_v);
        nb_v = _mm256_fmadd_ps(vb, vb, nb_v);
    }

    alignas(32) float dot_arr[8];
    alignas(32) float na_arr[8];
    alignas(32) float nb_arr[8];
    _mm256_store_ps(dot_arr, dot_v);
    _mm256_store_ps(na_arr, na_v);
    _mm256_store_ps(nb_arr, nb_v);

    float dot = 0.0F;
    float na = 0.0F;
    float nb = 0.0F;
    for (int j = 0; j < 8; ++j) {
        dot += dot_arr[j];
        na += na_arr[j];
        nb += nb_arr[j];
    }

    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    const float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-8F) {
        return 1.0F;
    }
    return 1.0F - dot / denom;
}

}  // namespace VectorMath
}  // namespace lumina
