#include "lumina/vector/vector_math.h"

#include "lumina/common/simd_dispatch.h"

#include <algorithm>
#include <cmath>

#if defined(LUMINA_HAS_NEON)
#include <arm_neon.h>
#endif

#if defined(LUMINA_HAS_AVX2) || defined(LUMINA_HAS_AVX512)
#include <immintrin.h>
#endif

namespace lumina {
namespace VectorMath {

float l2_distance_naive(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float cosine_distance_naive(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    const float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-8f) {
        return 1.0f;
    }
    return 1.0f - dot / denom;
}

#if defined(LUMINA_HAS_NEON)
float l2_distance_neon(const float* a, const float* b, size_t dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        const float32x4_t diff = vsubq_f32(va, vb);
        sum_vec = vmlaq_f32(sum_vec, diff, diff);
    }

    float32x2_t sum2 = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    sum2 = vpadd_f32(sum2, sum2);
    float sum = vget_lane_f32(sum2, 0);

    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float cosine_distance_neon(const float* a, const float* b, size_t dim) {
    float32x4_t dot_v = vdupq_n_f32(0.0f);
    float32x4_t na_v = vdupq_n_f32(0.0f);
    float32x4_t nb_v = vdupq_n_f32(0.0f);
    size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        dot_v = vmlaq_f32(dot_v, va, vb);
        na_v = vmlaq_f32(na_v, va, va);
        nb_v = vmlaq_f32(nb_v, vb, vb);
    }

    const auto hsum4 = [](float32x4_t v) {
        float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
        s = vpadd_f32(s, s);
        return vget_lane_f32(s, 0);
    };

    float dot = hsum4(dot_v);
    float na = hsum4(na_v);
    float nb = hsum4(nb_v);

    for (; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }

    const float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-8f) {
        return 1.0f;
    }
    return 1.0f - dot / denom;
}
#endif

#if defined(LUMINA_HAS_AVX2)
float l2_distance_avx2(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
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
    __m256 dot_v = _mm256_setzero_ps();
    __m256 na_v = _mm256_setzero_ps();
    __m256 nb_v = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
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

    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
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
    if (denom < 1e-8f) {
        return 1.0f;
    }
    return 1.0f - dot / denom;
}
#endif

#if defined(LUMINA_HAS_AVX512)
float l2_distance_avx512(const float* a, const float* b, size_t dim) {
    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m512 va = _mm512_loadu_ps(a + i);
        const __m512 vb = _mm512_loadu_ps(b + i);
        const __m512 diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    alignas(64) float arr[16];
    _mm512_store_ps(arr, sum);
    float result = 0.0f;
    for (float v : arr) {
        result += v;
    }

    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

float cosine_distance_avx512(const float* a, const float* b, size_t dim) {
    __m512 dot_v = _mm512_setzero_ps();
    __m512 na_v = _mm512_setzero_ps();
    __m512 nb_v = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= dim; i += 16) {
        const __m512 va = _mm512_loadu_ps(a + i);
        const __m512 vb = _mm512_loadu_ps(b + i);
        dot_v = _mm512_fmadd_ps(va, vb, dot_v);
        na_v = _mm512_fmadd_ps(va, va, na_v);
        nb_v = _mm512_fmadd_ps(vb, vb, nb_v);
    }

    alignas(64) float dot_arr[16];
    alignas(64) float na_arr[16];
    alignas(64) float nb_arr[16];
    _mm512_store_ps(dot_arr, dot_v);
    _mm512_store_ps(na_arr, na_v);
    _mm512_store_ps(nb_arr, nb_v);

    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (int j = 0; j < 16; ++j) {
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
    if (denom < 1e-8f) {
        return 1.0f;
    }
    return 1.0f - dot / denom;
}
#endif

void init() {
#if defined(LUMINA_HAS_AVX512)
    g_l2_distance_fn = l2_distance_avx512;
    g_cosine_distance_fn = cosine_distance_avx512;
#elif defined(LUMINA_HAS_AVX2)
    g_l2_distance_fn = l2_distance_avx2;
    g_cosine_distance_fn = cosine_distance_avx2;
#elif defined(LUMINA_HAS_NEON)
    g_l2_distance_fn = l2_distance_neon;
    g_cosine_distance_fn = cosine_distance_neon;
#else
    g_l2_distance_fn = l2_distance_naive;
    g_cosine_distance_fn = cosine_distance_naive;
#endif
}

float l2_distance(const float* a, const float* b, size_t dim) {
    if (!g_l2_distance_fn) {
        init();
    }
    return g_l2_distance_fn(a, b, dim);
}

float cosine_distance(const float* a, const float* b, size_t dim) {
    if (!g_cosine_distance_fn) {
        init();
    }
    return g_cosine_distance_fn(a, b, dim);
}

}  // namespace VectorMath
}  // namespace lumina
