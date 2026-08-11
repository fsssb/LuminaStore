#include "lumina/vector/vector_math.h"

#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace lumina {
namespace VectorMath {

static inline __m512 load_f32_512(const float* p, size_t i, bool ab64_aligned) {
    if (ab64_aligned) {
        return _mm512_load_ps(p + i);
    }
    return _mm512_loadu_ps(p + i);
}

float l2_distance_avx512(const float* a, const float* b, size_t dim) {
    const bool ab64 =
        ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 63U) == 0U;

    __m512 sum = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        const __m512 va = load_f32_512(a, i, ab64);
        const __m512 vb = load_f32_512(b, i, ab64);
        const __m512 diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    alignas(64) float arr[16];
    _mm512_store_ps(arr, sum);
    float result = 0.0F;
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
    const bool ab64 =
        ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 63U) == 0U;

    __m512 dot_v = _mm512_setzero_ps();
    __m512 na_v = _mm512_setzero_ps();
    __m512 nb_v = _mm512_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= dim; i += 16) {
        const __m512 va = load_f32_512(a, i, ab64);
        const __m512 vb = load_f32_512(b, i, ab64);
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

    float dot = 0.0F;
    float na = 0.0F;
    float nb = 0.0F;
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
    if (denom < 1e-8F) {
        return 1.0F;
    }
    return 1.0F - dot / denom;
}

}  // namespace VectorMath
}  // namespace lumina
