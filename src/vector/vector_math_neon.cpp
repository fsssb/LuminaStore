#include "lumina/vector/vector_math.h"

#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <cstdint>

namespace lumina {
namespace VectorMath {

float l2_distance_neon(const float* a, const float* b, size_t dim) {
    const bool pair16 =
        ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 15U) == 0U;

    float32x4_t sum_vec = vdupq_n_f32(0.0F);
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
#if defined(__GNUC__) || defined(__clang__)
        const float32x4_t va = pair16 ? vld1q_f32(static_cast<const float*>(__builtin_assume_aligned(a + i, 16)))
                                      : vld1q_f32(a + i);
        const float32x4_t vb = pair16 ? vld1q_f32(static_cast<const float*>(__builtin_assume_aligned(b + i, 16)))
                                      : vld1q_f32(b + i);
#else
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
#endif
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
    const bool pair16 =
        ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 15U) == 0U;

    float32x4_t dot_v = vdupq_n_f32(0.0F);
    float32x4_t na_v = vdupq_n_f32(0.0F);
    float32x4_t nb_v = vdupq_n_f32(0.0F);
    size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
#if defined(__GNUC__) || defined(__clang__)
        const float32x4_t va = pair16 ? vld1q_f32(static_cast<const float*>(__builtin_assume_aligned(a + i, 16)))
                                      : vld1q_f32(a + i);
        const float32x4_t vb = pair16 ? vld1q_f32(static_cast<const float*>(__builtin_assume_aligned(b + i, 16)))
                                      : vld1q_f32(b + i);
#else
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
#endif
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
    if (denom < 1e-8F) {
        return 1.0F;
    }
    return 1.0F - dot / denom;
}

}  // namespace VectorMath
}  // namespace lumina
