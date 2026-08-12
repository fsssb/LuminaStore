#include "lumina/vector/quantized_distance.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace lumina {
namespace quant {
namespace {

inline uint64_t popcount64(uint64_t x) {
    return static_cast<uint64_t>(__builtin_popcountll(x));
}

// ---- scalar fallbacks ----

[[maybe_unused]] float sq8_l2_bytes_scalar(const uint8_t* a, const uint8_t* b, size_t n,
                                           float scale) {
    uint64_t acc = 0;
    for (size_t i = 0; i < n; ++i) {
        const int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        acc += static_cast<uint64_t>(d * d);
    }
    return static_cast<float>(acc) * scale * scale;
}

}  // namespace

float sq8_l2_bytes(const uint8_t* a, const uint8_t* b, size_t n, float scale) {
#if defined(__ARM_NEON)
    uint32x4_t acc = vdupq_n_u32(0);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const uint8x16_t va = vld1q_u8(a + i);
        const uint8x16_t vb = vld1q_u8(b + i);
        const uint8x16_t d8 = vabdq_u8(va, vb);                 // |a-b| per byte
        const uint16x8_t d16_lo = vmovl_u8(vget_low_u8(d8));   // widen to u16
        const uint16x8_t d16_hi = vmovl_u8(vget_high_u8(d8));
        // d^2 fits in u32 for d <= 255 (65535).
        const uint32x4_t sq_lo = vmull_u16(vget_low_u16(d16_lo), vget_low_u16(d16_lo));
        const uint32x4_t sq_hi = vmull_u16(vget_high_u16(d16_lo), vget_high_u16(d16_lo));
        acc = vaddq_u32(acc, sq_lo);
        acc = vaddq_u32(acc, sq_hi);
        const uint32x4_t sq_lo2 = vmull_u16(vget_low_u16(d16_hi), vget_low_u16(d16_hi));
        const uint32x4_t sq_hi2 = vmull_u16(vget_high_u16(d16_hi), vget_high_u16(d16_hi));
        acc = vaddq_u32(acc, sq_lo2);
        acc = vaddq_u32(acc, sq_hi2);
    }
    uint64_t total = vaddvq_u32(acc);
    for (; i < n; ++i) {
        const int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        total += static_cast<uint64_t>(d * d);
    }
    return static_cast<float>(total) * scale * scale;
#elif defined(__AVX2__)
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        const __m256i abs_d = _mm256_abs_epi8(_mm256_sub_epi8(va, vb));
        const __m256i d_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(abs_d));
        const __m256i d_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(abs_d, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_lo, d_lo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d_hi, d_hi));
    }
    alignas(32) uint32_t tmp[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), acc);
    uint64_t total = 0;
    for (int j = 0; j < 8; ++j) {
        total += tmp[j];
    }
    for (; i < n; ++i) {
        const int32_t d = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
        total += static_cast<uint64_t>(d * d);
    }
    return static_cast<float>(total) * scale * scale;
#else
    return sq8_l2_bytes_scalar(a, b, n, scale);
#endif
}

float sq8_l2_query(const uint8_t* code, const float* query, size_t dim, float min_val,
                   float max_val) {
    if (max_val <= min_val) {
        max_val = min_val + 1.0F;
    }
    // Quantize the query once, then reuse the byte kernel. Stack buffer for
    // typical embedding dims avoids per-call heap allocation.
    const float scale = 255.0F / (max_val - min_val);
    std::vector<uint8_t> heap;
    uint8_t* q = nullptr;
    if (dim <= 2048) {
        thread_local std::vector<uint8_t> scratch;
        if (scratch.size() < dim) {
            scratch.resize(dim);
        }
        q = scratch.data();
    } else {
        heap.resize(dim);
        q = heap.data();
    }
    for (size_t i = 0; i < dim; ++i) {
        float v = (query[i] - min_val) * scale;
        v = std::max(0.0F, std::min(255.0F, v));
        q[i] = static_cast<uint8_t>(std::lround(v));
    }
    return sq8_l2_bytes(code, q, dim, 1.0F / scale);
}

float binary_hamming(const uint8_t* a, const uint8_t* b, size_t bytes) {
    uint64_t total = 0;
    for (size_t i = 0; i < bytes; ++i) {
        total += popcount64(a[i] ^ b[i]);
    }
    return static_cast<float>(total);
}

float binary_hamming_query(const uint8_t* code, const float* query, size_t dim) {
    const size_t bytes = (dim + 7) / 8;
    thread_local std::vector<uint8_t> scratch;
    if (scratch.size() < bytes) {
        scratch.resize(bytes);
    }
    // Extract sign bits via the float bit pattern (bit 31); bit j of byte b
    // corresponds to element b*8+j.
    const uint32_t* qi = reinterpret_cast<const uint32_t*>(query);
    for (size_t b = 0; b < bytes; ++b) {
        uint8_t bits = 0;
        for (size_t j = 0; j < 8; ++j) {
            const size_t idx = b * 8 + j;
            if (idx < dim && (qi[idx] >> 31) == 0) {
                bits |= static_cast<uint8_t>(1u << j);
            }
        }
        scratch[b] = bits;
    }
    return binary_hamming(code, scratch.data(), bytes);
}

} // namespace quant
} // namespace lumina
