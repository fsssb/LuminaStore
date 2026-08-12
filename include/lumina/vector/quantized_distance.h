#pragma once

#include <cstddef>
#include <cstdint>

namespace lumina {
namespace quant {

// SIMD-accelerated quantized distance kernels.
//
// SQ8 codes are uint8 values mapped linearly from [min,max] to [0,255]; the
// exact dequantized L2 equals scale^2 * sum((a-b)^2) with scale=(max-min)/255,
// so the integer path below is *exact* for code-to-code distances.
//
// Kernels dispatch at runtime to NEON (arm64) / AVX2 (x86) with a scalar
// fallback; results are bit-identical or within 1e-3 for mixed paths.

// Exact squared-L2 between two uint8 codes, scaled by scale^2.
float sq8_l2_bytes(const uint8_t* a, const uint8_t* b, size_t n, float scale);

// Approx squared-L2 between a uint8 code and a raw float query: the query is
// quantized to uint8 first, then the integer path runs. Same scale factor.
float sq8_l2_query(const uint8_t* code, const float* query, size_t dim, float min_val,
                   float max_val);

// Hamming distance between two byte arrays (count of differing bits).
float binary_hamming(const uint8_t* a, const uint8_t* b, size_t bytes);

// Hamming distance between a code and the sign bits of a raw float query.
float binary_hamming_query(const uint8_t* code, const float* query, size_t dim);

} // namespace quant
} // namespace lumina
