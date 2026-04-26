#pragma once

#include <cstddef>
#include <functional>

namespace lumina {

// Function pointer type for distance kernels.
// Returns squared L2 distance between two float vectors of length `dim`.
using DistanceFn = float (*)(const float* a, const float* b, size_t dim);

// Filled in by VectorMath::init() during startup.
// Access through VectorMath::l2_distance() rather than directly.
extern DistanceFn g_l2_distance_fn;
extern DistanceFn g_cosine_distance_fn;

namespace VectorMath {

// Must be called once at startup (or the first time VectorMath is used).
// Detects available SIMD and sets the global function pointers.
void init();

// Dispatched L2 squared distance.
inline float l2_distance(const float* a, const float* b, size_t dim) {
    return g_l2_distance_fn(a, b, dim);
}

// Dispatched cosine distance (1 - cosine_similarity).
inline float cosine_distance(const float* a, const float* b, size_t dim) {
    return g_cosine_distance_fn(a, b, dim);
}

} // namespace VectorMath
} // namespace lumina
