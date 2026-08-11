#pragma once

#include <cstddef>

namespace lumina {

// Function pointer type for distance kernels.
// Returns squared L2 distance between two float vectors of length `dim`.
using DistanceFn = float (*)(const float* a, const float* b, size_t dim);

// Kernel registration is internal: use VectorMath::l2_distance / init().

namespace VectorMath {

// Detects available SIMD and selects kernels (idempotent, thread-safe).
void init();

}  // namespace VectorMath
}  // namespace lumina
