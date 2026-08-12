#pragma once

#include "lumina/common/simd_dispatch.h"
#include "lumina/common/types.h"

namespace lumina {
namespace pipeline {

// Select the distance kernel for a metric. Cosine returns (1 - cosine
// similarity); IP returns negative inner product (smaller = better).
DistanceFn distance_for_metric(Metric metric);

} // namespace pipeline
} // namespace lumina
