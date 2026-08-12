#include "lumina/engine/pipeline.h"

#include "lumina/vector/vector_math.h"

namespace lumina {
namespace pipeline {
namespace {

float ip_neg_distance(const float* a, const float* b, size_t dim) {
    float s = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return -s;  // smaller = larger inner product
}

}  // namespace

DistanceFn distance_for_metric(Metric metric) {
    switch (metric) {
        case Metric::kIP:
            return ip_neg_distance;
        case Metric::kCosine:
            return VectorMath::cosine_distance;
        case Metric::kL2:
        default:
            return VectorMath::l2_distance;
    }
}

} // namespace pipeline
} // namespace lumina
