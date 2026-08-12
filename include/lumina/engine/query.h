#pragma once

#include <cstddef>
#include <cstdint>

namespace lumina {

// How a filter is applied during ANN search.
enum class FilterMode : uint8_t {
    kNone      = 0,  // no filtering
    kInFilter  = 1,  // predicate checked when candidates enter the result set
    kPostFilter = 2, // plain search with oversampled ef, then filter + retry
};

// Options for a single ANN search.
struct SearchOptions {
    size_t ef_search = 50;   // beam width for HNSW traversal
    // Quantized distance for the coarse search and exact re-scoring of the
    // candidates are wired in P5/P8; these knobs reserve the API surface.
    bool use_quantized_distance = true;
    bool rescore = true;
    size_t rescore_candidates = 0;  // 0 = auto (2 * ef_search)

    FilterMode filter_mode = FilterMode::kNone;
};

} // namespace lumina
