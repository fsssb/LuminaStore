#pragma once

#include <cstddef>

namespace lumina {

// Options for a single ANN search.
struct SearchOptions {
    size_t ef_search = 50;   // beam width for HNSW traversal
    // Quantized distance for the coarse search and exact re-scoring of the
    // candidates are wired in P5; these knobs reserve the API surface.
    bool use_quantized_distance = true;
    bool rescore = true;
    size_t rescore_candidates = 0;  // 0 = auto (2 * ef_search)
};

} // namespace lumina
