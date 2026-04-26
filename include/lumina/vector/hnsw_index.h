#pragma once

#include "lumina/common/types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lumina {

struct SearchResult {
    uint64_t id;
    float    distance;
};

class HNSWIndex {
public:
    explicit HNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200);
    ~HNSWIndex();

    // Non-copyable
    HNSWIndex(const HNSWIndex&)            = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;

    // Insert a vector with a given id.
    Status add_item(uint64_t id, const float* vec);

    // Find the k nearest neighbours to query.
    std::vector<SearchResult> search_top_k(const float* query, size_t k,
                                           size_t ef_search = 50) const;

    // Number of vectors in the index.
    size_t size() const;

    // Persist the graph structure to a file.
    Status save(const std::string& path) const;

    // Load graph structure from a file.
    Status load(const std::string& path);

    size_t dim() const { return dim_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    size_t dim_;
};

} // namespace lumina
