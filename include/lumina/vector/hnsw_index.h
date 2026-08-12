#pragma once

#include "lumina/common/simd_dispatch.h"
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

// HNSW approximate nearest neighbour index (v2).
//
// v2 changes over v1:
//   - heuristic (diverse) neighbour selection instead of distance-cutoff pruning
//   - tombstone delete (mark-deleted, no physical edge removal) + update
//   - injectable distance function (L2 / IP / COSINE / quantized)
//   - fine-grained concurrency: per-id stripe locks + per-node locks + short
//     global lock for structural changes; search is safe against concurrent
//     add/remove/update at node granularity.
class HNSWIndex {
public:
    // dist == nullptr selects the default L2 kernel (VectorMath::l2_distance).
    explicit HNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200,
                       DistanceFn dist = nullptr);
    ~HNSWIndex();

    // Non-copyable, movable
    HNSWIndex(const HNSWIndex&)            = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;
    HNSWIndex(HNSWIndex&&) noexcept;
    HNSWIndex& operator=(HNSWIndex&&) noexcept;

    // Insert a vector with a given id. Fails with kInvalidArgument on duplicate id.
    Status add_item(uint64_t id, const float* vec);

    // Tombstone-delete an id. No-op (kNotFound) if the id does not exist.
    Status remove(uint64_t id);

    // Replace the vector of an existing id (revives a tombstoned id) and
    // re-attach it to the graph.
    Status update_item(uint64_t id, const float* vec);

    // Find the k nearest neighbours to query (excluding deleted entries).
    std::vector<SearchResult> search_top_k(const float* query, size_t k,
                                           size_t ef_search = 50) const;

    // Number of live (non-deleted) entries.
    size_t size() const;

    bool contains(uint64_t id) const;

    // Persist the graph structure to a file (v2 on-disk format).
    Status save(const std::string& path) const;

    // Load graph structure from a v2 file.
    Status load(const std::string& path);

    size_t dim() const { return dim_; }

public:
    // Incomplete type defined in hnsw_index.cpp; only forward-declared here.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
    size_t dim_;
};

} // namespace lumina
