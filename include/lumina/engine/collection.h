#pragma once

#include "lumina/common/types.h"
#include "lumina/engine/filter.h"
#include "lumina/engine/query.h"
#include "lumina/vector/hnsw_index.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lumina {

// Collection is the top-level v2 engine: one vector + payload + filter-field
// dataset backed by a WAL (StorageEngine) and an HNSW index.
//
// Write path:  add(id, vec, payload, scalars)
//                -> WAL VectorPutV2 (encoded EntryMeta) -> HNSW add_item
// Read path:   get(id) -> WAL read at index offset
// Query path:  search(query, k, opts) -> HNSW traversal
// Recovery:    open() -> StorageEngine (snapshot + WAL replay) -> rebuild HNSW
//              from live entries.
class Collection {
public:
    // `dir` is a directory holding data.wal and the snapshot subdir.
    Collection(std::string dir, size_t dim, Metric metric = Metric::kL2,
               size_t M = 16, size_t ef_construction = 200);
    ~Collection();

    Collection(const Collection&)            = delete;
    Collection& operator=(const Collection&) = delete;

    Status open();

    // Insert a vector with optional payload and filter fields.
    Status add(uint64_t id, const float* vec, const std::string& payload = {},
               const std::vector<ScalarField>& scalars = {});

    // Tombstone-delete an id (no-op if absent).
    Status remove(uint64_t id);

    // Replace the vector/payload/scalars of an existing id (revives tombstones).
    Status update(uint64_t id, const float* vec, const std::string& payload = {},
                  const std::vector<ScalarField>& scalars = {});

    // Read the payload of an id.
    Status get(uint64_t id, std::string* payload) const;

    // ANN top-k search.
    std::vector<SearchResult> search(const float* query, size_t k,
                                     const SearchOptions& opts = {}) const;

    Status snapshot();

    size_t size() const;
    size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }

    // --- metrics (for benchmarks) ---
    struct Stats {
        size_t live_entries = 0;
        size_t wal_bytes    = 0;
    };
    Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    size_t dim_;
    Metric metric_;
};

} // namespace lumina
