#pragma once

#include "lumina/common/types.h"
#include "lumina/storage/index_manager.h"
#include "lumina/storage/log_manager.h"
#include <memory>
#include <shared_mutex>
#include <string>

namespace lumina {

class StorageEngine {
public:
    explicit StorageEngine(Options opts = {});
    ~StorageEngine();

    // Non-copyable
    StorageEngine(const StorageEngine&)            = delete;
    StorageEngine& operator=(const StorageEngine&) = delete;

    // Open engine: creates/opens WAL file and runs recovery.
    Status open();

    // Write a key-value pair. Thread-safe.
    Status put(const Slice& key, const Slice& value);

    // Write a vector entry (same as put but records OpType::kVectorPut).
    Status put_vector(const Slice& key, const Slice& value);

    // Read a value by key. Thread-safe.
    // Returns Status::NotFound if key does not exist.
    Status get(const Slice& key, std::string* out_value) const;

    // Soft-delete a key. Thread-safe.
    Status remove(const Slice& key);

    // Rebuild the in-memory index by replaying the WAL from the beginning.
    // Called automatically by open(); can be called manually for testing.
    Status recover();

    // Flush pending writes to disk.
    Status sync();

    size_t key_count() const;

private:
    Options            opts_;
    LogManager         log_;
    IndexManager       index_;
    mutable std::shared_mutex mutex_;
};

} // namespace lumina
