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

    // Write a v2 vector entry whose value is an encoded EntryMeta
    // (vec + payload + scalar filter fields). Records OpType::kVectorPutV2.
    Status put_vector_v2(const Slice& key, const Slice& value);

    // Read a value by key. Thread-safe.
    // Returns Status::NotFound if key does not exist.
    Status get(const Slice& key, std::string* out_value) const;

    // Soft-delete a key. Thread-safe.
    Status remove(const Slice& key);

    // Rebuild the in-memory index by replaying the WAL from the beginning.
    // Called automatically by open(); can be called manually for testing.
    Status recover();

    // Visit every live key with its current value (used to rebuild derived
    // structures such as the HNSW graph after recovery). Thread-safe.
    Status visit_live(std::function<bool(const std::string& key, const std::string& value)> cb) const;

    // Current WAL size in bytes (thread-safe).
    size_t wal_size() const;

    // Flush pending writes to disk.
    Status sync();

    // Write an index snapshot (key->offset table + WAL watermark) and append a
    // MANIFEST record. Subsequent open() loads the snapshot and replays only the
    // WAL tail after the watermark instead of scanning the whole log.
    Status snapshot();

    size_t key_count() const;

private:
    Status recover_inner();
    Options            opts_;
    LogManager         log_;
    IndexManager       index_;
    mutable std::shared_mutex mutex_;
    size_t            appends_since_group_sync_ = 0;
    uint64_t          snap_seq_counter_ = 0;
};

} // namespace lumina
