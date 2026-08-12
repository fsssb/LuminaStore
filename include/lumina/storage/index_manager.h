#pragma once

#include "lumina/common/types.h"
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace lumina {

// In-memory index: maps key -> WAL byte offset of the latest Put record.
// A key absent from the map is either deleted or never inserted.

class IndexManager {
public:
    IndexManager() = default;

    // Update the offset for a key (insert or overwrite).
    void put(const std::string& key, uint64_t offset);

    // Remove a key (soft delete).
    void remove(const std::string& key);

    // Look up the WAL offset for a key.
    // Returns std::nullopt if the key does not exist.
    std::optional<uint64_t> get(const std::string& key) const;

    // Returns true if the key exists.
    bool contains(const std::string& key) const;

    // Number of live keys.
    size_t size() const { return map_.size(); }

    // Erase all entries (used during recovery to rebuild from scratch).
    void clear() { map_.clear(); }

    // Snapshot support: full dump of the live key->offset table.
    const std::unordered_map<std::string, uint64_t>& dump() const { return map_; }

    // Replace the whole table (used when loading a snapshot).
    void load(std::unordered_map<std::string, uint64_t> map) { map_ = std::move(map); }

private:
    std::unordered_map<std::string, uint64_t> map_;
};

} // namespace lumina
