#pragma once

#include "lumina/common/types.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace lumina {

// .snap file layout (all multi-byte fields little-endian):
//
//   [4B magic 'LMSN'][4B version=1]
//   [8B wal_offset]          // watermark: WAL size covered by this snapshot
//   [8B entry_count]
//   per entry: [4B key_len][key bytes][8B offset]
//   [4B CRC32] over everything above (running CRC over all preceding bytes)
//
// Loading is strict: bad magic / version / CRC returns kCorruption so the
// caller can fall back to an earlier snapshot or a full WAL replay.

struct SnapshotMeta {
    uint64_t wal_offset = 0;
    std::unordered_map<std::string, uint64_t> index;
};

// Write the snapshot to `path` (caller is responsible for atomic placement,
// e.g. writing to a .tmp then renaming).
Status write_snapshot(const std::string& path, uint64_t wal_offset,
                      const std::unordered_map<std::string, uint64_t>& index);

// Parse a snapshot file into `out`.
Status read_snapshot(const std::string& path, SnapshotMeta* out);

} // namespace lumina
