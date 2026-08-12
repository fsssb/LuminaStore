#pragma once

#include "lumina/common/types.h"

#include <cstdint>
#include <string>

namespace lumina {

// MANIFEST (text file in the snapshot dir) records the latest snapshots:
//
//   snap <seq> <wal_offset> <filename>
//
// The newest record is the last line; at most kManifestKeep records are kept.
// Updates are atomic: write MANIFEST.tmp -> fsync -> rename -> fsync dir.

struct ManifestEntry {
    uint64_t seq        = 0;
    uint64_t wal_offset = 0;  // WAL watermark covered by the snapshot
    std::string filename;
};

// Read the newest valid snapshot record. Returns kNotFound when the manifest
// is missing or empty, kCorruption on an unparseable record.
Status read_manifest_latest(const std::string& manifest_path, ManifestEntry* out);

// Atomically append a snapshot record (keeps only the most recent few).
Status write_manifest_append(const std::string& manifest_path, const ManifestEntry& entry);

} // namespace lumina
