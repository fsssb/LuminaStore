#pragma once

#include "lumina/common/types.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace lumina {

// On-disk format (WAL v2, default for new / empty files):
//   [8B file header]
//   then repeating frames
//
// File header (big-endian, total 8 bytes):
//   [4B magic 0x4C4D5354 = 'LMST']
//   [2B format version = 1]
//   [2B reserved, must be 0]
//
// Per-frame layout (WAL v2, all multi-byte fields big-endian in CRC input):
//   [1B OpType: 0x01 Put, 0x02 Delete, 0x03 VectorPut]
//   [4B CRC32]
//   [4B PayloadLen]
//   Payload = [2B KeyLen BE][key bytes][value bytes]
// CRC32 covers: 1B Op + 4B PayloadLen (as encoded) + full payload.
//
// Legacy (v0) WAL: no file header, frame starts at offset 0, CRC input uses
//   host-endian encoding for 4B length (legacy) — kept for older test files
//   until the file is fully replaced.
//
// Recovery:
//   - Trailing partial write (incomplete frame at EOF) is discarded via ftruncate.
//   - A complete last frame with bad CRC: treated as tail damage, truncates before it.
//   - A bad frame with more valid-looking data after it: middle corruption -> error.

struct WalEntry {
    OpType      op_type;
    std::string key;
    std::string value;   // empty for kDelete
    uint64_t    offset;  // byte offset of this frame in the WAL file
};

class LogManager {
public:
    explicit LogManager(std::string path);
    ~LogManager();

    // Non-copyable, movable
    LogManager(const LogManager&)            = delete;
    LogManager& operator=(const LogManager&) = delete;
    LogManager(LogManager&&)                 = default;

    // Open (or create) the WAL file, validate header, repair tail (truncate if needed).
    Status open();

    // Append a record; returns offset of the written frame (start of 9B frame header).
    Status append(OpType op, const Slice& key, const Slice& value,
                  uint64_t* out_offset = nullptr);

    // Force pages to disk.
    Status sync();

    // Iterate valid frames; stops at first middle corruption if not repaired (open should have
    // repaired tail-only issues).
    Status iterate(std::function<bool(const WalEntry&)> cb) const;

    // Read and decode a full WAL frame at byte offset (must point to frame start, not file hdr).
    Status read_value_at(uint64_t offset, std::string* out_key, std::string* out_value) const;

    uint64_t size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lumina
