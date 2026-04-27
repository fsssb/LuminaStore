#pragma once

#include "lumina/common/types.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace lumina {

// WAL frame layout (on disk):
//   [1B OpType][4B CRC32][4B PayloadLen][N Bytes Payload]
//
// Payload layout:
//   [2B KeyLen][KeyLen Bytes Key][Value bytes (PayloadLen - 2 - KeyLen)]
//
// CRC32 covers: OpType + PayloadLen (4B) + Payload bytes.

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

    // Open (or create) the WAL file.
    Status open();

    // Append a record; returns offset of the written frame.
    Status append(OpType op, const Slice& key, const Slice& value,
                  uint64_t* out_offset = nullptr);

    // Force pages to disk.
    Status sync();

    // Iterate over all valid frames from the beginning of the file.
    // Callback receives each entry; return false to stop early.
    Status iterate(std::function<bool(const WalEntry&)> cb) const;

    // Read and decode a full WAL frame at byte offset.
    Status read_value_at(uint64_t offset, std::string* out_key, std::string* out_value) const;

    // Current end-of-file offset (next write position).
    uint64_t size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lumina
