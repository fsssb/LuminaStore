#include "lumina/storage/snapshot.h"

#include "lumina/common/crc32.h"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace lumina {
namespace {

constexpr uint32_t kSnapVersion = 1U;
constexpr uint32_t kMaxEntries  = 100000000U;  // sanity cap
constexpr uint32_t kMaxKeyLen   = 0xFFFFU;

Status io_error(const std::string& what) {
    return Status::IOError(what + " (" + std::strerror(errno) + ")");
}

void put_u32(std::vector<uint8_t>* b, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        b->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
    }
}

void put_u64(std::vector<uint8_t>* b, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        b->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
    }
}

struct Reader {
    explicit Reader(const std::vector<uint8_t>& buf) : buf_(buf) {}

    bool take(size_t n, uint8_t* out) {
        if (n > buf_.size() - pos_) {
            return false;
        }
        if (out) {
            std::memcpy(out, buf_.data() + pos_, n);
        }
        pos_ += n;
        return true;
    }

    bool take_u32(uint32_t* out) {
        uint8_t b[4];
        if (!take(4, b)) {
            return false;
        }
        *out = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
               (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
        return true;
    }

    bool take_u64(uint64_t* out) {
        uint8_t b[8];
        if (!take(8, b)) {
            return false;
        }
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<uint64_t>(b[i]) << (8 * i);
        }
        *out = v;
        return true;
    }

    bool take_bytes(size_t n, std::string* out) {
        if (n > buf_.size() - pos_) {
            return false;
        }
        if (out) {
            out->assign(reinterpret_cast<const char*>(buf_.data() + pos_), n);
        }
        pos_ += n;
        return true;
    }

    size_t pos() const { return pos_; }

private:
    const std::vector<uint8_t>& buf_;
    size_t pos_ = 0;
};

}  // namespace

Status write_snapshot(const std::string& path, uint64_t wal_offset,
                      const std::unordered_map<std::string, uint64_t>& index) {
    if (index.size() > kMaxEntries) {
        return Status::InvalidArgument("too many snapshot entries");
    }

    std::vector<uint8_t> body;
    body.reserve(24 + index.size() * 16);
    body.push_back('L');
    body.push_back('M');
    body.push_back('S');
    body.push_back('N');
    put_u32(&body, kSnapVersion);
    put_u64(&body, wal_offset);
    put_u32(&body, static_cast<uint32_t>(index.size()));
    for (const auto& [key, offset] : index) {
        if (key.size() > kMaxKeyLen) {
            return Status::InvalidArgument("snapshot key too long");
        }
        put_u32(&body, static_cast<uint32_t>(key.size()));
        body.insert(body.end(), key.begin(), key.end());
        put_u64(&body, offset);
    }
    const uint32_t crc = crc32(body.data(), body.size());
    put_u32(&body, crc);

    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return io_error("open snapshot for write");
    }
    bool ok = true;
    size_t written = 0;
    while (written < body.size()) {
        const ssize_t n = ::write(fd, body.data() + written, body.size() - written);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            ok = false;
            break;
        }
        if (n == 0) {
            ok = false;
            break;
        }
        written += static_cast<size_t>(n);
    }
    if (ok && ::fsync(fd) != 0) {
        ok = false;
    }
    ::close(fd);
    if (!ok) {
        return io_error("write snapshot");
    }
    return Status::OK();
}

Status read_snapshot(const std::string& path, SnapshotMeta* out) {
    if (out == nullptr) {
        return Status::InvalidArgument("out is null");
    }
    out->wal_offset = 0;
    out->index.clear();

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Status::NotFound("snapshot missing: " + path);
    }

    std::vector<uint8_t> buf;
    std::array<uint8_t, 65536> chunk{};
    while (true) {
        const ssize_t n = ::read(fd, chunk.data(), chunk.size());
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            ::close(fd);
            return io_error("read snapshot");
        }
        if (n == 0) {
            break;
        }
        buf.insert(buf.end(), chunk.data(), chunk.data() + n);
    }
    ::close(fd);

    if (buf.size() < 24) {  // header(20) + trailing crc(4)
        return Status::Corruption("snapshot too small");
    }

    Reader r(buf);
    uint8_t magic[4];
    uint32_t version = 0, entry_count = 0;
    uint64_t wal_offset = 0;
    if (!r.take(4, magic)) {
        return Status::Corruption("snapshot magic truncated");
    }
    if (magic[0] != 'L' || magic[1] != 'M' || magic[2] != 'S' || magic[3] != 'N') {
        return Status::Corruption("bad snapshot magic");
    }
    if (!r.take_u32(&version) || !r.take_u64(&wal_offset) || !r.take_u32(&entry_count)) {
        return Status::Corruption("snapshot header truncated");
    }
    if (version != kSnapVersion) {
        return Status::Corruption("unsupported snapshot version");
    }
    if (entry_count > kMaxEntries) {
        return Status::Corruption("snapshot entry count out of range");
    }

    // Verify trailing CRC before trusting any content.
    if (buf.size() < 4) {
        return Status::Corruption("snapshot missing crc");
    }
    const uint32_t crc_on_disk = static_cast<uint32_t>(buf[buf.size() - 4]) |
                                 (static_cast<uint32_t>(buf[buf.size() - 3]) << 8) |
                                 (static_cast<uint32_t>(buf[buf.size() - 2]) << 16) |
                                 (static_cast<uint32_t>(buf[buf.size() - 1]) << 24);
    const uint32_t expect = crc32(buf.data(), buf.size() - 4);
    if (expect != crc_on_disk) {
        return Status::Corruption("snapshot crc mismatch");
    }

    for (uint32_t i = 0; i < entry_count; ++i) {
        uint32_t key_len = 0;
        if (!r.take_u32(&key_len)) {
            return Status::Corruption("snapshot key_len truncated");
        }
        if (key_len > kMaxKeyLen) {
            return Status::Corruption("snapshot key too long");
        }
        std::string key;
        if (!r.take_bytes(key_len, &key)) {
            return Status::Corruption("snapshot key truncated");
        }
        uint64_t offset = 0;
        if (!r.take_u64(&offset)) {
            return Status::Corruption("snapshot offset truncated");
        }
        out->index.emplace(std::move(key), offset);
    }
    out->wal_offset = wal_offset;
    return Status::OK();
}

} // namespace lumina
