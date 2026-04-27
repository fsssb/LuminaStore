#include "lumina/storage/log_manager.h"

#include "lumina/common/crc32.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <array>
#include <utility>
#include <vector>

namespace lumina {
namespace {

constexpr size_t kHeaderSize = 1 + 4 + 4;

bool write_all(int fd, const void* buf, size_t len) {
    const auto* p = static_cast<const uint8_t*>(buf);
    size_t written = 0;
    while (written < len) {
        const ssize_t n = ::write(fd, p + written, len - written);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        written += static_cast<size_t>(n);
    }
    return true;
}

bool pread_all(int fd, uint64_t offset, void* buf, size_t len) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t read_total = 0;
    while (read_total < len) {
        const ssize_t n = ::pread(fd, p + read_total, len - read_total,
                                  static_cast<off_t>(offset + read_total));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (n == 0) {
            return false;
        }
        read_total += static_cast<size_t>(n);
    }
    return true;
}

uint32_t compute_frame_crc(uint8_t op_raw, uint32_t payload_len, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> crc_buf;
    crc_buf.reserve(1 + 4 + payload.size());
    crc_buf.push_back(op_raw);

    std::array<uint8_t, 4> len_bytes{};
    std::memcpy(len_bytes.data(), &payload_len, sizeof(payload_len));
    crc_buf.insert(crc_buf.end(), len_bytes.begin(), len_bytes.end());
    crc_buf.insert(crc_buf.end(), payload.begin(), payload.end());

    return crc32(crc_buf.data(), crc_buf.size());
}

}  // namespace

struct LogManager::Impl {
    explicit Impl(std::string p) : path(std::move(p)) {}

    std::string path;
    int fd = -1;
    uint64_t eof_offset = 0;
};

LogManager::LogManager(std::string path) : impl_(std::make_unique<Impl>(std::move(path))) {}

LogManager::~LogManager() {
    if (impl_ && impl_->fd >= 0) {
        ::close(impl_->fd);
        impl_->fd = -1;
    }
}

Status LogManager::open() {
    if (impl_->fd >= 0) {
        return Status::OK();
    }

    impl_->fd = ::open(impl_->path.c_str(), O_CREAT | O_RDWR, 0644);
    if (impl_->fd < 0) {
        return Status::IOError("failed to open WAL: " + impl_->path);
    }

    const off_t end = ::lseek(impl_->fd, 0, SEEK_END);
    if (end < 0) {
        ::close(impl_->fd);
        impl_->fd = -1;
        return Status::IOError("failed to seek WAL end");
    }
    impl_->eof_offset = static_cast<uint64_t>(end);
    return Status::OK();
}

Status LogManager::append(OpType op, const Slice& key, const Slice& value, uint64_t* out_offset) {
    if (impl_->fd < 0) {
        return Status::IOError("WAL not opened");
    }
    if (key.size() > UINT16_MAX) {
        return Status::InvalidArgument("key length exceeds uint16");
    }

    const uint16_t key_len = static_cast<uint16_t>(key.size());
    const uint32_t payload_len = static_cast<uint32_t>(2 + key.size() + value.size());

    std::vector<uint8_t> payload(payload_len);
    std::memcpy(payload.data(), &key_len, sizeof(key_len));
    std::memcpy(payload.data() + 2, key.data(), key.size());
    if (!value.empty()) {
        std::memcpy(payload.data() + 2 + key.size(), value.data(), value.size());
    }

    const uint8_t op_raw = static_cast<uint8_t>(op);
    const uint32_t crc = compute_frame_crc(op_raw, payload_len, payload);

    std::array<uint8_t, kHeaderSize> header{};
    header[0] = op_raw;
    std::memcpy(header.data() + 1, &crc, sizeof(crc));
    std::memcpy(header.data() + 5, &payload_len, sizeof(payload_len));

    const uint64_t offset = impl_->eof_offset;
    if (!write_all(impl_->fd, header.data(), header.size())) {
        return Status::IOError("failed to write WAL header");
    }
    if (!payload.empty() && !write_all(impl_->fd, payload.data(), payload.size())) {
        return Status::IOError("failed to write WAL payload");
    }

    impl_->eof_offset += header.size() + payload.size();
    if (out_offset) {
        *out_offset = offset;
    }
    return Status::OK();
}

Status LogManager::sync() {
    if (impl_->fd < 0) {
        return Status::IOError("WAL not opened");
    }
    if (::fsync(impl_->fd) != 0) {
        return Status::IOError("fsync WAL failed");
    }
    return Status::OK();
}

Status LogManager::iterate(std::function<bool(const WalEntry&)> cb) const {
    if (impl_->fd < 0) {
        return Status::IOError("WAL not opened");
    }

    uint64_t offset = 0;
    while (offset + kHeaderSize <= impl_->eof_offset) {
        std::array<uint8_t, kHeaderSize> header{};
        if (!pread_all(impl_->fd, offset, header.data(), header.size())) {
            break;
        }

        const uint8_t op_raw = header[0];
        uint32_t crc = 0;
        uint32_t payload_len = 0;
        std::memcpy(&crc, header.data() + 1, sizeof(crc));
        std::memcpy(&payload_len, header.data() + 5, sizeof(payload_len));

        if (payload_len < 2 || offset + kHeaderSize + payload_len > impl_->eof_offset) {
            break;
        }

        std::vector<uint8_t> payload(payload_len);
        if (!pread_all(impl_->fd, offset + kHeaderSize, payload.data(), payload.size())) {
            break;
        }

        const uint32_t expected_crc = compute_frame_crc(op_raw, payload_len, payload);
        if (expected_crc != crc) {
            offset += kHeaderSize + payload_len;
            continue;
        }

        uint16_t key_len = 0;
        std::memcpy(&key_len, payload.data(), sizeof(key_len));
        if (static_cast<size_t>(2 + key_len) > payload.size()) {
            offset += kHeaderSize + payload_len;
            continue;
        }

        WalEntry entry;
        entry.op_type = static_cast<OpType>(op_raw);
        entry.offset = offset;
        entry.key.assign(reinterpret_cast<const char*>(payload.data() + 2), key_len);

        const size_t value_len = payload.size() - 2 - key_len;
        if (value_len > 0) {
            entry.value.assign(reinterpret_cast<const char*>(payload.data() + 2 + key_len), value_len);
        }

        if (!cb(entry)) {
            return Status::OK();
        }

        offset += kHeaderSize + payload_len;
    }

    return Status::OK();
}

Status LogManager::read_value_at(uint64_t offset, std::string* out_key, std::string* out_value) const {
    if (impl_->fd < 0) {
        return Status::IOError("WAL not opened");
    }
    if (!out_key || !out_value) {
        return Status::InvalidArgument("out_key/out_value is null");
    }
    if (offset + kHeaderSize > impl_->eof_offset) {
        return Status::NotFound("offset out of range");
    }

    std::array<uint8_t, kHeaderSize> header{};
    if (!pread_all(impl_->fd, offset, header.data(), header.size())) {
        return Status::IOError("failed to read frame header");
    }

    const uint8_t op_raw = header[0];
    uint32_t crc = 0;
    uint32_t payload_len = 0;
    std::memcpy(&crc, header.data() + 1, sizeof(crc));
    std::memcpy(&payload_len, header.data() + 5, sizeof(payload_len));

    if (payload_len < 2 || offset + kHeaderSize + payload_len > impl_->eof_offset) {
        return Status::Corruption("invalid payload length");
    }

    std::vector<uint8_t> payload(payload_len);
    if (!pread_all(impl_->fd, offset + kHeaderSize, payload.data(), payload.size())) {
        return Status::IOError("failed to read frame payload");
    }

    const uint32_t expected_crc = compute_frame_crc(op_raw, payload_len, payload);
    if (expected_crc != crc) {
        return Status::Corruption("CRC mismatch");
    }

    uint16_t key_len = 0;
    std::memcpy(&key_len, payload.data(), sizeof(key_len));
    if (static_cast<size_t>(2 + key_len) > payload.size()) {
        return Status::Corruption("invalid key length in frame");
    }

    out_key->assign(reinterpret_cast<const char*>(payload.data() + 2), key_len);
    const size_t value_len = payload.size() - 2 - key_len;
    out_value->assign(reinterpret_cast<const char*>(payload.data() + 2 + key_len), value_len);
    return Status::OK();
}

uint64_t LogManager::size() const {
    return impl_->eof_offset;
}

}  // namespace lumina
