#include "lumina/storage/log_manager.h"

#include "lumina/common/crc32.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <array>
#include <vector>

namespace lumina {
namespace detail {

constexpr size_t kFrameHeaderSize = 1U + 4U + 4U;
constexpr size_t kFileHeaderSize  = 8U;

constexpr uint32_t kMaxPayloadBytes = 256U * 1024U * 1024U;

const std::array<uint8_t, 4> kWalMagic = {{'L', 'M', 'S', 'T'}};
constexpr uint16_t kWalFormatVersion = 1U;

Status io_error(std::string what) {
    return Status::IOError(what + " (" + std::strerror(errno) + ")");
}

bool is_valid_op(uint8_t op) {
    return op == static_cast<uint8_t>(OpType::kPut) || op == static_cast<uint8_t>(OpType::kDelete) ||
           op == static_cast<uint8_t>(OpType::kVectorPut) ||
           op == static_cast<uint8_t>(OpType::kVectorPutV2);
}

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

uint32_t read_u32_be_d(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}
uint32_t read_u32_le_d(const uint8_t* p) {
    uint32_t v = 0;
    std::memcpy(&v, p, 4);
    return v;
}
uint16_t read_u16_be_d(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]));
}
uint16_t read_u16_le_d(const uint8_t* p) {
    uint16_t v = 0;
    std::memcpy(&v, p, 2);
    return v;
}

uint32_t crc_frame_v0(uint8_t op_raw, uint32_t payload_len_le, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> crc_buf;
    crc_buf.reserve(1 + 4 + payload.size());
    crc_buf.push_back(op_raw);
    const auto* pl = reinterpret_cast<const uint8_t*>(&payload_len_le);
    crc_buf.insert(crc_buf.end(), pl, pl + 4);
    crc_buf.insert(crc_buf.end(), payload.begin(), payload.end());
    return crc32(crc_buf.data(), crc_buf.size());
}

uint32_t crc_frame_v1(uint8_t op_raw, uint32_t payload_len, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> crc_buf;
    crc_buf.reserve(1 + 4 + payload.size());
    crc_buf.push_back(op_raw);
    std::array<uint8_t, 4> be{};
    be[0] = static_cast<uint8_t>((payload_len >> 24) & 0xFFu);
    be[1] = static_cast<uint8_t>((payload_len >> 16) & 0xFFu);
    be[2] = static_cast<uint8_t>((payload_len >> 8) & 0xFFu);
    be[3] = static_cast<uint8_t>(payload_len & 0xFFu);
    crc_buf.insert(crc_buf.end(), be.begin(), be.end());
    crc_buf.insert(crc_buf.end(), payload.begin(), payload.end());
    return crc32(crc_buf.data(), crc_buf.size());
}

bool magic_is_v2(const uint8_t* p) {
    return p[0] == kWalMagic[0] && p[1] == kWalMagic[1] && p[2] == kWalMagic[2] && p[3] == kWalMagic[3];
}

bool decode_key_value_v1(const std::vector<uint8_t>& p, WalEntry* e) {
    if (p.size() < 2) {
        return false;
    }
    const uint16_t klen = read_u16_be_d(p.data());
    if (2ULL + klen > p.size()) {
        return false;
    }
    e->key.assign(reinterpret_cast<const char*>(p.data() + 2), klen);
    const size_t vlen = p.size() - 2U - klen;
    e->value.assign(reinterpret_cast<const char*>(p.data() + 2 + klen), vlen);
    return true;
}

bool decode_key_value_v0(const std::vector<uint8_t>& p, WalEntry* e) {
    if (p.size() < 2) {
        return false;
    }
    const uint16_t klen = read_u16_le_d(p.data());
    if (2ULL + klen > p.size()) {
        return false;
    }
    e->key.assign(reinterpret_cast<const char*>(p.data() + 2), klen);
    const size_t vlen = p.size() - 2U - klen;
    e->value.assign(reinterpret_cast<const char*>(p.data() + 2 + klen), vlen);
    return true;
}

Status build_payload_v0(const Slice& key, const Slice& value, std::vector<uint8_t>* out) {
    const uint32_t n = static_cast<uint32_t>(2U + key.size() + value.size());
    if (2ULL + key.size() + value.size() < key.size() || n > kMaxPayloadBytes) {
        return Status::InvalidArgument("WAL payload too large");
    }
    if (key.size() > 0xFFFFU) {
        return Status::InvalidArgument("WAL key too long");
    }
    out->resize(n);
    const uint16_t kl = static_cast<uint16_t>(key.size());
    std::memcpy(out->data(), &kl, sizeof(kl));
    if (!key.empty()) {
        std::memcpy(out->data() + 2, key.data(), key.size());
    }
    if (!value.empty()) {
        std::memcpy(out->data() + 2 + key.size(), value.data(), value.size());
    }
    return Status::OK();
}

Status build_payload_v1(const Slice& key, const Slice& value, std::vector<uint8_t>* out) {
    if (2ULL + key.size() + value.size() < key.size()) {
        return Status::InvalidArgument("WAL key/value overflow");
    }
    if (key.size() > 0xFFFFU) {
        return Status::InvalidArgument("WAL key too long");
    }
    const uint64_t total = 2ULL + key.size() + value.size();
    if (total > kMaxPayloadBytes) {
        return Status::InvalidArgument("WAL payload too large");
    }
    out->resize(static_cast<size_t>(total));
    out->at(0) = static_cast<uint8_t>(((key.size() >> 8) & 0xFFU));
    out->at(1) = static_cast<uint8_t>((key.size() & 0xFFU));
    if (!key.empty()) {
        std::memcpy(out->data() + 2, key.data(), key.size());
    }
    if (!value.empty()) {
        std::memcpy(out->data() + 2 + key.size(), value.data(), value.size());
    }
    return Status::OK();
}

}  // namespace detail

// Make WAL helpers visible to struct LogManager::Impl (Pimpl) without repeating detail:: in every line.
using namespace detail;

struct LogManager::Impl {
    explicit Impl(std::string p) : path(std::move(p)) {}

    std::string  path;
    int          fd         = -1;
    uint64_t     eof_offset = 0;
    bool         legacy_wal = false;
    uint64_t     data_start = 0;

    void refresh_eof() {
        if (fd < 0) {
            return;
        }
        const off_t end = ::lseek(fd, 0, SEEK_END);
        if (end >= 0) {
            eof_offset = static_cast<uint64_t>(end);
        }
    }

    Status write_file_header() {
        std::array<uint8_t, kFileHeaderSize> hdr{};
        std::memcpy(hdr.data(), kWalMagic.data(), 4);
        hdr[4] = static_cast<uint8_t>((kWalFormatVersion >> 8) & 0xFFu);
        hdr[5] = static_cast<uint8_t>(kWalFormatVersion & 0xFFu);
        hdr[6] = 0;
        hdr[7] = 0;
        if (::lseek(fd, 0, SEEK_SET) < 0) {
            return io_error("lseek(0) for WAL header");
        }
        if (!write_all(fd, hdr.data(), hdr.size())) {
            return io_error("write WAL file header");
        }
        if (::lseek(fd, 0, SEEK_END) < 0) {
            return io_error("lseek end after header");
        }
        data_start = kFileHeaderSize;
        eof_offset = kFileHeaderSize;
        return Status::OK();
    }

    Status open_and_repair() {
        const off_t end = ::lseek(fd, 0, SEEK_END);
        if (end < 0) {
            return io_error("lseek WAL end");
        }
        eof_offset = static_cast<uint64_t>(end);

        if (eof_offset == 0) {
            legacy_wal  = false;
            data_start  = 0;
            return Status::OK();
        }

        if (eof_offset < kFileHeaderSize) {
            legacy_wal  = true;
            data_start  = 0;
        } else {
            std::array<uint8_t, 8> hdr{};
            if (!pread_all(fd, 0, hdr.data(), hdr.size())) {
                return io_error("read WAL header / prefix");
            }
            if (magic_is_v2(hdr.data())) {
                const uint16_t ver = read_u16_be_d(hdr.data() + 4);
                if (ver != kWalFormatVersion) {
                    return Status::Corruption("unsupported WAL version: " + std::to_string(ver));
                }
                legacy_wal  = false;
                data_start  = kFileHeaderSize;
            } else {
                legacy_wal  = true;
                data_start  = 0;
            }
        }

        return repair_truncate_tail();
    }

    Status repair_truncate_tail() {
        uint64_t last_good = data_start;
        uint64_t cursor    = data_start;
        const bool v2      = !legacy_wal;

        while (cursor < eof_offset) {
            if (eof_offset - cursor < kFrameHeaderSize) {
                if (::ftruncate(fd, static_cast<off_t>(last_good)) != 0) {
                    return io_error("ftruncate tail partial frame header");
                }
                refresh_eof();
                return Status::OK();
            }

            std::array<uint8_t, kFrameHeaderSize> fhdr{};
            if (!pread_all(fd, cursor, fhdr.data(), fhdr.size())) {
                if (::ftruncate(fd, static_cast<off_t>(last_good)) != 0) {
                    return io_error("ftruncate after pread frame header (tail)");
                }
                refresh_eof();
                return Status::OK();
            }

            const uint8_t  op_raw      = fhdr[0];
            const uint32_t crc_on_disk = v2 ? read_u32_be_d(fhdr.data() + 1) : read_u32_le_d(fhdr.data() + 1);
            const uint32_t payload_len = v2 ? read_u32_be_d(fhdr.data() + 5) : read_u32_le_d(fhdr.data() + 5);
            const uint32_t pl_le_in_hdr = read_u32_le_d(fhdr.data() + 5);

            if (!is_valid_op(op_raw)) {
                if (v2) {
                    return Status::Corruption("invalid OpType: " + std::to_string(static_cast<int>(op_raw)));
                }
                if (::ftruncate(fd, static_cast<off_t>(last_good)) != 0) {
                    return io_error("ftruncate after invalid op (legacy tail)");
                }
                refresh_eof();
                return Status::OK();
            }

            if (payload_len < 2) {
                if (v2) {
                    return Status::Corruption("payload too small: " + std::to_string(payload_len));
                }
                if (::ftruncate(fd, static_cast<off_t>(last_good)) != 0) {
                    return io_error("ftruncate after bad len (legacy)");
                }
                refresh_eof();
                return Status::OK();
            }

            if (payload_len > kMaxPayloadBytes) {
                return Status::Corruption("payload over cap: " + std::to_string(payload_len));
            }
            {
                const uint64_t nxt = static_cast<uint64_t>(cursor) + kFrameHeaderSize +
                                       static_cast<uint64_t>(payload_len);
                if (nxt < cursor) {
                    return Status::Corruption("payload length overflow");
                }
            }
            if (cursor + kFrameHeaderSize + payload_len > eof_offset) {
                if (::ftruncate(fd, static_cast<off_t>(last_good)) != 0) {
                    return io_error("ftruncate incomplete payload (tail)");
                }
                refresh_eof();
                return Status::OK();
            }

            std::vector<uint8_t> payload(payload_len);
            if (!pread_all(fd, cursor + kFrameHeaderSize, payload.data(), payload.size())) {
                if (::ftruncate(fd, static_cast<off_t>(last_good)) != 0) {
                    return io_error("ftruncate after pread payload (tail)");
                }
                refresh_eof();
                return Status::OK();
            }

            const uint32_t expect_crc =
                v2 ? crc_frame_v1(op_raw, payload_len, payload)
                   : crc_frame_v0(op_raw, pl_le_in_hdr, payload);

            if (expect_crc != crc_on_disk) {
                const bool more_after = (cursor + kFrameHeaderSize + payload_len < eof_offset);
                if (more_after) {
                    return Status::Corruption("WAL CRC mismatch (middle corruption) at offset " +
                                             std::to_string(cursor));
                }
                if (::ftruncate(fd, static_cast<off_t>(last_good)) != 0) {
                    return io_error("ftruncate after bad tail frame CRC");
                }
                refresh_eof();
                return Status::OK();
            }

            last_good = cursor + kFrameHeaderSize + payload_len;
            cursor    = last_good;
        }
        return Status::OK();
    }
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
        return io_error("open WAL: " + impl_->path);
    }
    return impl_->open_and_repair();
}

Status LogManager::append(OpType op, const Slice& key, const Slice& value, uint64_t* out_offset) {
    if (impl_->fd < 0) {
        return Status::IOError("WAL not opened");
    }
    if (key.size() > 0xFFFFU) {
        return Status::InvalidArgument("WAL key too long");
    }

    if (!impl_->legacy_wal && impl_->eof_offset == 0) {
        const Status h = impl_->write_file_header();
        if (!h.ok()) {
            return h;
        }
    }

    std::vector<uint8_t> payload;
    const uint8_t        op_raw = static_cast<uint8_t>(op);
    Status                 ps   = Status::OK();
    uint32_t               plen = 0;
    if (impl_->legacy_wal) {
        ps = build_payload_v0(key, value, &payload);
        if (!ps.ok()) {
            return ps;
        }
        plen = static_cast<uint32_t>(payload.size());
    } else {
        ps = build_payload_v1(key, value, &payload);
        if (!ps.ok()) {
            return ps;
        }
        plen = static_cast<uint32_t>(payload.size());
    }

    if (!is_valid_op(op_raw)) {
        return Status::InvalidArgument("invalid OpType");
    }

    const uint32_t crc =
        impl_->legacy_wal ? crc_frame_v0(op_raw, plen, payload) : crc_frame_v1(op_raw, plen, payload);

    std::array<uint8_t, kFrameHeaderSize> fhdr{};
    fhdr[0] = op_raw;
    if (impl_->legacy_wal) {
        std::memcpy(fhdr.data() + 1, &crc, sizeof(crc));
        std::memcpy(fhdr.data() + 5, &plen, sizeof(plen));
    } else {
        fhdr[1] = static_cast<uint8_t>((crc >> 24) & 0xFFu);
        fhdr[2] = static_cast<uint8_t>((crc >> 16) & 0xFFu);
        fhdr[3] = static_cast<uint8_t>((crc >> 8) & 0xFFu);
        fhdr[4] = static_cast<uint8_t>(crc & 0xFFu);
        fhdr[5] = static_cast<uint8_t>((plen >> 24) & 0xFFu);
        fhdr[6] = static_cast<uint8_t>((plen >> 16) & 0xFFu);
        fhdr[7] = static_cast<uint8_t>((plen >> 8) & 0xFFu);
        fhdr[8] = static_cast<uint8_t>(plen & 0xFFu);
    }

    const uint64_t off = impl_->eof_offset;
    if (!write_all(impl_->fd, fhdr.data(), fhdr.size())) {
        return io_error("write WAL frame header");
    }
    if (!payload.empty() && !write_all(impl_->fd, payload.data(), payload.size())) {
        return io_error("write WAL payload");
    }
    impl_->eof_offset = off + fhdr.size() + payload.size();
    if (out_offset) {
        *out_offset = off;
    }
    return Status::OK();
}

Status LogManager::sync() {
    if (impl_->fd < 0) {
        return Status::IOError("WAL not opened");
    }
    if (::fsync(impl_->fd) != 0) {
        return io_error("fsync WAL");
    }
    return Status::OK();
}

Status LogManager::iterate(std::function<bool(const WalEntry&)> cb) const {
    return iterate_from(impl_->data_start, std::move(cb));
}

Status LogManager::iterate_from(uint64_t from_offset,
                                std::function<bool(const WalEntry&)> cb) const {
    if (impl_->fd < 0) {
        return Status::IOError("WAL not opened");
    }
    if (from_offset < impl_->data_start || from_offset > impl_->eof_offset) {
        // Frame boundaries have no fixed alignment; the watermark comes from a
        // CRC-verified snapshot written by the engine, so range checking suffices.
        return Status::InvalidArgument("bad iterate start offset");
    }

    const bool v2 = !impl_->legacy_wal;
    uint64_t   c  = from_offset;

    while (c + kFrameHeaderSize <= impl_->eof_offset) {
        std::array<uint8_t, kFrameHeaderSize> fhdr{};
        if (!pread_all(impl_->fd, c, fhdr.data(), fhdr.size())) {
            return io_error("pread iterate frame header");
        }

        const uint8_t  op_raw      = fhdr[0];
        const uint32_t crc_on_disk = v2 ? read_u32_be_d(fhdr.data() + 1) : read_u32_le_d(fhdr.data() + 1);
        const uint32_t payload_len = v2 ? read_u32_be_d(fhdr.data() + 5) : read_u32_le_d(fhdr.data() + 5);
        const uint32_t pl_le_in    = read_u32_le_d(fhdr.data() + 5);

        if (!is_valid_op(op_raw) || payload_len < 2 || payload_len > kMaxPayloadBytes) {
            return Status::Corruption("WAL frame invalid while iterating (should have been repaired)");
        }
        if (c + kFrameHeaderSize + payload_len > impl_->eof_offset) {
            return Status::Corruption("WAL iterate sees incomplete file (not repaired?)");
        }

        std::vector<uint8_t> payload(payload_len);
        if (!pread_all(impl_->fd, c + kFrameHeaderSize, payload.data(), payload.size())) {
            return io_error("pread iterate payload");
        }

        const uint32_t expect = v2 ? crc_frame_v1(op_raw, payload_len, payload)
                                    : crc_frame_v0(op_raw, pl_le_in, payload);
        if (expect != crc_on_disk) {
            return Status::Corruption("WAL frame CRC in iterate (not repaired?)");
        }

        WalEntry e;
        e.op_type    = static_cast<OpType>(op_raw);
        e.offset     = c;
        const bool dec = v2 ? decode_key_value_v1(payload, &e) : decode_key_value_v0(payload, &e);
        if (!dec) {
            return Status::Corruption("WAL key/value parse failed");
        }

        if (!cb(e)) {
            return Status::OK();
        }
        c += kFrameHeaderSize + payload_len;
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
    if (offset < impl_->data_start || offset + kFrameHeaderSize > impl_->eof_offset) {
        return Status::NotFound("WAL offset out of range");
    }
    const bool v2 = !impl_->legacy_wal;

    std::array<uint8_t, kFrameHeaderSize> fhdr{};
    if (!pread_all(impl_->fd, offset, fhdr.data(), fhdr.size())) {
        return io_error("pread read_value_at frame header");
    }

    const uint8_t  op_raw    = fhdr[0];
    const uint32_t plen      = v2 ? read_u32_be_d(fhdr.data() + 5) : read_u32_le_d(fhdr.data() + 5);
    const uint32_t pl_le     = read_u32_le_d(fhdr.data() + 5);
    const uint32_t crc_disk  = v2 ? read_u32_be_d(fhdr.data() + 1) : read_u32_le_d(fhdr.data() + 1);
    if (!is_valid_op(op_raw)) {
        return Status::Corruption("invalid OpType");
    }
    if (plen < 2 || plen > kMaxPayloadBytes || offset + kFrameHeaderSize + plen > impl_->eof_offset) {
        return Status::Corruption("invalid frame length in read_value_at");
    }

    std::vector<uint8_t> payload(plen);
    if (!pread_all(impl_->fd, offset + kFrameHeaderSize, payload.data(), plen)) {
        return io_error("pread read_value_at payload");
    }

    const uint32_t expect = v2 ? crc_frame_v1(op_raw, plen, payload) : crc_frame_v0(op_raw, pl_le, payload);
    if (expect != crc_disk) {
        return Status::Corruption("read_value_at CRC mismatch");
    }
    WalEntry tmp;
    if (v2) {
        if (!decode_key_value_v1(payload, &tmp)) {
            return Status::Corruption("key/value v1");
        }
    } else {
        if (!decode_key_value_v0(payload, &tmp)) {
            return Status::Corruption("key/value v0");
        }
    }
    *out_key   = std::move(tmp.key);
    *out_value = std::move(tmp.value);
    return Status::OK();
}

uint64_t LogManager::size() const {
    return impl_->eof_offset;
}

}  // namespace lumina
