#include "lumina/storage/entry_meta.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace lumina {

namespace {

constexpr uint32_t kFlagsVec     = 0x01u;
constexpr uint32_t kFlagsPayload = 0x02u;
constexpr uint32_t kFlagsScalars = 0x04u;
constexpr uint32_t kFlagsKnown   = kFlagsVec | kFlagsPayload | kFlagsScalars;

enum class ScalarType : uint8_t {
    kInt64  = 0,
    kDouble = 1,
    kString = 2,
};

constexpr size_t kMaxVecLen      = 64ULL * 1024ULL;        // dim * 4B * 4 = 1MB vec is plenty
constexpr size_t kMaxPayloadLen  = 256ULL * 1024ULL * 1024ULL;
constexpr size_t kMaxScalarCount = 1ULL << 16;

// ---- little-endian readers/writers on std::string ----

void put_u8(std::string* s, uint8_t v) { s->push_back(static_cast<char>(v)); }

void put_u16(std::string* s, uint16_t v) {
    s->push_back(static_cast<char>(v & 0xFFu));
    s->push_back(static_cast<char>((v >> 8) & 0xFFu));
}

void put_u32(std::string* s, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        s->push_back(static_cast<char>((v >> (8 * i)) & 0xFFu));
    }
}

void put_u64(std::string* s, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        s->push_back(static_cast<char>((v >> (8 * i)) & 0xFFu));
    }
}

// A bounded cursor over the input buffer; every read validates bounds.
struct Reader {
    explicit Reader(const std::string& b) : buf(b) {}

    bool take(uint8_t* out, size_t n) {
        if (n > remaining()) {
            return false;
        }
        if (out != nullptr) {
            std::memcpy(out, buf.data() + pos, n);
        }
        pos += n;
        return true;
    }

    bool take_u8(uint8_t* out) { return take(out, 1); }
    bool take_u16(uint16_t* out) {
        uint8_t b[2];
        if (!take(b, 2)) {
            return false;
        }
        *out = static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
        return true;
    }
    bool take_u32(uint32_t* out) {
        uint8_t b[4];
        if (!take(b, 4)) {
            return false;
        }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<uint32_t>(b[i]) << (8 * i);
        }
        *out = v;
        return true;
    }
    bool take_u64(uint64_t* out) {
        uint8_t b[8];
        if (!take(b, 8)) {
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
        if (n > remaining()) {
            return false;
        }
        if (out != nullptr) {
            out->assign(buf.data() + pos, n);
        }
        pos += n;
        return true;
    }

    size_t remaining() const { return buf.size() - pos; }
    size_t consumed() const { return pos; }

private:
    const std::string& buf;
    size_t pos = 0;
};

}  // namespace

Status encode_entry_meta(const EntryMeta& meta, std::string* out) {
    if (out == nullptr) {
        return Status::InvalidArgument("out is null");
    }
    out->clear();
    if (meta.vec.size() > kMaxVecLen) {
        return Status::InvalidArgument("vec too long");
    }
    if (meta.payload.size() > kMaxPayloadLen) {
        return Status::InvalidArgument("payload too long");
    }
    if (meta.scalars.size() > kMaxScalarCount) {
        return Status::InvalidArgument("too many scalars");
    }

    uint32_t flags = 0;
    if (!meta.vec.empty()) {
        flags |= kFlagsVec;
    }
    if (!meta.payload.empty()) {
        flags |= kFlagsPayload;
    }
    if (!meta.scalars.empty()) {
        flags |= kFlagsScalars;
    }
    put_u32(out, flags);

    if (flags & kFlagsVec) {
        put_u32(out, static_cast<uint32_t>(meta.vec.size()));
        const char* base = reinterpret_cast<const char*>(meta.vec.data());
        out->append(base, meta.vec.size() * sizeof(float));
    }
    if (flags & kFlagsPayload) {
        put_u32(out, static_cast<uint32_t>(meta.payload.size()));
        out->append(meta.payload);
    }
    if (flags & kFlagsScalars) {
        put_u16(out, static_cast<uint16_t>(meta.scalars.size()));
        // Deterministic order: sort by name.
        std::vector<ScalarField> sorted = meta.scalars;
        std::sort(sorted.begin(), sorted.end(),
                  [](const ScalarField& a, const ScalarField& b) { return a.name < b.name; });
        for (const auto& field : sorted) {
            if (field.name.size() > 0xFFFFU) {
                return Status::InvalidArgument("scalar name too long");
            }
            put_u8(out, static_cast<uint8_t>(field.value.index()));
            put_u16(out, static_cast<uint16_t>(field.name.size()));
            out->append(field.name);
            switch (field.value.index()) {
                case 0: {  // int64
                    const int64_t v = std::get<int64_t>(field.value);
                    put_u64(out, static_cast<uint64_t>(v));
                    break;
                }
                case 1: {  // double
                    const double v = std::get<double>(field.value);
                    uint64_t bits = 0;
                    static_assert(sizeof(bits) == sizeof(v));
                    std::memcpy(&bits, &v, sizeof(v));
                    put_u64(out, bits);
                    break;
                }
                case 2: {  // string
                    const std::string& s = std::get<std::string>(field.value);
                    if (s.size() > std::numeric_limits<uint32_t>::max()) {
                        return Status::InvalidArgument("scalar string too long");
                    }
                    put_u32(out, static_cast<uint32_t>(s.size()));
                    out->append(s);
                    break;
                }
                default:
                    return Status::InvalidArgument("unknown scalar type");
            }
        }
    }
    return Status::OK();
}

Status decode_entry_meta(const std::string& bytes, EntryMeta* out) {
    if (out == nullptr) {
        return Status::InvalidArgument("out is null");
    }
    out->vec.clear();
    out->payload.clear();
    out->scalars.clear();

    Reader r(bytes);
    uint32_t flags = 0;
    if (!r.take_u32(&flags)) {
        return Status::Corruption("meta too short for flags");
    }
    if ((flags & ~kFlagsKnown) != 0u) {
        return Status::Corruption("meta unknown flags: " + std::to_string(flags));
    }

    if (flags & kFlagsVec) {
        uint32_t vec_len = 0;
        if (!r.take_u32(&vec_len)) {
            return Status::Corruption("meta missing vec_len");
        }
        if (vec_len > kMaxVecLen) {
            return Status::Corruption("meta vec_len too large: " + std::to_string(vec_len));
        }
        out->vec.resize(vec_len);
        const size_t nbytes = static_cast<size_t>(vec_len) * sizeof(float);
        if (r.remaining() < nbytes) {
            return Status::Corruption("meta vec bytes truncated");
        }
        std::memcpy(out->vec.data(), bytes.data() + r.consumed(), nbytes);
        if (!r.take(nullptr, nbytes)) {
            return Status::Corruption("meta vec consume failed");
        }
    }

    if (flags & kFlagsPayload) {
        uint32_t payload_len = 0;
        if (!r.take_u32(&payload_len)) {
            return Status::Corruption("meta missing payload_len");
        }
        if (payload_len > kMaxPayloadLen) {
            return Status::Corruption("meta payload_len too large: " + std::to_string(payload_len));
        }
        if (!r.take_bytes(payload_len, &out->payload)) {
            return Status::Corruption("meta payload truncated");
        }
    }

    if (flags & kFlagsScalars) {
        uint16_t nscalars = 0;
        if (!r.take_u16(&nscalars)) {
            return Status::Corruption("meta missing scalar count");
        }
        out->scalars.reserve(nscalars);
        for (uint16_t i = 0; i < nscalars; ++i) {
            uint8_t type = 0;
            uint16_t key_len = 0;
            if (!r.take_u8(&type)) {
                return Status::Corruption("meta scalar type truncated");
            }
            if (!r.take_u16(&key_len)) {
                return Status::Corruption("meta scalar key_len truncated");
            }
            std::string key;
            if (!r.take_bytes(key_len, &key)) {
                return Status::Corruption("meta scalar key truncated");
            }
            ScalarField field;
            field.name = std::move(key);
            switch (type) {
                case static_cast<uint8_t>(ScalarType::kInt64): {
                    uint64_t raw = 0;
                    if (!r.take_u64(&raw)) {
                        return Status::Corruption("meta scalar int64 truncated");
                    }
                    field.value = static_cast<int64_t>(raw);
                    break;
                }
                case static_cast<uint8_t>(ScalarType::kDouble): {
                    uint64_t raw = 0;
                    if (!r.take_u64(&raw)) {
                        return Status::Corruption("meta scalar double truncated");
                    }
                    double v = 0;
                    std::memcpy(&v, &raw, sizeof(v));
                    field.value = v;
                    break;
                }
                case static_cast<uint8_t>(ScalarType::kString): {
                    uint32_t slen = 0;
                    if (!r.take_u32(&slen)) {
                        return Status::Corruption("meta scalar string len truncated");
                    }
                    std::string s;
                    if (!r.take_bytes(slen, &s)) {
                        return Status::Corruption("meta scalar string truncated");
                    }
                    field.value = std::move(s);
                    break;
                }
                default:
                    return Status::Corruption("meta unknown scalar type: " + std::to_string(type));
            }
            out->scalars.push_back(std::move(field));
        }
    }

    if (r.consumed() != bytes.size()) {
        return Status::Corruption("meta trailing bytes");
    }
    return Status::OK();
}

} // namespace lumina
