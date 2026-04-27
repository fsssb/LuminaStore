#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

namespace lumina {

// Software CRC32 (IEEE 802.3 polynomial).
// For hot paths we could use hardware intrinsics (_mm_crc32_u8 / __crc32cb),
// but a table-based implementation keeps the code portable.

namespace detail {

constexpr uint32_t kCrcPoly = 0xEDB88320u;

inline constexpr uint32_t make_crc_entry(uint32_t i) noexcept {
    uint32_t c = i;
    for (int j = 0; j < 8; ++j)
        c = (c >> 1) ^ (kCrcPoly & ~((c & 1u) - 1u));
    return c;
}

} // namespace detail

// Compute CRC32 over a byte buffer.
// Initial value should be 0xFFFFFFFF; final value is XOR'd with 0xFFFFFFFF.
inline uint32_t crc32(const void* data, size_t len, uint32_t init = 0xFFFFFFFFu) noexcept {
    // Build lookup table at runtime (generated once via static local).
    static const auto table = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i)
            t[i] = detail::make_crc_entry(i);
        return t;
    }();

    const auto* p = static_cast<const uint8_t*>(data);
    uint32_t crc = init;
    while (len--) {
        crc = (crc >> 8) ^ table[(crc ^ *p++) & 0xFFu];
    }
    return crc ^ 0xFFFFFFFFu;
}

// Convenience: compute CRC32 of two concatenated buffers without copying.
inline uint32_t crc32_combine(const void* a, size_t a_len,
                               const void* b, size_t b_len) noexcept {
    const uint32_t mid = crc32(a, a_len) ^ 0xFFFFFFFFu; // keep running state
    return crc32(b, b_len, mid);
}

} // namespace lumina
