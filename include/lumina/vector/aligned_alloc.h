#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <malloc.h>
#endif

namespace lumina {

// Recommended alignment for float SIMD (covers AVX-512, cache-friendly).
inline constexpr size_t kVectorBufferAlignment = 64U;

// Stateless allocator: allocates with posix_memalign or _aligned_malloc.
template <typename T, size_t Alignment = kVectorBufferAlignment>
struct AlignedAllocator {
    static_assert(Alignment % alignof(T) == 0U, "Alignment must be multiple of alignof(T)");
    static_assert((Alignment & (Alignment - 1U)) == 0U, "Alignment must be power of two");
    static_assert(Alignment > 0U, "Alignment must be positive");

    using value_type = T;

    AlignedAllocator() noexcept = default;
    template <typename U>
    explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) {
            return nullptr;
        }
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void* raw = nullptr;
#if defined(_WIN32) || defined(_WIN64)
        raw = _aligned_malloc(n * sizeof(T), Alignment);
        if (raw == nullptr) {
            throw std::bad_alloc();
        }
#else
        if (posix_memalign(&raw, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<T*>(raw);
    }

    void deallocate(T* p, std::size_t) noexcept {
        if (p == nullptr) {
            return;
        }
#if defined(_WIN32) || defined(_WIN64)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <typename T, typename U, size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return true;  // 无状态、同 A
}

template <typename T, typename U, size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept {
    return false;
}

// Primary storage type for f32 vector data in hot paths (HNSW, benchmarks, batch buffers).
using AlignedFloatVector = std::vector<float, AlignedAllocator<float, kVectorBufferAlignment>>;

}  // namespace lumina
