#include "lumina/vector/vector_math.h"

#include "lumina/common/simd_dispatch.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>

#if defined(_WIN32) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)) && !defined(_WIN32)
#include <cpuid.h>
#endif

#if !defined(LUMINA_ALLOW_AVX512_KERNEL)
#define LUMINA_ALLOW_AVX512_KERNEL 1
#endif

namespace lumina {
namespace VectorMath {

float l2_distance_naive(const float* a, const float* b, size_t dim) {
    float sum = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

float cosine_distance_naive(const float* a, const float* b, size_t dim) {
    float dot = 0.0F;
    float na = 0.0F;
    float nb = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    const float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-8F) {
        return 1.0F;
    }
    return 1.0F - dot / denom;
}

#if LUMINA_BUILD_NEON
float l2_distance_neon(const float* a, const float* b, size_t dim);
float cosine_distance_neon(const float* a, const float* b, size_t dim);
#endif
#if LUMINA_BUILD_AVX2
float l2_distance_avx2(const float* a, const float* b, size_t dim);
float cosine_distance_avx2(const float* a, const float* b, size_t dim);
#endif
#if LUMINA_BUILD_AVX512 && LUMINA_ALLOW_AVX512_KERNEL
float l2_distance_avx512(const float* a, const float* b, size_t dim);
float cosine_distance_avx512(const float* a, const float* b, size_t dim);
#endif

namespace {

struct KernelTable {
    DistanceFn l2;
    DistanceFn cos;
};

static const KernelTable k_table_naive{l2_distance_naive, cosine_distance_naive};

#if LUMINA_BUILD_NEON
static const KernelTable k_table_neon{l2_distance_neon, cosine_distance_neon};
#endif
#if LUMINA_BUILD_AVX2
static const KernelTable k_table_avx2{l2_distance_avx2, cosine_distance_avx2};
#endif
#if LUMINA_BUILD_AVX512 && LUMINA_ALLOW_AVX512_KERNEL
static const KernelTable k_table_avx512{l2_distance_avx512, cosine_distance_avx512};
#endif

// 单一原子指针，避免 l2 / cosine 两指针的并发撕裂
static std::atomic<const KernelTable*> g_kernels{&k_table_naive};

std::once_flag g_init_flag;

// ------------------------ x86: CPUID +（OSXSAVE 后）XGETBV -------------------------
#if (defined(_WIN32) && (defined(_M_X64) || defined(_M_IX86))) ||        \
    (!defined(_WIN32) && (defined(__x86_64__) || defined(__i386__)))

struct Cpuid {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::uint32_t c = 0;
    std::uint32_t d = 0;
};

#if defined(_WIN32) && (defined(_M_X64) || defined(_M_IX86))
static Cpuid cpuidex(std::uint32_t leaf, std::uint32_t sub) {
    int r[4] = {0, 0, 0, 0};
    __cpuidex(r, static_cast<int>(leaf), static_cast<int>(sub));
    Cpuid o{};
    o.a = static_cast<std::uint32_t>(r[0]);
    o.b = static_cast<std::uint32_t>(r[1]);
    o.c = static_cast<std::uint32_t>(r[2]);
    o.d = static_cast<std::uint32_t>(r[3]);
    return o;
}
#else
static Cpuid cpuidex(std::uint32_t leaf, std::uint32_t sub) {
    std::uint32_t a = 0;
    std::uint32_t b = 0;
    std::uint32_t c = 0;
    std::uint32_t d = 0;
    Cpuid o{};
    if (!__get_cpuid_count(leaf, sub, &a, &b, &c, &d)) {
        return o;
    }
    o.a = a;
    o.b = b;
    o.c = c;
    o.d = d;
    return o;
}
#endif

// Not all x86_64 Clang 版本对 __xgetbv 提供可链接 intrinsic；用内联汇编/MSVC 内建，避免对 x86intrin 的强依赖
#if defined(_WIN32) && (defined(_M_X64) || defined(_M_IX86))
static std::uint64_t lumina_xgetbv0() { return _xgetbv(0); }
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
static std::uint64_t lumina_xgetbv0() {
    std::uint32_t eax = 0U;
    std::uint32_t edx = 0U;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0) : "memory");
    return (static_cast<std::uint64_t>(edx) << 32U) | static_cast<std::uint64_t>(eax);
#else
    (void)eax;
    (void)edx;
    return 0U;
#endif
}
#endif

static std::uint32_t max_cpuid_leaf() { return cpuidex(0, 0).a; }

static std::uint64_t xcr0() {
    const Cpuid id1 = cpuidex(1, 0);
    if ((id1.c & (1U << 27U)) == 0U) {
        return 0;  // 无 OSXSAVE 则不执行 XGETBV
    }
    return lumina_xgetbv0();
}

static bool xcr_ok_for_ymm() {
    const std::uint64_t x = xcr0();
    return (x & 6U) == 6U;
}

// AVX2 + FMA，且 XCR0 允许 XMM/YMM
static bool cpu_can_use_avx2() {
    const Cpuid id1 = cpuidex(1, 0);
    if ((id1.c & (1U << 28U)) == 0U) {
        return false;
    }
    if (!xcr_ok_for_ymm()) {
        return false;
    }
    if ((id1.c & (1U << 12U)) == 0U) {
        return false;
    }
    if (max_cpuid_leaf() < 7U) {
        return false;
    }
    const Cpuid e7 = cpuidex(7, 0);
    if ((e7.b & (1U << 5U)) == 0U) {
        return false;
    }
    return true;
}

#if LUMINA_BUILD_AVX512 && LUMINA_ALLOW_AVX512_KERNEL
static bool xcr_ok_for_avx512_reg_state() {
    const std::uint64_t x = xcr0();
    return (x & 0xE0U) == 0xE0U;
}

// 在 AVX2 失败后再尝试；须 512F 且 XCR0 中 ZMM/OPMASK 等就绪
static bool cpu_can_use_avx512f() {
    if (max_cpuid_leaf() < 7U) {
        return false;
    }
    const Cpuid e7 = cpuidex(7, 0);
    if ((e7.b & (1U << 16U)) == 0U) {
        return false;
    }
    if (!xcr_ok_for_ymm()) {
        return false;
    }
    if (!xcr_ok_for_avx512_reg_state()) {
        return false;
    }
    return true;
}
#endif  // LUMINA_BUILD_AVX512 && LUMINA_ALLOW_AVX512_KERNEL

#endif  // x86

static void select_kernels() {
#if LUMINA_BUILD_NEON
#if defined(__aarch64__) || defined(_M_ARM64)
    g_kernels.store(&k_table_neon, std::memory_order_release);
    return;
#endif
#endif
#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#if LUMINA_BUILD_AVX2
    if (cpu_can_use_avx2()) {
        g_kernels.store(&k_table_avx2, std::memory_order_release);
        return;
    }
#endif
#if LUMINA_BUILD_AVX512 && LUMINA_ALLOW_AVX512_KERNEL
    if (cpu_can_use_avx512f()) {
        g_kernels.store(&k_table_avx512, std::memory_order_release);
        return;
    }
#endif
#endif
    g_kernels.store(&k_table_naive, std::memory_order_release);
}

}  // namespace

void init() { std::call_once(g_init_flag, select_kernels); }

static float l2_entry(const float* a, const float* b, size_t dim) {
    if (dim == 0U) {
        return 0.0F;
    }
    if (a == nullptr || b == nullptr) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::call_once(g_init_flag, select_kernels);
    return g_kernels.load(std::memory_order_acquire)->l2(a, b, dim);
}

static float cosine_entry(const float* a, const float* b, size_t dim) {
    if (dim == 0U) {
        return 1.0F;
    }
    if (a == nullptr || b == nullptr) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::call_once(g_init_flag, select_kernels);
    return g_kernels.load(std::memory_order_acquire)->cos(a, b, dim);
}

float l2_distance(const float* a, const float* b, size_t dim) {
    return l2_entry(a, b, dim);
}

float cosine_distance(const float* a, const float* b, size_t dim) {
    return cosine_entry(a, b, dim);
}

}  // namespace VectorMath
}  // namespace lumina
