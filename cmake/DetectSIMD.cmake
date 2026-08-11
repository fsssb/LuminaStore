# 设置: LUMINA_SIMD_NEON, LUMINA_SIMD_AVX2, LUMINA_SIMD_AVX512
# 仅做「编译器能否生成对应 intrinsic」的检测；实际用哪条路径由运行时在 vector_math_scalar.cpp 决定。

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

set(LUMINA_SIMD_NEON   OFF)
set(LUMINA_SIMD_AVX2   OFF)
set(LUMINA_SIMD_AVX512 OFF)

message(STATUS "Detecting optional SIMD TUs (compile-only checks; no global -mavx* flags)...")

# ---- ARM64 NEON ----
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main() {
            float32x4_t a = vdupq_n_f32(1.0f);
            (void)a;
            return 0;
        }
    " LUMINA_CC_HAS_NEON_HEADERS)
    if(LUMINA_CC_HAS_NEON_HEADERS)
        set(LUMINA_SIMD_NEON ON)
        message(STATUS "  NEON TU: enabled (arm_neon.h available)")
    else()
        message(STATUS "  NEON TU: disabled (compiler test failed)")
    endif()
endif()

# ---- x86 AVX2 + FMA ----
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|x86|X86|amd64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        cmake_push_check_state()
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m256 a = _mm256_set1_ps(1.0f);
                __m256 b = _mm256_set1_ps(2.0f);
                __m256 c = _mm256_fmadd_ps(a, b, a);
                (void)_mm256_cvtss_f32(c);
                return 0;
            }
        " LUMINA_CC_AVX2_MSVC)
        cmake_pop_check_state()
        if(LUMINA_CC_AVX2_MSVC)
            set(LUMINA_SIMD_AVX2 ON)
        endif()
    else()
        cmake_push_check_state()
        set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m256 a = _mm256_set1_ps(1.0f);
                __m256 b = _mm256_set1_ps(2.0f);
                __m256 c = _mm256_fmadd_ps(a, b, a);
                (void)_mm256_cvtss_f32(c);
                return 0;
            }
        " LUMINA_CC_AVX2_GNU)
        cmake_pop_check_state()
        if(LUMINA_CC_AVX2_GNU)
            set(LUMINA_SIMD_AVX2 ON)
        endif()
    endif()
    if(LUMINA_SIMD_AVX2)
        message(STATUS "  AVX2+FMA TU: enabled (vector_math_avx2.cpp 使用 -mavx2 -mfma 或 /arch:AVX2)")
    else()
        message(STATUS "  AVX2+FMA TU: disabled (compile check failed for this host toolchain)")
    endif()
endif()

# ---- x86 AVX-512F ----
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|x86|X86|amd64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        cmake_push_check_state()
        set(CMAKE_REQUIRED_FLAGS "/arch:AVX512")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m512 a = _mm512_set1_ps(1.0f);
                __m512 b = _mm512_set1_ps(2.0f);
                __m512 c = _mm512_fmadd_ps(a, b, a);
                (void)_mm512_cvtss_f32(c);
                return 0;
            }
        " LUMINA_CC_AVX512_MSVC)
        cmake_pop_check_state()
        if(LUMINA_CC_AVX512_MSVC)
            set(LUMINA_SIMD_AVX512 ON)
        endif()
    else()
        cmake_push_check_state()
        set(CMAKE_REQUIRED_FLAGS "-mavx512f")
        check_cxx_source_compiles("
            #include <immintrin.h>
            int main() {
                __m512 a = _mm512_set1_ps(1.0f);
                __m512 b = _mm512_set1_ps(2.0f);
                __m512 c = _mm512_fmadd_ps(a, b, a);
                (void)_mm512_cvtss_f32(c);
                return 0;
            }
        " LUMINA_CC_AVX512_GNU)
        cmake_pop_check_state()
        if(LUMINA_CC_AVX512_GNU)
            set(LUMINA_SIMD_AVX512 ON)
        endif()
    endif()
    if(LUMINA_SIMD_AVX512)
        message(STATUS "  AVX-512F TU: enabled (vector_math_avx512.cpp 使用 -mavx512f 或 /arch:AVX512)")
    else()
        message(STATUS "  AVX-512F TU: disabled (compile check failed for this host toolchain)")
    endif()
endif()
