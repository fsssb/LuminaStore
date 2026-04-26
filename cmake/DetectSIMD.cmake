include(CheckCXXSourceRuns)

message(STATUS "Detecting SIMD capabilities...")

# Detect ARM NEON
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(CMAKE_REQUIRED_FLAGS "-O0")
    check_cxx_source_runs("
        #include <arm_neon.h>
        int main() {
            float32x4_t a = vdupq_n_f32(1.0f);
            float32x4_t b = vsubq_f32(a, a);
            (void)b;
            return 0;
        }
    " LUMINA_HAS_NEON)
    if(LUMINA_HAS_NEON)
        message(STATUS "  NEON: YES")
        add_compile_definitions(LUMINA_HAS_NEON)
    else()
        message(STATUS "  NEON: NO")
    endif()
endif()

# Detect AVX2
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
    set(CMAKE_REQUIRED_FLAGS "-mavx2")
    check_cxx_source_runs("
        #include <immintrin.h>
        int main() {
            __m256 a = _mm256_setzero_ps();
            (void)a;
            return 0;
        }
    " LUMINA_HAS_AVX2)
    if(LUMINA_HAS_AVX2)
        message(STATUS "  AVX2: YES")
        add_compile_definitions(LUMINA_HAS_AVX2)
        add_compile_options(-mavx2)
    else()
        message(STATUS "  AVX2: NO")
    endif()

    # Detect AVX-512
    set(CMAKE_REQUIRED_FLAGS "-mavx512f")
    check_cxx_source_runs("
        #include <immintrin.h>
        int main() {
            __m512 a = _mm512_setzero_ps();
            (void)a;
            return 0;
        }
    " LUMINA_HAS_AVX512)
    if(LUMINA_HAS_AVX512)
        message(STATUS "  AVX-512: YES")
        add_compile_definitions(LUMINA_HAS_AVX512)
        add_compile_options(-mavx512f)
    else()
        message(STATUS "  AVX-512: NO")
    endif()
endif()

unset(CMAKE_REQUIRED_FLAGS)
