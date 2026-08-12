cmake_minimum_required(VERSION 3.20)
project(LuminaStore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(LUMINA_ENABLE_MARCH_NATIVE "Enable -march=native for local optimized builds" OFF)
# 若已编译 AVX-512F 目标文件，是否允许运行时在通过特性检测时选用之（关闭则始终不选 AVX-512 路径）
option(LUMINA_RUNTIME_USE_AVX512 "Allow runtime dispatch to AVX-512F kernels when the TU is built" ON)

# Compiler flags
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-O0>
)

if(LUMINA_ENABLE_MARCH_NATIVE)
    if(MSVC)
        # MSVC 没有 -march=native；在 Windows 上请改用对应 /arch: 或忽略此选项
        message(WARNING "LUMINA_ENABLE_MARCH_NATIVE: MSVC 不支持 -march=native，已跳过")
    else()
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
endif()

# AddressSanitizer support
option(LUMINA_ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(LUMINA_ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# SIMD: which *optional* TUs to build; flags are per-file (not global -mavx*).
include(cmake/DetectSIMD.cmake)

# Include directories
include_directories(include)

# ---- Core library ----
file(GLOB_RECURSE LUMINA_SOURCES
    src/common/*.cpp
    src/engine/*.cpp
    src/index/*.cpp
    src/storage/*.cpp
)
file(GLOB LUMINA_VECTOR_BASE src/vector/*.cpp)
list(REMOVE_ITEM LUMINA_VECTOR_BASE
    ${CMAKE_SOURCE_DIR}/src/vector/vector_math_neon.cpp
    ${CMAKE_SOURCE_DIR}/src/vector/vector_math_avx2.cpp
    ${CMAKE_SOURCE_DIR}/src/vector/vector_math_avx512.cpp
)
if(LUMINA_SIMD_NEON)
    list(APPEND LUMINA_VECTOR_BASE ${CMAKE_SOURCE_DIR}/src/vector/vector_math_neon.cpp)
endif()
if(LUMINA_SIMD_AVX2)
    list(APPEND LUMINA_VECTOR_BASE ${CMAKE_SOURCE_DIR}/src/vector/vector_math_avx2.cpp)
endif()
if(LUMINA_SIMD_AVX512)
    list(APPEND LUMINA_VECTOR_BASE ${CMAKE_SOURCE_DIR}/src/vector/vector_math_avx512.cpp)
endif()
list(APPEND LUMINA_SOURCES ${LUMINA_VECTOR_BASE})

add_library(lumina_core STATIC ${LUMINA_SOURCES})
target_include_directories(lumina_core PUBLIC include)
set_target_properties(lumina_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LUMINA_RUNTIME_USE_AVX512)
    set(LUMINA_DEF_ALLOW_AVX512_RT 1)
else()
    set(LUMINA_DEF_ALLOW_AVX512_RT 0)
endif()

if(LUMINA_SIMD_NEON)
    set(LUMINA_DEF_NEON 1)
else()
    set(LUMINA_DEF_NEON 0)
endif()

if(LUMINA_SIMD_AVX2)
    set(LUMINA_DEF_AVX2 1)
    if(MSVC)
        set_source_files_properties(
            src/vector/vector_math_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
        )
    else()
        set_source_files_properties(
            src/vector/vector_math_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
        )
    endif()
else()
    set(LUMINA_DEF_AVX2 0)
endif()

if(LUMINA_SIMD_AVX512)
    set(LUMINA_DEF_AVX512 1)
    if(MSVC)
        set_source_files_properties(
            src/vector/vector_math_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX512"
        )
    else()
        set_source_files_properties(
            src/vector/vector_math_avx512.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx512f"
        )
    endif()
else()
    set(LUMINA_DEF_AVX512 0)
endif()

target_compile_definitions(
    lumina_core
    PUBLIC
        LUMINA_BUILD_NEON=${LUMINA_DEF_NEON}
        LUMINA_BUILD_AVX2=${LUMINA_DEF_AVX2}
        LUMINA_BUILD_AVX512=${LUMINA_DEF_AVX512}
        LUMINA_ALLOW_AVX512_KERNEL=${LUMINA_DEF_ALLOW_AVX512_RT}
)

# ---- C API shared library ----
add_library(luminastore_shared SHARED cpp_engine/c_api.cpp)
target_link_libraries(luminastore_shared PRIVATE lumina_core)
target_include_directories(luminastore_shared PRIVATE include)
set_target_properties(
    luminastore_shared
    PROPERTIES
        OUTPUT_NAME "luminastore_shared"
)
add_custom_command(
    TARGET luminastore_shared
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_SOURCE_DIR}/../Tax_RAG_demo/cpp_engine/build"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:luminastore_shared>"
            "${CMAKE_SOURCE_DIR}/../Tax_RAG_demo/cpp_engine/build/"
    COMMENT "Copy luminastore_shared to Tax_RAG_demo/cpp_engine/build"
)

# ---- Google Test ----
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ---- Tests ----
file(GLOB TEST_SOURCES tests/*.cpp)
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE lumina_core GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ---- Google Benchmark ----
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# ---- Benchmarks ----
file(GLOB BENCH_SOURCES bench/*.cpp)
foreach(bench_src ${BENCH_SOURCES})
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_link_libraries(${bench_name} PRIVATE lumina_core benchmark::benchmark)
endforeach()
