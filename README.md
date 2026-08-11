# LuminaStore

LuminaStore is a high-performance persistent vector storage engine prototype in C++20.
It provides:

- Append-only WAL for durability
- In-memory key->offset index for O(1) lookup path
- Recovery by replaying WAL on startup
- SIMD-dispatched vector distance kernels
- HNSW approximate nearest neighbor index

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Enable ASAN:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DLUMINA_ENABLE_ASAN=ON
cmake --build build-asan -j
```

If you build the AVX-512F translation unit on x86 but want to **exclude** the AVX-512 *dispatch* branch at runtime (e.g. to avoid 512-related frequency side effects, while still linking the TU), set `-DLUMINA_RUNTIME_USE_AVX512=OFF` (this sets `LUMINA_ALLOW_AVX512_KERNEL=0`).

**x86 runtime dispatch order** (when the corresponding object files are built): the implementation tries **AVX2 + FMA** first, then **AVX-512F** only if the AVX2 path was not taken. On typical CPUs that expose both, **AVX2 remains selected** — AVX-512 is used mainly when AVX2 is absent or not usable. That is a conservative default; a future `LUMINA_PREFER_AVX512` (or similar) could change that policy. “Allow AVX-512 at runtime” does **not** mean “prefer AVX-512 over AVX2”.

For a fully native-tuned build on your machine (optional, not portable to older CPUs of the same ISA):

```bash
cmake -S . -B build-native -DLUMINA_ENABLE_MARCH_NATIVE=ON
cmake --build build-native -j
```

**SIMD 与距离函数**：实现位于 `src/vector/vector_math_scalar.cpp`（标量、运行时 CPU 能力检测与派发）及可选的 `vector_math_neon.cpp` / `vector_math_avx2.cpp` / `vector_math_avx512.cpp`（由 `cmake/DetectSIMD.cmake` 决定是否编译，并为各 TU 单独加 `-mavx2`、`-mfma`、`-mavx512f` 等标志）。x86 上默认 **先尝试 AVX2，再考虑 AVX-512**；与 “允许运行时选 AVX-512” 的 CMake 开关含义见上文英语段落。`include/lumina/vector/aligned_alloc.h` 提供 64 字节对齐的 `AlignedFloatVector`，供 HNSW 节点向量等热路径使用。

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

## Benchmarks

```bash
./build/storage_bench
./build/vector_bench
```

The benchmark binaries compare:

- sequential write throughput and random read latency for storage
- naive vs SIMD-dispatched L2 distance
- HNSW top-k search latency

## WAL on-disk format (v2, default for new files)

A new WAL file starts with an 8-byte file header (big-endian multi-byte fields):

```text
[4B magic ASCII "LMST"]
[2B format version, currently 1]
[2B reserved, 0]
```

Each frame (also big-endian in the length fields used for CRC input):

```text
[1B OpType: 1=Put, 2=Delete, 3=VectorPut]
[4B CRC32]
[4B PayloadLen]
Payload: [2B KeyLen BE][key bytes][value bytes]
```

Legacy v0 (no file header) is still read if the file does not start with the magic: host-endian length fields in the frame and in the payload key length.

CRC covers `Op + encoded 4B payload length + full payload` (encoding matches the on-disk field layout).

`open()` also runs a repair pass: **incomplete data at the end of the file is truncated** with `ftruncate` to the last valid frame. A **bad CRC on the last complete frame** is treated as tail damage and can be truncated. A **bad CRC on a frame that is not at EOF** (more data after it) is **middle corruption** and `open()` returns `Status::IsCorruption()`.

I/O errors include `strerror` text from the OS.

## Group commit (batch fsync)

In `Options`:

- `group_commit` — when true, `put` / `put_vector` / `remove` do not call `fsync` every time; use `StorageEngine::sync()` or set `sync_every_n_appends > 0` to auto-fsync every N writes.
- `sync_writes` — if false, never fsync from the engine (manual `sync()` only).

## Recovery flow

`StorageEngine::open()`:

1. Opens the WAL, detects v1 vs v2, runs tail repair.
2. Replays the WAL, validates frame CRC, rejects invalid `OpType`.
3. Rebuilds the in-memory index and applies delete tombstones.

## Acceptance Checklist

- Crash recovery via WAL replay
- SIMD path selected at runtime (CPUID / CPU features) with optional per-ISA object files, not only the build host
- HNSW `add_item` and `search_top_k` available
- Save/load for HNSW graph index
- Unit tests and benchmark scaffolding included
