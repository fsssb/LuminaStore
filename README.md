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
- SIMD path selected by compile-time capability macros
- HNSW `add_item` and `search_top_k` available
- Save/load for HNSW graph index
- Unit tests and benchmark scaffolding included
