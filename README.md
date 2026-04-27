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

## WAL Frame Format

Binary layout:

```text
[1B OpType][4B CRC32][4B PayloadLen][Payload]
Payload = [2B KeyLen][Key][Value]
```

CRC32 covers `OpType + PayloadLen + Payload`.

## Recovery Flow

On startup, `StorageEngine::open()` does:

1. open/create WAL file
2. iterate WAL from offset 0
3. validate frame CRC
4. rebuild in-memory key->offset index
5. apply tombstones for deletes

## Acceptance Checklist

- Crash recovery via WAL replay
- SIMD path selected by compile-time capability macros
- HNSW `add_item` and `search_top_k` available
- Save/load for HNSW graph index
- Unit tests and benchmark scaffolding included
