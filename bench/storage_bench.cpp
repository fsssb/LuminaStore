#include <benchmark/benchmark.h>

#include "lumina/storage/storage_engine.h"

#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

std::string bench_wal_path() {
    const auto p = std::filesystem::temp_directory_path() / "lumina_storage_bench.wal";
    return p.string();
}

static void BM_StorageSequentialWrite(benchmark::State& state) {
    const std::string path = bench_wal_path();
    std::filesystem::remove(path);

    lumina::Options opts;
    opts.wal_path = path;
    opts.sync_writes = false;

    lumina::StorageEngine engine(opts);
    if (!engine.open().ok()) {
        state.SkipWithError("failed to open storage engine");
        return;
    }

    size_t i = 0;
    for (auto _ : state) {
        std::string key = "key_" + std::to_string(i);
        std::string value(128, 'v');
        auto s = engine.put(key, value);
        if (!s.ok()) {
            state.SkipWithError(s.ToString().c_str());
            break;
        }
        ++i;
    }

    benchmark::DoNotOptimize(i);
    std::filesystem::remove(path);
}

static void BM_StorageRandomRead(benchmark::State& state) {
    const std::string path = bench_wal_path();
    std::filesystem::remove(path);

    lumina::Options opts;
    opts.wal_path = path;
    opts.sync_writes = false;

    lumina::StorageEngine engine(opts);
    if (!engine.open().ok()) {
        state.SkipWithError("failed to open storage engine");
        return;
    }

    constexpr int kPreload = 10000;
    for (int i = 0; i < kPreload; ++i) {
        std::string key = "key_" + std::to_string(i);
        std::string value(128, 'a' + (i % 26));
        auto s = engine.put(key, value);
        if (!s.ok()) {
            state.SkipWithError(s.ToString().c_str());
            return;
        }
    }

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, kPreload - 1);
    std::string out;

    for (auto _ : state) {
        std::string key = "key_" + std::to_string(dist(rng));
        auto s = engine.get(key, &out);
        if (!s.ok()) {
            state.SkipWithError(s.ToString().c_str());
            break;
        }
        benchmark::DoNotOptimize(out);
    }

    std::filesystem::remove(path);
}

BENCHMARK(BM_StorageSequentialWrite);
BENCHMARK(BM_StorageRandomRead);

}  // namespace

BENCHMARK_MAIN();
