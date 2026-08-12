#include <gtest/gtest.h>

#include "lumina/storage/storage_engine.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

struct TestPaths {
    std::string wal;
    std::string snap_dir;
};

TestPaths make_paths(const std::string& name) {
    const auto base = std::filesystem::temp_directory_path() /
                      ("lumina_recovery_" + name + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(base);
    return {base.string() + ".wal", base.string() + "_snap"};
}

lumina::Options make_opts(const TestPaths& p) {
    lumina::Options opts;
    opts.wal_path = p.wal;
    opts.snapshot_dir = p.snap_dir;
    opts.sync_writes = true;
    return opts;
}

void put_range(lumina::StorageEngine& eng, uint64_t begin, uint64_t end) {
    for (uint64_t i = begin; i < end; ++i) {
        ASSERT_TRUE(eng.put("k" + std::to_string(i), "v" + std::to_string(i)).ok());
    }
}

}  // namespace

TEST(RecoveryTest, ReopenWithoutSnapshotReplaysAll) {
    const auto p = make_paths("no_snap");
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        put_range(eng, 0, 100);
    }
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        EXPECT_EQ(eng.key_count(), 100U);
        std::string out;
        ASSERT_TRUE(eng.get("k42", &out).ok());
        EXPECT_EQ(out, "v42");
    }
}

TEST(RecoveryTest, SnapshotThenReopen) {
    const auto p = make_paths("snap_reopen");
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        put_range(eng, 0, 100);
        ASSERT_TRUE(eng.snapshot().ok());
    }
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        EXPECT_EQ(eng.key_count(), 100U);
        std::string out;
        ASSERT_TRUE(eng.get("k99", &out).ok());
        EXPECT_EQ(out, "v99");
    }
}

TEST(RecoveryTest, SnapshotThenIncrementalWrites) {
    const auto p = make_paths("snap_incremental");
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        put_range(eng, 0, 50);
        ASSERT_TRUE(eng.snapshot().ok());
        // Writes after the snapshot watermark must be replayed on reopen.
        put_range(eng, 50, 120);
        ASSERT_TRUE(eng.remove("k10").ok());
    }
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        EXPECT_EQ(eng.key_count(), 119U);  // 120 - 1 removed
        std::string out;
        ASSERT_TRUE(eng.get("k119", &out).ok());
        EXPECT_EQ(out, "v119");
        EXPECT_TRUE(eng.get("k10", &out).IsNotFound());
        ASSERT_TRUE(eng.get("k5", &out).ok());
        EXPECT_EQ(out, "v5");
    }
}

TEST(RecoveryTest, MultipleSnapshotsUsesLatest) {
    const auto p = make_paths("snap_multiple");
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        put_range(eng, 0, 30);
        ASSERT_TRUE(eng.snapshot().ok());
        put_range(eng, 30, 60);
        ASSERT_TRUE(eng.snapshot().ok());
        put_range(eng, 60, 90);
    }
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        EXPECT_EQ(eng.key_count(), 90U);
        std::string out;
        ASSERT_TRUE(eng.get("k85", &out).ok());
        EXPECT_EQ(out, "v85");
    }
}

TEST(RecoveryTest, CorruptSnapshotFallsBackToFullReplay) {
    const auto p = make_paths("snap_corrupt");
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        put_range(eng, 0, 50);
        ASSERT_TRUE(eng.snapshot().ok());
    }
    // Corrupt every snapshot file in the dir.
    for (const auto& entry : std::filesystem::directory_iterator(p.snap_dir)) {
        if (entry.path().extension() == ".snap") {
            std::fstream fs(entry.path(), std::ios::binary | std::ios::in | std::ios::out);
            char c = 0;
            fs.read(&c, 1);
            c ^= 0xFF;
            fs.seekp(0);
            fs.write(&c, 1);
        }
    }
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());  // must not fail; falls back to full replay
        EXPECT_EQ(eng.key_count(), 50U);
        std::string out;
        ASSERT_TRUE(eng.get("k7", &out).ok());
        EXPECT_EQ(out, "v7");
    }
}

TEST(RecoveryTest, DeleteTombstoneInSnapshot) {
    const auto p = make_paths("snap_delete");
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        put_range(eng, 0, 20);
        ASSERT_TRUE(eng.remove("k3").ok());
        ASSERT_TRUE(eng.snapshot().ok());
    }
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        EXPECT_EQ(eng.key_count(), 19U);
        std::string out;
        EXPECT_TRUE(eng.get("k3", &out).IsNotFound());
    }
}

TEST(RecoveryTest, CrashWithoutSyncTailRepair) {
    // Simulate a crash: writes are buffered (group_commit with no auto sync),
    // the engine object is destroyed without fsync (part of the tail may be
    // lost), then we reopen. Data that was fsynced must survive.
    const auto p = make_paths("crash_nosync");
    {
        lumina::Options opts = make_opts(p);
        opts.group_commit = true;
        opts.sync_every_n_appends = 0;  // never auto-fsync
        opts.sync_writes = true;
        lumina::StorageEngine eng(opts);
        ASSERT_TRUE(eng.open().ok());
        put_range(eng, 0, 50);
        ASSERT_TRUE(eng.sync().ok());  // durable up to here
        put_range(eng, 50, 60);        // these may be lost (not fsynced)
    }
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());  // tail repair truncates partial/lost tail
        // At least the first 50 are durable; 50..60 may or may not survive.
        EXPECT_GE(eng.key_count(), 50U);
        EXPECT_LE(eng.key_count(), 60U);
        std::string out;
        ASSERT_TRUE(eng.get("k10", &out).ok());
        EXPECT_EQ(out, "v10");
    }
}

TEST(RecoveryTest, SnapshotWatermarkNotAheadOfDurableData) {
    // snapshot() must fsync the WAL first so the watermark never points past
    // durable bytes; verify reopen always sees exactly the snapshot content
    // plus any replayed tail.
    const auto p = make_paths("snap_watermark");
    {
        lumina::Options opts = make_opts(p);
        opts.group_commit = true;
        opts.sync_every_n_appends = 0;
        lumina::StorageEngine eng(opts);
        ASSERT_TRUE(eng.open().ok());
        put_range(eng, 0, 40);
        ASSERT_TRUE(eng.snapshot().ok());  // internally syncs WAL
        put_range(eng, 40, 70);
    }
    {
        lumina::StorageEngine eng(make_opts(p));
        ASSERT_TRUE(eng.open().ok());
        // Snapshot covers 0..40 (fsynced); 40..70 may be partially lost, but
        // never less than the snapshot content.
        EXPECT_GE(eng.key_count(), 40U);
        std::string out;
        ASSERT_TRUE(eng.get("k39", &out).ok());
        EXPECT_EQ(out, "v39");
    }
}
