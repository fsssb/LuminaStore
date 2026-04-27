#include <gtest/gtest.h>

#include "lumina/storage/storage_engine.h"

#include <filesystem>
#include <string>
#include <unistd.h>

namespace {

std::string temp_wal(const std::string& suffix) {
    const auto p = std::filesystem::temp_directory_path() /
                   ("lumina_engine_" + suffix + "_" + std::to_string(::getpid()) + ".wal");
    return p.string();
}

}  // namespace

TEST(StorageEngineTest, PutAndGet) {
    const std::string path = temp_wal("put_get");
    std::filesystem::remove(path);

    lumina::Options opts;
    opts.wal_path = path;
    opts.sync_writes = true;

    lumina::StorageEngine eng(opts);
    ASSERT_TRUE(eng.open().ok());
    ASSERT_TRUE(eng.put("a", "1").ok());
    ASSERT_TRUE(eng.put("b", "2").ok());

    std::string out;
    ASSERT_TRUE(eng.get("a", &out).ok());
    EXPECT_EQ(out, "1");
    ASSERT_TRUE(eng.get("b", &out).ok());
    EXPECT_EQ(out, "2");

    std::filesystem::remove(path);
}

TEST(StorageEngineTest, DeleteKey) {
    const std::string path = temp_wal("delete");
    std::filesystem::remove(path);

    lumina::Options opts;
    opts.wal_path = path;

    lumina::StorageEngine eng(opts);
    ASSERT_TRUE(eng.open().ok());
    ASSERT_TRUE(eng.put("x", "value").ok());
    ASSERT_TRUE(eng.remove("x").ok());

    std::string out;
    const auto s = eng.get("x", &out);
    EXPECT_TRUE(s.IsNotFound());

    std::filesystem::remove(path);
}

TEST(StorageEngineTest, RecoveryAfterReopen) {
    const std::string path = temp_wal("recover");
    std::filesystem::remove(path);

    {
        lumina::Options opts;
        opts.wal_path = path;
        opts.sync_writes = true;

        lumina::StorageEngine eng(opts);
        ASSERT_TRUE(eng.open().ok());
        ASSERT_TRUE(eng.put("k1", "v1").ok());
        ASSERT_TRUE(eng.put("k2", "v2").ok());
        ASSERT_TRUE(eng.remove("k1").ok());
    }

    lumina::Options opts2;
    opts2.wal_path = path;
    opts2.sync_writes = true;
    lumina::StorageEngine reopened(opts2);
    ASSERT_TRUE(reopened.open().ok());

    std::string out;
    EXPECT_TRUE(reopened.get("k1", &out).IsNotFound());
    ASSERT_TRUE(reopened.get("k2", &out).ok());
    EXPECT_EQ(out, "v2");

    std::filesystem::remove(path);
}

TEST(StorageEngineTest, GetNonExistent) {
    const std::string path = temp_wal("missing");
    std::filesystem::remove(path);

    lumina::Options opts;
    opts.wal_path = path;

    lumina::StorageEngine eng(opts);
    ASSERT_TRUE(eng.open().ok());

    std::string out;
    const auto s = eng.get("no_such_key", &out);
    EXPECT_TRUE(s.IsNotFound());

    std::filesystem::remove(path);
}

TEST(StorageEngineTest, GroupCommitFsyncsEveryN) {
    const std::string path = temp_wal("group");
    std::filesystem::remove(path);

    lumina::Options opts;
    opts.wal_path              = path;
    opts.sync_writes           = true;
    opts.group_commit          = true;
    opts.sync_every_n_appends  = 2;

    {
        lumina::StorageEngine eng(opts);
        ASSERT_TRUE(eng.open().ok());
        ASSERT_TRUE(eng.put("a", "1").ok());
        ASSERT_TRUE(eng.put("b", "2").ok());
    }

    {
        lumina::StorageEngine again(opts);
        ASSERT_TRUE(again.open().ok());
        std::string v;
        ASSERT_TRUE(again.get("a", &v).ok());
        EXPECT_EQ(v, "1");
        ASSERT_TRUE(again.get("b", &v).ok());
        EXPECT_EQ(v, "2");
    }
    std::filesystem::remove(path);
}
