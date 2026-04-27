#include <gtest/gtest.h>

#include "lumina/storage/log_manager.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string temp_path(const std::string& name) {
    const auto p = std::filesystem::temp_directory_path() /
                   ("lumina_" + name + "_" + std::to_string(::getpid()) + ".wal");
    return p.string();
}

}  // namespace

TEST(LogManagerTest, AppendAndIterate) {
    const std::string path = temp_path("append_iter");
    std::filesystem::remove(path);

    lumina::LogManager log(path);
    ASSERT_TRUE(log.open().ok());

    uint64_t off0 = 0;
    uint64_t off1 = 0;
    uint64_t off2 = 0;
    ASSERT_TRUE(log.append(lumina::OpType::kPut, "k1", "v1", &off0).ok());
    ASSERT_TRUE(log.append(lumina::OpType::kPut, "k2", "v2", &off1).ok());
    ASSERT_TRUE(log.append(lumina::OpType::kPut, "k3", "v3", &off2).ok());

    std::vector<lumina::WalEntry> entries;
    ASSERT_TRUE(log.iterate([&](const lumina::WalEntry& e) {
        entries.push_back(e);
        return true;
    }).ok());

    ASSERT_EQ(entries.size(), 3U);
    EXPECT_EQ(entries[0].key, "k1");
    EXPECT_EQ(entries[0].value, "v1");
    EXPECT_EQ(entries[0].offset, off0);
    EXPECT_EQ(entries[1].key, "k2");
    EXPECT_EQ(entries[1].value, "v2");
    EXPECT_EQ(entries[1].offset, off1);
    EXPECT_EQ(entries[2].key, "k3");
    EXPECT_EQ(entries[2].value, "v3");
    EXPECT_EQ(entries[2].offset, off2);

    std::filesystem::remove(path);
}

TEST(LogManagerTest, CrcCorruption) {
    const std::string path = temp_path("corruption");
    std::filesystem::remove(path);

    {
        lumina::LogManager log(path);
        ASSERT_TRUE(log.open().ok());
        ASSERT_TRUE(log.append(lumina::OpType::kPut, "k1", "hello", nullptr).ok());
    }

    {
        std::fstream fs(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(fs.is_open());
        fs.seekp(10);  // inside payload region
        char x = 0x7F;
        fs.write(&x, 1);
    }

    lumina::LogManager log(path);
    ASSERT_TRUE(log.open().ok());

    size_t count = 0;
    ASSERT_TRUE(log.iterate([&](const lumina::WalEntry&) {
        ++count;
        return true;
    }).ok());

    EXPECT_EQ(count, 0U);
    std::filesystem::remove(path);
}

TEST(LogManagerTest, PersistAcrossReopen) {
    const std::string path = temp_path("reopen");
    std::filesystem::remove(path);

    {
        lumina::LogManager log(path);
        ASSERT_TRUE(log.open().ok());
        ASSERT_TRUE(log.append(lumina::OpType::kPut, "k1", "v1", nullptr).ok());
        ASSERT_TRUE(log.append(lumina::OpType::kPut, "k2", "v2", nullptr).ok());
        ASSERT_TRUE(log.sync().ok());
    }

    lumina::LogManager reopened(path);
    ASSERT_TRUE(reopened.open().ok());

    std::vector<std::string> keys;
    ASSERT_TRUE(reopened.iterate([&](const lumina::WalEntry& e) {
        keys.push_back(e.key);
        return true;
    }).ok());

    ASSERT_EQ(keys.size(), 2U);
    EXPECT_EQ(keys[0], "k1");
    EXPECT_EQ(keys[1], "k2");

    std::filesystem::remove(path);
}
