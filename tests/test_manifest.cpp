#include <gtest/gtest.h>

#include "lumina/storage/manifest.h"
#include "lumina/storage/snapshot.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <unordered_map>

namespace {

std::string temp_dir(const std::string& name) {
    const auto p = std::filesystem::temp_directory_path() /
                   ("lumina_manifest_" + name + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(p);
    std::filesystem::create_directories(p);
    return p.string();
}

}  // namespace

TEST(ManifestTest, AppendAndReadLatest) {
    const std::string dir = temp_dir("basic");
    const std::string path = dir + "/MANIFEST";

    lumina::ManifestEntry e1{1, 4096, "snap-1.snap"};
    lumina::ManifestEntry e2{2, 8192, "snap-2.snap"};
    ASSERT_TRUE(lumina::write_manifest_append(path, e1).ok());
    ASSERT_TRUE(lumina::write_manifest_append(path, e2).ok());

    lumina::ManifestEntry out;
    ASSERT_TRUE(lumina::read_manifest_latest(path, &out).ok());
    EXPECT_EQ(out.seq, 2U);
    EXPECT_EQ(out.wal_offset, 8192U);
    EXPECT_EQ(out.filename, "snap-2.snap");
}

TEST(ManifestTest, KeepsOnlyRecent) {
    const std::string dir = temp_dir("recent");
    const std::string path = dir + "/MANIFEST";
    for (uint64_t i = 1; i <= 5; ++i) {
        ASSERT_TRUE(lumina::write_manifest_append(
                        path, lumina::ManifestEntry{i, i * 100, "snap-" + std::to_string(i) + ".snap"})
                        .ok());
    }
    lumina::ManifestEntry out;
    ASSERT_TRUE(lumina::read_manifest_latest(path, &out).ok());
    EXPECT_EQ(out.seq, 5U);
}

TEST(ManifestTest, MissingManifest) {
    const std::string dir = temp_dir("missing");
    lumina::ManifestEntry out;
    EXPECT_TRUE(lumina::read_manifest_latest(dir + "/MANIFEST", &out).IsNotFound());
}

TEST(ManifestTest, RejectsGarbageLine) {
    const std::string dir = temp_dir("garbage");
    const std::string path = dir + "/MANIFEST";
    {
        std::ofstream ofs(path);
        ofs << "this is not a valid manifest line\n";
    }
    lumina::ManifestEntry out;
    EXPECT_TRUE(lumina::read_manifest_latest(path, &out).IsCorruption());
}

TEST(SnapshotTest, WriteReadRoundTrip) {
    const std::string dir = temp_dir("snap_roundtrip");
    const std::string path = dir + "/snap-1.snap";

    std::unordered_map<std::string, uint64_t> index = {
        {"hello", 4096},
        {"world", 8192},
        {"tax", 12288},
    };
    ASSERT_TRUE(lumina::write_snapshot(path, 12345, index).ok());

    lumina::SnapshotMeta meta;
    ASSERT_TRUE(lumina::read_snapshot(path, &meta).ok());
    EXPECT_EQ(meta.wal_offset, 12345U);
    ASSERT_EQ(meta.index.size(), 3U);
    EXPECT_EQ(meta.index["hello"], 4096U);
    EXPECT_EQ(meta.index["world"], 8192U);
    EXPECT_EQ(meta.index["tax"], 12288U);
}

TEST(SnapshotTest, EmptyIndex) {
    const std::string dir = temp_dir("snap_empty");
    const std::string path = dir + "/snap-empty.snap";
    ASSERT_TRUE(lumina::write_snapshot(path, 0, {}).ok());
    lumina::SnapshotMeta meta;
    ASSERT_TRUE(lumina::read_snapshot(path, &meta).ok());
    EXPECT_EQ(meta.wal_offset, 0U);
    EXPECT_TRUE(meta.index.empty());
}

TEST(SnapshotTest, RejectsCorruptCrc) {
    const std::string dir = temp_dir("snap_crc");
    const std::string path = dir + "/snap-corrupt.snap";
    ASSERT_TRUE(lumina::write_snapshot(path, 100, {{"a", 1}}).ok());

    // Flip a byte in the body (skip the 4-byte trailing CRC).
    {
        std::fstream fs(path, std::ios::binary | std::ios::in | std::ios::out);
        fs.seekp(0, std::ios::beg);
        char c = 0;
        fs.read(&c, 1);
        c ^= 0xFF;
        fs.seekp(0, std::ios::beg);
        fs.write(&c, 1);
    }
    lumina::SnapshotMeta meta;
    EXPECT_TRUE(lumina::read_snapshot(path, &meta).IsCorruption());
}

TEST(SnapshotTest, RejectsBadMagic) {
    const std::string dir = temp_dir("snap_magic");
    const std::string path = dir + "/snap-bad.snap";
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs << "NOTLMST";
        ofs << std::string(64, '\0');
    }
    lumina::SnapshotMeta meta;
    EXPECT_TRUE(lumina::read_snapshot(path, &meta).IsCorruption());
}

TEST(SnapshotTest, MissingFile) {
    lumina::SnapshotMeta meta;
    EXPECT_TRUE(lumina::read_snapshot("/nonexistent/path.snap", &meta).IsNotFound());
}
