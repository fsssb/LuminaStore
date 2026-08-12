#include <gtest/gtest.h>

#include "lumina/storage/entry_meta.h"
#include "lumina/storage/log_manager.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string temp_path(const std::string& name) {
    const auto p = std::filesystem::temp_directory_path() /
                   ("lumina_" + name + "_" + std::to_string(::getpid()) + ".wal");
    return p.string();
}

lumina::EntryMeta make_full_meta() {
    lumina::EntryMeta m;
    m.vec = {1.0f, 2.0f, 3.0f, 4.0f};
    m.payload = "tax chunk payload";
    m.scalars = {
        {"cat", int64_t(42)},
        {"score", 3.14},
        {"note", std::string("hello")},
    };
    return m;
}

void expect_meta_eq(const lumina::EntryMeta& a, const lumina::EntryMeta& b) {
    EXPECT_EQ(a.vec, b.vec);
    EXPECT_EQ(a.payload, b.payload);
    ASSERT_EQ(a.scalars.size(), b.scalars.size());
    // Encoding sorts scalars by name; treat them as an unordered set keyed by name.
    for (const auto& fa : a.scalars) {
        const auto it = std::find_if(b.scalars.begin(), b.scalars.end(),
                                     [&](const lumina::ScalarField& fb) { return fb.name == fa.name; });
        ASSERT_NE(it, b.scalars.end()) << "missing scalar: " << fa.name;
        EXPECT_EQ(it->value, fa.value) << "scalar: " << fa.name;
    }
}

}  // namespace

TEST(EntryMetaTest, RoundTripEmpty) {
    lumina::EntryMeta m;
    std::string bytes;
    ASSERT_TRUE(lumina::encode_entry_meta(m, &bytes).ok());
    lumina::EntryMeta out;
    ASSERT_TRUE(lumina::decode_entry_meta(bytes, &out).ok());
    expect_meta_eq(m, out);
}

TEST(EntryMetaTest, RoundTripVecOnly) {
    lumina::EntryMeta m;
    m.vec = {0.5f, -1.25f, 3.0f};
    std::string bytes;
    ASSERT_TRUE(lumina::encode_entry_meta(m, &bytes).ok());
    lumina::EntryMeta out;
    ASSERT_TRUE(lumina::decode_entry_meta(bytes, &out).ok());
    expect_meta_eq(m, out);
}

TEST(EntryMetaTest, RoundTripPayloadOnly) {
    lumina::EntryMeta m;
    m.payload = std::string("\x00\x01\x02", 3);  // binary-safe payload
    std::string bytes;
    ASSERT_TRUE(lumina::encode_entry_meta(m, &bytes).ok());
    lumina::EntryMeta out;
    ASSERT_TRUE(lumina::decode_entry_meta(bytes, &out).ok());
    expect_meta_eq(m, out);
}

TEST(EntryMetaTest, RoundTripScalarsOnly) {
    lumina::EntryMeta m;
    m.scalars = {
        {"a", int64_t(-9223372036854775807LL - 1)},
        {"b", 2.718281828},
        {"c", std::string()},  // empty string scalar
        {"d", int64_t(0)},
    };
    std::string bytes;
    ASSERT_TRUE(lumina::encode_entry_meta(m, &bytes).ok());
    lumina::EntryMeta out;
    ASSERT_TRUE(lumina::decode_entry_meta(bytes, &out).ok());
    // Encoding sorts by name; compare by lookup.
    ASSERT_EQ(out.scalars.size(), 4U);
    EXPECT_EQ(out.scalars[0].name, "a");
    EXPECT_EQ(out.scalars[0].value, m.scalars[0].value);
    EXPECT_EQ(out.scalars[1].name, "b");
    EXPECT_EQ(out.scalars[1].value, m.scalars[1].value);
    EXPECT_EQ(out.scalars[2].name, "c");
    EXPECT_EQ(out.scalars[2].value, m.scalars[2].value);
    EXPECT_EQ(out.scalars[3].name, "d");
    EXPECT_EQ(out.scalars[3].value, m.scalars[3].value);
}

TEST(EntryMetaTest, RoundTripFull) {
    lumina::EntryMeta m = make_full_meta();
    std::string bytes;
    ASSERT_TRUE(lumina::encode_entry_meta(m, &bytes).ok());
    lumina::EntryMeta out;
    ASSERT_TRUE(lumina::decode_entry_meta(bytes, &out).ok());
    expect_meta_eq(m, out);
}

TEST(EntryMetaTest, RejectsTruncated) {
    lumina::EntryMeta m = make_full_meta();
    std::string bytes;
    ASSERT_TRUE(lumina::encode_entry_meta(m, &bytes).ok());
    for (size_t cut = 0; cut < bytes.size(); ++cut) {
        lumina::EntryMeta out;
        EXPECT_FALSE(lumina::decode_entry_meta(bytes.substr(0, cut), &out).ok())
            << "should reject truncation at byte " << cut;
    }
}

TEST(EntryMetaTest, RejectsUnknownFlags) {
    std::string bytes = std::string("\x08\x00\x00\x00", 4);  // flag bit3 unknown
    lumina::EntryMeta out;
    EXPECT_FALSE(lumina::decode_entry_meta(bytes, &out).ok());
}

TEST(EntryMetaTest, RejectsTrailingBytes) {
    lumina::EntryMeta m = make_full_meta();
    std::string bytes;
    ASSERT_TRUE(lumina::encode_entry_meta(m, &bytes).ok());
    bytes.push_back('\xAB');
    lumina::EntryMeta out;
    EXPECT_FALSE(lumina::decode_entry_meta(bytes, &out).ok());
}

TEST(EntryMetaTest, RejectsUnknownScalarType) {
    // flags = scalars(0x04), nscalars=1, type=0x07(unknown), key_len=1, key='a'
    std::string bytes;
    bytes.push_back(0x04); bytes.push_back(0x00); bytes.push_back(0x00); bytes.push_back(0x00);
    bytes.push_back(0x01); bytes.push_back(0x00);  // nscalars = 1
    bytes.push_back(0x07);                          // unknown type
    bytes.push_back(0x01); bytes.push_back(0x00);  // key_len = 1
    bytes.push_back('a');
    lumina::EntryMeta out;
    EXPECT_FALSE(lumina::decode_entry_meta(bytes, &out).ok());
}

TEST(EntryMetaTest, WalVectorPutV2RoundTrip) {
    const std::string path = temp_path("v2_meta");
    std::filesystem::remove(path);

    lumina::EntryMeta m = make_full_meta();
    std::string value;
    ASSERT_TRUE(lumina::encode_entry_meta(m, &value).ok());

    uint64_t off = 0;
    {
        lumina::LogManager log(path);
        ASSERT_TRUE(log.open().ok());
        ASSERT_TRUE(log.append(lumina::OpType::kVectorPutV2, "42", value, &off).ok());
        ASSERT_TRUE(log.sync().ok());
    }

    {
        lumina::LogManager log(path);
        ASSERT_TRUE(log.open().ok());
        bool found = false;
        ASSERT_TRUE(log.iterate([&](const lumina::WalEntry& e) {
            EXPECT_EQ(e.op_type, lumina::OpType::kVectorPutV2);
            EXPECT_EQ(e.key, "42");
            EXPECT_EQ(e.offset, off);
            lumina::EntryMeta out;
            EXPECT_TRUE(lumina::decode_entry_meta(e.value, &out).ok());
            expect_meta_eq(m, out);
            found = true;
            return true;
        }).ok());
        EXPECT_TRUE(found);

        std::string rkey, rvalue;
        ASSERT_TRUE(log.read_value_at(off, &rkey, &rvalue).ok());
        EXPECT_EQ(rkey, "42");
        lumina::EntryMeta out;
        ASSERT_TRUE(lumina::decode_entry_meta(rvalue, &out).ok());
        expect_meta_eq(m, out);
    }

    std::filesystem::remove(path);
}

TEST(EntryMetaTest, WalRejectsInvalidOpStill) {
    const std::string path = temp_path("bad_op");
    std::filesystem::remove(path);
    {
        lumina::LogManager log(path);
        ASSERT_TRUE(log.open().ok());
        // kVectorPutV2 with raw bytes (not valid meta) is a *valid WAL record*; decoding
        // is the engine's job. Sanity check that the opcode itself is accepted.
        uint64_t off = 0;
        ASSERT_TRUE(log.append(lumina::OpType::kVectorPutV2, "x", "not-meta", &off).ok());
    }
    {
        lumina::LogManager log(path);
        ASSERT_TRUE(log.open().ok());
        bool found = false;
        ASSERT_TRUE(log.iterate([&](const lumina::WalEntry& e) {
            EXPECT_EQ(e.op_type, lumina::OpType::kVectorPutV2);
            found = true;
            return true;
        }).ok());
        EXPECT_TRUE(found);
    }
    std::filesystem::remove(path);
}
