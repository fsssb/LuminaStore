#include <gtest/gtest.h>

#include "lumina/engine/collection.h"
#include "lumina/index/filter_index.h"

#include <filesystem>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

std::string temp_dir(const std::string& name) {
    const auto p = std::filesystem::temp_directory_path() /
                   ("lumina_filter_" + name + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(p);
    return p.string();
}

std::vector<float> random_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> out(dim);
    for (auto& x : out) {
        x = dist(rng);
    }
    return out;
}

}  // namespace

TEST(FilterIndexTest, BasicPredicates) {
    lumina::FilterIndex fi;
    fi.add(1, {{"cat", int64_t(42)}, {"price", 9.5}, {"tag", std::string("tax")}});
    fi.add(2, {{"cat", int64_t(7)}, {"price", 3.0}});
    fi.add(3, {{"cat", int64_t(42)}, {"price", 100.0}});

    lumina::FilterExpr eq;
    eq.clauses.push_back({"cat", lumina::FilterOp::kEq, int64_t(42)});
    EXPECT_TRUE(fi.matches(1, eq));
    EXPECT_FALSE(fi.matches(2, eq));
    EXPECT_TRUE(fi.matches(3, eq));

    lumina::FilterExpr ne;
    ne.clauses.push_back({"cat", lumina::FilterOp::kNe, int64_t(42)});
    EXPECT_FALSE(fi.matches(1, ne));
    EXPECT_TRUE(fi.matches(2, ne));

    lumina::FilterExpr lt;
    lt.clauses.push_back({"price", lumina::FilterOp::kLt, 10.0});
    EXPECT_TRUE(fi.matches(1, lt));
    EXPECT_TRUE(fi.matches(2, lt));
    EXPECT_FALSE(fi.matches(3, lt));

    lumina::FilterExpr gt;
    gt.clauses.push_back({"price", lumina::FilterOp::kGe, 9.5});
    EXPECT_TRUE(fi.matches(1, gt));
    EXPECT_FALSE(fi.matches(2, gt));
    EXPECT_TRUE(fi.matches(3, gt));

    // AND semantics.
    lumina::FilterExpr both;
    both.clauses.push_back({"cat", lumina::FilterOp::kEq, int64_t(42)});
    both.clauses.push_back({"price", lumina::FilterOp::kLt, 50.0});
    EXPECT_TRUE(fi.matches(1, both));
    EXPECT_FALSE(fi.matches(3, both));

    // Remove clears predicates.
    fi.remove(1);
    EXPECT_FALSE(fi.matches(1, eq));
    EXPECT_TRUE(fi.matches(3, eq));
}

TEST(FilterIndexTest, StringEq) {
    lumina::FilterIndex fi;
    fi.add(1, {{"region", std::string("north")}});
    fi.add(2, {{"region", std::string("south")}});

    lumina::FilterExpr eq;
    eq.clauses.push_back({"region", lumina::FilterOp::kEq, std::string("north")});
    EXPECT_TRUE(fi.matches(1, eq));
    EXPECT_FALSE(fi.matches(2, eq));
}

TEST(CollectionFilterTest, InFilterReturnsOnlyMatches) {
    const std::string dir = temp_dir("in_filter");
    constexpr size_t dim = 32;

    lumina::Collection col(dir, dim);
    ASSERT_TRUE(col.open().ok());

    std::mt19937 rng(11);
    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < 500; ++i) {
        vecs.push_back(random_vector(dim, rng));
        std::vector<lumina::ScalarField> scalars;
        scalars.push_back({"cat", int64_t(i % 5)});  // 5 buckets
        ASSERT_TRUE(col.add(i, vecs.back().data(), "p" + std::to_string(i), scalars).ok());
    }

    lumina::FilterExpr filter;
    filter.clauses.push_back({"cat", lumina::FilterOp::kEq, int64_t(0)});
    lumina::SearchOptions opts;
    opts.ef_search = 200;
    opts.filter_mode = lumina::FilterMode::kInFilter;

    const auto hits = col.search_filtered(vecs[5].data(), 10, filter, opts);
    ASSERT_EQ(hits.size(), 10U);
    for (const auto& h : hits) {
        EXPECT_EQ(h.id % 5, 0U) << "in-filter returned non-matching id";
    }
}

TEST(CollectionFilterTest, PostFilterRecoversWithLargerEf) {
    const std::string dir = temp_dir("post_filter");
    constexpr size_t dim = 32;

    lumina::Collection col(dir, dim);
    ASSERT_TRUE(col.open().ok());

    std::mt19937 rng(12);
    std::vector<std::vector<float>> vecs;
    for (uint64_t i = 0; i < 800; ++i) {
        vecs.push_back(random_vector(dim, rng));
        std::vector<lumina::ScalarField> scalars;
        scalars.push_back({"cat", int64_t(i % 10)});  // 10% selectivity
        ASSERT_TRUE(col.add(i, vecs.back().data(), "p", scalars).ok());
    }

    lumina::FilterExpr filter;
    filter.clauses.push_back({"cat", lumina::FilterOp::kEq, int64_t(3)});
    lumina::SearchOptions opts;
    opts.ef_search = 50;

    opts.filter_mode = lumina::FilterMode::kPostFilter;
    const auto post = col.search_filtered(vecs[0].data(), 10, filter, opts);
    ASSERT_EQ(post.size(), 10U);
    for (const auto& h : post) {
        EXPECT_EQ(h.id % 10, 3U);
    }

    // In-filter returns matches too; verify both find the same nearest matching
    // vector for a known query.
    opts.filter_mode = lumina::FilterMode::kInFilter;
    const auto in = col.search_filtered(vecs[0].data(), 10, filter, opts);
    ASSERT_EQ(in.size(), 10U);
    for (const auto& h : in) {
        EXPECT_EQ(h.id % 10, 3U);
    }
}

TEST(CollectionFilterTest, FilterPersistsThroughReopen) {
    const std::string dir = temp_dir("filter_reopen");
    constexpr size_t dim = 16;

    {
        lumina::Collection col(dir, dim);
        ASSERT_TRUE(col.open().ok());
        std::mt19937 rng(13);
        for (uint64_t i = 0; i < 100; ++i) {
            auto v = random_vector(dim, rng);
            std::vector<lumina::ScalarField> scalars;
            scalars.push_back({"cat", int64_t(i % 4)});
            ASSERT_TRUE(col.add(i, v.data(), "p", scalars).ok());
        }
    }
    {
        lumina::Collection col(dir, dim);
        ASSERT_TRUE(col.open().ok());
        EXPECT_EQ(col.size(), 100U);
        lumina::FilterExpr filter;
        filter.clauses.push_back({"cat", lumina::FilterOp::kEq, int64_t(1)});
        lumina::SearchOptions opts;
        opts.ef_search = 100;
        opts.filter_mode = lumina::FilterMode::kInFilter;
        auto q = std::vector<float>(dim, 0.1F);
        const auto hits = col.search_filtered(q.data(), 5, filter, opts);
        ASSERT_EQ(hits.size(), 5U);
        for (const auto& h : hits) {
            EXPECT_EQ(h.id % 4, 1U);
        }
    }
}
