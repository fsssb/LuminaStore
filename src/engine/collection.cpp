#include "lumina/engine/collection.h"

#include "lumina/engine/filter.h"
#include "lumina/engine/pipeline.h"
#include "lumina/engine/query.h"
#include "lumina/storage/entry_meta.h"
#include "lumina/storage/storage_engine.h"

#include <filesystem>
#include <mutex>
#include <string>
#include <utility>

namespace lumina {
namespace {

std::string key_of(uint64_t id) {
    return std::to_string(id);
}

bool parse_id(const std::string& key, uint64_t* out) {
    if (key.empty()) {
        return false;
    }
    size_t consumed = 0;
    try {
        *out = std::stoull(key, &consumed);
    } catch (...) {
        return false;
    }
    return consumed == key.size();
}

}  // namespace

struct Collection::Impl {
    Options opts;
    std::unique_ptr<StorageEngine> storage;  // constructed after opts is configured
    HNSWIndex index;
    FilterIndex filter_index;
    std::mutex write_mutex;  // serializes add/remove/update (WAL + graph consistency)

    Impl(Options o, DistanceFn dist, size_t dim)
        : opts(std::move(o)), index(dim, opts.hnsw_M, opts.hnsw_ef_construction, dist) {}
};
Collection::Collection(std::string dir, size_t dim, Metric metric, size_t M, size_t ef_construction)
    : impl_(new Impl(Options{}, pipeline::distance_for_metric(Metric::kL2), dim)), dim_(dim),
      metric_(metric) {
    // Ensure the data directory exists (like a mini mkdir -p).
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    auto& o = impl_->opts;
    o.wal_path = dir + "/data.wal";
    o.snapshot_dir = dir + "/snap";
    o.vector_dim = dim;
    o.hnsw_M = M;
    o.hnsw_ef_construction = ef_construction;
    impl_->storage = std::make_unique<StorageEngine>(impl_->opts);
    impl_->index = HNSWIndex(dim, M, ef_construction, pipeline::distance_for_metric(metric));
}

Collection::~Collection() = default;

Status Collection::open() {
    Status s = impl_->storage->open();
    if (!s.ok()) {
        return s;
    }
    // Rebuild the HNSW graph and filter index from live WAL entries.
    return impl_->storage->visit_live([this](const std::string& key, const std::string& value) {
        uint64_t id = 0;
        if (!parse_id(key, &id)) {
            return false;  // abort on unexpected key format
        }
        EntryMeta meta;
        const Status s = decode_entry_meta(value, &meta);
        if (!s.ok()) {
            return false;
        }
        if (meta.vec.size() != dim_) {
            return false;
        }
        if (!impl_->index.add_item(id, meta.vec.data()).ok()) {
            return false;
        }
        impl_->filter_index.add(id, meta.scalars);
        return true;
    });
}

Status Collection::add(uint64_t id, const float* vec, const std::string& payload,
                       const std::vector<ScalarField>& scalars) {
    if (vec == nullptr) {
        return Status::InvalidArgument("vec is null");
    }
    std::lock_guard<std::mutex> lock(impl_->write_mutex);

    if (impl_->index.contains(id)) {
        return Status::InvalidArgument("duplicate id");
    }

    EntryMeta meta;
    meta.vec.assign(vec, vec + dim_);
    meta.payload = payload;
    meta.scalars = scalars;
    std::string bytes;
    Status s = encode_entry_meta(meta, &bytes);
    if (!s.ok()) {
        return s;
    }
    s = impl_->storage->put_vector_v2(key_of(id), bytes);
    if (!s.ok()) {
        return s;
    }
    const Status is = impl_->index.add_item(id, vec);
    if (!is.ok()) {
        return is;
    }
    impl_->filter_index.add(id, scalars);
    return Status::OK();
}

Status Collection::remove(uint64_t id) {
    std::lock_guard<std::mutex> lock(impl_->write_mutex);
    Status s = impl_->storage->remove(key_of(id));
    if (!s.ok()) {
        return s;
    }
    const Status rs = impl_->index.remove(id);  // NotFound if absent
    impl_->filter_index.remove(id);
    return rs;
}

Status Collection::update(uint64_t id, const float* vec, const std::string& payload,
                          const std::vector<ScalarField>& scalars) {
    if (vec == nullptr) {
        return Status::InvalidArgument("vec is null");
    }
    std::lock_guard<std::mutex> lock(impl_->write_mutex);

    if (!impl_->index.contains(id)) {
        return Status::NotFound("id not found");
    }

    EntryMeta meta;
    meta.vec.assign(vec, vec + dim_);
    meta.payload = payload;
    meta.scalars = scalars;
    std::string bytes;
    Status s = encode_entry_meta(meta, &bytes);
    if (!s.ok()) {
        return s;
    }
    s = impl_->storage->put_vector_v2(key_of(id), bytes);
    if (!s.ok()) {
        return s;
    }
    const Status us = impl_->index.update_item(id, vec);
    if (!us.ok()) {
        return us;
    }
    impl_->filter_index.remove(id);
    impl_->filter_index.add(id, scalars);
    return Status::OK();
}

Status Collection::get(uint64_t id, std::string* payload) const {
    if (payload == nullptr) {
        return Status::InvalidArgument("payload is null");
    }
    std::string bytes;
    const Status s = impl_->storage->get(key_of(id), &bytes);
    if (!s.ok()) {
        return s;
    }
    EntryMeta meta;
    const Status d = decode_entry_meta(bytes, &meta);
    if (!d.ok()) {
        return d;
    }
    *payload = std::move(meta.payload);
    return Status::OK();
}

std::vector<SearchResult> Collection::search(const float* query, size_t k,
                                             const SearchOptions& opts) const {
    if (query == nullptr || k == 0) {
        return {};
    }
    return impl_->index.search_top_k(query, k, opts.ef_search);
}

std::vector<SearchResult> Collection::search_filtered(const float* query, size_t k,
                                                      const FilterExpr& filter,
                                                      const SearchOptions& opts) const {
    if (query == nullptr || k == 0) {
        return {};
    }
    if (filter.empty()) {
        return search(query, k, opts);
    }

    if (opts.filter_mode == FilterMode::kInFilter) {
        const auto fn = [this, &filter](uint64_t id) {
            return impl_->filter_index.matches(id, filter);
        };
        return impl_->index.search_top_k_filtered(query, k, opts.ef_search, fn);
    }

    // Post-filter: plain search with oversampled ef, filter, retry with a
    // larger ef until enough results survive (or the budget is exhausted).
    size_t ef = std::max(opts.ef_search, k * 2);
    for (int attempt = 0; attempt < 4; ++attempt) {
        auto hits = impl_->index.search_top_k(query, ef, ef);
        std::vector<SearchResult> filtered;
        filtered.reserve(hits.size());
        for (const auto& h : hits) {
            if (impl_->filter_index.matches(h.id, filter)) {
                filtered.push_back(h);
            }
        }
        if (filtered.size() >= k || attempt == 3) {
            if (filtered.size() > k) {
                filtered.resize(k);
            }
            return filtered;
        }
        ef *= 2;
    }
    return {};
}

Status Collection::snapshot() {
    return impl_->storage->snapshot();
}

size_t Collection::size() const {
    return impl_->index.size();
}

Collection::Stats Collection::stats() const {
    Stats s;
    s.live_entries = impl_->index.size();
    s.wal_bytes = impl_->storage->wal_size();
    return s;
}

} // namespace lumina
