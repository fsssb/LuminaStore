#include "lumina/storage/storage_engine.h"

#include <mutex>
#include <utility>

namespace lumina {
namespace {

Status maybe_fsync_after_append(const Options& o, size_t& appends_since, LogManager& log) {
    if (!o.sync_writes) {
        return Status::OK();
    }
    if (!o.group_commit) {
        return log.sync();
    }
    if (o.sync_every_n_appends == 0) {
        return Status::OK();
    }
    ++appends_since;
    if (appends_since < o.sync_every_n_appends) {
        return Status::OK();
    }
    appends_since = 0;
    return log.sync();
}

}  // namespace

StorageEngine::StorageEngine(Options opts)
    : opts_(std::move(opts)), log_(opts_.wal_path) {}

StorageEngine::~StorageEngine() = default;

Status StorageEngine::open() {
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        appends_since_group_sync_ = 0;
        const Status s = log_.open();
        if (!s.ok()) {
            return s;
        }
    }
    return recover();
}

Status StorageEngine::put(const Slice& key, const Slice& value) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    uint64_t offset = 0;
    Status s = log_.append(OpType::kPut, key, value, &offset);
    if (!s.ok()) {
        return s;
    }

    index_.put(key.ToString(), offset);
    s = maybe_fsync_after_append(opts_, appends_since_group_sync_, log_);
    if (!s.ok()) {
        return s;
    }
    return Status::OK();
}

Status StorageEngine::put_vector(const Slice& key, const Slice& value) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    uint64_t offset = 0;
    Status s = log_.append(OpType::kVectorPut, key, value, &offset);
    if (!s.ok()) {
        return s;
    }

    index_.put(key.ToString(), offset);
    s = maybe_fsync_after_append(opts_, appends_since_group_sync_, log_);
    if (!s.ok()) {
        return s;
    }
    return Status::OK();
}

Status StorageEngine::get(const Slice& key, std::string* out_value) const {
    if (!out_value) {
        return Status::InvalidArgument("out_value is null");
    }

    std::shared_lock<std::shared_mutex> lock(mutex_);
    const auto offset = index_.get(key.ToString());
    if (!offset.has_value()) {
        return Status::NotFound("key not found");
    }

    std::string wal_key;
    std::string wal_value;
    Status s = log_.read_value_at(*offset, &wal_key, &wal_value);
    if (!s.ok()) {
        return s;
    }

    if (wal_key != key.view()) {
        return Status::Corruption("index points to mismatched key");
    }

    *out_value = std::move(wal_value);
    return Status::OK();
}

Status StorageEngine::remove(const Slice& key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    Status s = log_.append(OpType::kDelete, key, Slice{}, nullptr);
    if (!s.ok()) {
        return s;
    }

    index_.remove(key.ToString());
    s = maybe_fsync_after_append(opts_, appends_since_group_sync_, log_);
    if (!s.ok()) {
        return s;
    }
    return Status::OK();
}

Status StorageEngine::recover() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    index_.clear();

    return log_.iterate([this](const WalEntry& entry) {
        if (entry.op_type == OpType::kDelete) {
            index_.remove(entry.key);
        } else {
            index_.put(entry.key, entry.offset);
        }
        return true;
    });
}

Status StorageEngine::sync() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    appends_since_group_sync_ = 0;
    return log_.sync();
}

size_t StorageEngine::key_count() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return index_.size();
}

}  // namespace lumina
