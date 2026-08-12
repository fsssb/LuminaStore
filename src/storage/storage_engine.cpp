#include "lumina/storage/storage_engine.h"

#include "lumina/storage/manifest.h"
#include "lumina/storage/snapshot.h"

#include <cerrno>
#include <cstring>
#include <ctime>
#include <mutex>
#include <sys/stat.h>
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

void apply_wal_entry(IndexManager& index, const WalEntry& entry) {
    if (entry.op_type == OpType::kDelete) {
        index.remove(entry.key);
    } else {
        index.put(entry.key, entry.offset);
    }
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

        // Fast path: load the latest snapshot, then replay only the WAL tail
        // after the snapshot watermark. Falls back to a full replay when the
        // manifest or snapshot is missing/corrupt.
        const std::string manifest_path = opts_.snapshot_dir + "/MANIFEST";
        ManifestEntry entry;
        Status ms = read_manifest_latest(manifest_path, &entry);
        if (ms.ok()) {
            SnapshotMeta meta;
            const Status ss = read_snapshot(opts_.snapshot_dir + "/" + entry.filename, &meta);
            if (ss.ok() && meta.wal_offset <= log_.size()) {
                index_.load(std::move(meta.index));
                return log_.iterate_from(meta.wal_offset,
                                         [this](const WalEntry& e) {
                                             apply_wal_entry(index_, e);
                                             return true;
                                         });
            }
            // fall through to full replay
        }
        return recover_inner();
    }
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
    return recover_inner();
}

Status StorageEngine::recover_inner() {
    index_.clear();
    return log_.iterate([this](const WalEntry& entry) {
        apply_wal_entry(index_, entry);
        return true;
    });
}

Status StorageEngine::snapshot() {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // The watermark must cover only durable WAL bytes; fsync first.
    Status s = log_.sync();
    if (!s.ok()) {
        return s;
    }

    if (::mkdir(opts_.snapshot_dir.c_str(), 0755) != 0 && errno != EEXIST) {
        return Status::IOError("mkdir snapshot dir: " + opts_.snapshot_dir + " (" +
                               std::strerror(errno) + ")");
    }

    const uint64_t wal_offset = log_.size();
    const uint64_t seq = static_cast<uint64_t>(std::time(nullptr)) * 1000 + (snap_seq_counter_++ % 1000);
    const std::string filename = "snap-" + std::to_string(seq) + ".snap";
    const std::string tmp_path = opts_.snapshot_dir + "/" + filename + ".tmp";
    const std::string final_path = opts_.snapshot_dir + "/" + filename;

    s = write_snapshot(tmp_path, wal_offset, index_.dump());
    if (!s.ok()) {
        return s;
    }
    if (::rename(tmp_path.c_str(), final_path.c_str()) != 0) {
        return Status::IOError("rename snapshot (" + std::string(std::strerror(errno)) + ")");
    }
    return write_manifest_append(opts_.snapshot_dir + "/MANIFEST",
                                 ManifestEntry{seq, wal_offset, filename});
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
