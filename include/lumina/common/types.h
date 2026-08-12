#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

namespace lumina {

// ---- Status ----

enum class StatusCode : uint8_t {
    kOk = 0,
    kNotFound,
    kIOError,
    kCorruption,
    kInvalidArgument,
    kAlreadyExists,
};

class Status {
public:
    Status() : code_(StatusCode::kOk) {}
    explicit Status(StatusCode code, std::string msg = {})
        : code_(code), msg_(std::move(msg)) {}

    static Status OK() { return Status{}; }
    static Status NotFound(std::string msg = {}) {
        return Status{StatusCode::kNotFound, std::move(msg)};
    }
    static Status IOError(std::string msg = {}) {
        return Status{StatusCode::kIOError, std::move(msg)};
    }
    static Status Corruption(std::string msg = {}) {
        return Status{StatusCode::kCorruption, std::move(msg)};
    }
    static Status InvalidArgument(std::string msg = {}) {
        return Status{StatusCode::kInvalidArgument, std::move(msg)};
    }

    bool ok()          const { return code_ == StatusCode::kOk; }
    bool IsNotFound()  const { return code_ == StatusCode::kNotFound; }
    bool IsIOError()   const { return code_ == StatusCode::kIOError; }
    bool IsCorruption() const { return code_ == StatusCode::kCorruption; }

    StatusCode code() const { return code_; }
    const std::string& message() const { return msg_; }

    std::string ToString() const {
        switch (code_) {
            case StatusCode::kOk:              return "OK";
            case StatusCode::kNotFound:        return "NotFound: " + msg_;
            case StatusCode::kIOError:         return "IOError: " + msg_;
            case StatusCode::kCorruption:      return "Corruption: " + msg_;
            case StatusCode::kInvalidArgument: return "InvalidArgument: " + msg_;
            case StatusCode::kAlreadyExists:   return "AlreadyExists: " + msg_;
        }
        return "Unknown";
    }

private:
    StatusCode  code_;
    std::string msg_;
};

// ---- Slice ----
// Zero-copy string view wrapper with convenience constructors.

class Slice {
public:
    Slice() = default;
    Slice(const char* data, size_t size) : view_(data, size) {}
    Slice(std::string_view sv) : view_(sv) {}           // NOLINT(google-explicit-constructor)
    Slice(const std::string& s) : view_(s) {}           // NOLINT(google-explicit-constructor)
    Slice(const char* s) : view_(s) {}                  // NOLINT(google-explicit-constructor)

    const char* data()  const { return view_.data(); }
    size_t      size()  const { return view_.size(); }
    bool        empty() const { return view_.empty(); }

    std::string_view view()   const { return view_; }
    std::string      ToString() const { return std::string(view_); }

    bool operator==(const Slice& o) const { return view_ == o.view_; }
    bool operator!=(const Slice& o) const { return view_ != o.view_; }
    bool operator< (const Slice& o) const { return view_ <  o.view_; }

private:
    std::string_view view_;
};

// ---- Distance metric ----

enum class Metric : uint8_t {
    kL2     = 0,
    kIP     = 1,
    kCosine = 2,
};

// ---- Scalar filter fields ----
// 过滤字段值：int64 / double / string 三态。

using ScalarValue = std::variant<int64_t, double, std::string>;

struct ScalarField {
    std::string  name;
    ScalarValue value;
};

// ---- Quantization config ----

struct QuantConfig {
    enum class Mode : uint8_t {
        kNone   = 0,
        kSQ8    = 1,
        kBinary = 2,
        kPQ     = 3,
    };

    Mode   mode            = Mode::kNone;
    size_t pq_subspaces    = 16;    // PQ 子空间数 m（仅 kPQ 生效）
    size_t pq_centroids    = 256;   // 每子空间质心数 k（仅 kPQ 生效，<= 256 用 8bit 编码）

    bool enabled() const { return mode != Mode::kNone; }
};

// ---- Options ----

struct Options {
    // WAL file path
    std::string wal_path = "lumina.wal";

    // HNSW graph file path (for persistence)
    std::string hnsw_path = "lumina.hnsw";

    // When false, engine never fsyncs (caller must use sync()).
    // When true and group_commit is false, fsync after each mutating op (if sync writes enabled).
    bool sync_writes = true;

    // If true, do not fsync on each put/remove unless:
    //   - sync_every_n_appends > 0 and the counter reaches N, or
    //   - StorageEngine::sync() is called explicitly.
    bool group_commit = false;

    // When group_commit and value > 0, fsync every N appends. When 0, only manual sync().
    size_t sync_every_n_appends = 0;

    // Vector dimensionality
    size_t vector_dim = 128;

    // HNSW parameters
    size_t hnsw_M              = 16;
    size_t hnsw_ef_construction = 200;
    size_t hnsw_ef_search       = 50;

    // ---- v2 additions ----
    Metric metric = Metric::kL2;

    // Snapshot directory (manifest + .snap files live here)
    std::string snapshot_dir = "lumina_snap";

    // Trigger a snapshot after this many new WAL bytes since the last snapshot.
    size_t snapshot_interval_bytes = 256ULL * 1024 * 1024;

    QuantConfig quant;
};

// ---- WAL OpType ----

enum class OpType : uint8_t {
    kPut          = 0x01,
    kDelete       = 0x02,
    kVectorPut    = 0x03,
    kVectorPutV2  = 0x04,   // v2: value = encoded EntryMeta (vec + payload + scalars)
};

} // namespace lumina
