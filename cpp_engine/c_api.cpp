#include "lumina/engine/collection.h"
#include "lumina/engine/filter.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Handle-based C API v2 (replaces the v1 global-singleton API).
//
//   void* h = lumina_open("/data/dir", dim, metric);
//   lumina_add(h, id, vec, dim, payload);
//   char* json = lumina_search(h, query, dim, top_k);
//   ...
//   lumina_close(h);
//
// metric: 0=L2, 1=IP, 2=Cosine.

namespace {

struct Handle {
    std::unique_ptr<lumina::Collection> collection;
    std::mutex mutex;  // serializes search/read so JSON buffers stay consistent
};

std::unordered_map<void*, std::unique_ptr<Handle>> g_handles;
std::mutex g_handles_mutex;

Handle* lookup(void* h) {
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    const auto it = g_handles.find(h);
    return (it == g_handles.end()) ? nullptr : it->second.get();
}

char* alloc_c_string(const std::string& text) {
    char* buffer = static_cast<char*>(std::malloc(text.size() + 1));
    if (buffer == nullptr) {
        return nullptr;
    }
    std::memcpy(buffer, text.c_str(), text.size() + 1);
    return buffer;
}

std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (const char c : input) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(c);
                break;
        }
    }
    return out;
}

lumina::Metric metric_from_int(int metric) {
    switch (metric) {
        case 1:
            return lumina::Metric::kIP;
        case 2:
            return lumina::Metric::kCosine;
        default:
            return lumina::Metric::kL2;
    }
}

}  // namespace

extern "C" {

// ---- lifecycle ----

void* lumina_open(const char* dir, int dim, int metric) {
    if (dir == nullptr || dim <= 0) {
        return nullptr;
    }
    auto handle = std::make_unique<Handle>();
    handle->collection = std::make_unique<lumina::Collection>(dir, static_cast<size_t>(dim),
                                                              metric_from_int(metric));
    const lumina::Status s = handle->collection->open();
    if (!s.ok()) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_handles_mutex);
    void* key = handle.get();
    g_handles.emplace(key, std::move(handle));
    return key;
}

int lumina_close(void* h) {
    std::lock_guard<std::mutex> lock(g_handles_mutex);
    return g_handles.erase(h) == 1U ? 0 : -1;
}

// ---- writes ----

int lumina_add(void* h, uint64_t id, const float* vec, int dim, const char* payload) {
    Handle* handle = lookup(h);
    if (handle == nullptr || vec == nullptr || dim <= 0 || payload == nullptr) {
        return -1;
    }
    return handle->collection->add(id, vec, payload).ok() ? 0 : -2;
}

// Batch add: `vectors` holds n vectors of `dim` floats each (row-major).
int lumina_add_batch(void* h, const uint64_t* ids, const float* vectors, int n, int dim,
                     const char* const* payloads) {
    Handle* handle = lookup(h);
    if (handle == nullptr || ids == nullptr || vectors == nullptr || n <= 0 || dim <= 0 ||
        payloads == nullptr) {
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        const lumina::Status s = handle->collection->add(ids[i], vectors + static_cast<size_t>(i) * dim,
                                                         payloads[i] != nullptr ? payloads[i] : "");
        if (!s.ok()) {
            return -2;  // stop at first failure
        }
    }
    return 0;
}

int lumina_remove(void* h, uint64_t id) {
    Handle* handle = lookup(h);
    if (handle == nullptr) {
        return -1;
    }
    return handle->collection->remove(id).ok() ? 0 : -2;
}

// ---- reads ----

char* lumina_search(void* h, const float* query, int dim, int top_k) {
    Handle* handle = lookup(h);
    if (handle == nullptr || query == nullptr || dim <= 0) {
        return alloc_c_string("{\"error\":\"bad args\",\"results\":[]}");
    }
    std::lock_guard<std::mutex> lock(handle->mutex);

    const size_t k = static_cast<size_t>(std::max(1, top_k));
    const auto hits = handle->collection->search(query, k, {.ef_search = 200});

    std::string json = "{\"results\":[";
    bool first = true;
    for (const auto& hit : hits) {
        std::string payload;
        handle->collection->get(hit.id, &payload);  // best-effort
        if (!first) {
            json += ",";
        }
        first = false;
        json += "{\"id\":" + std::to_string(hit.id) +
                ",\"distance\":" + std::to_string(hit.distance) +
                ",\"payload\":\"" + json_escape(payload) + "\"}";
    }
    json += "]}";
    return alloc_c_string(json);
}

int lumina_get(void* h, uint64_t id, char** payload, int* out_len) {
    Handle* handle = lookup(h);
    if (handle == nullptr || payload == nullptr) {
        return -1;
    }
    std::string text;
    const lumina::Status s = handle->collection->get(id, &text);
    if (!s.ok()) {
        return -2;
    }
    char* copy = alloc_c_string(text);
    if (copy == nullptr) {
        return -3;
    }
    *payload = copy;
    if (out_len != nullptr) {
        *out_len = static_cast<int>(text.size());
    }
    return 0;
}

// Batch search: `queries` holds n vectors of `dim` floats each (row-major).
// Returns a JSON document: {"results":[[{...},{...}], ...]}.
char* lumina_search_batch(void* h, const float* queries, int n, int dim, int top_k) {
    Handle* handle = lookup(h);
    if (handle == nullptr || queries == nullptr || n <= 0 || dim <= 0) {
        return alloc_c_string("{\"error\":\"bad args\",\"results\":[]}");
    }
    std::lock_guard<std::mutex> lock(handle->mutex);

    const size_t k = static_cast<size_t>(std::max(1, top_k));
    std::string json = "{\"results\":[";
    for (int i = 0; i < n; ++i) {
        const float* q = queries + static_cast<size_t>(i) * dim;
        const auto hits = handle->collection->search(q, k, {.ef_search = 200});
        if (i > 0) {
            json += ",";
        }
        json += "[";
        bool first = true;
        for (const auto& hit : hits) {
            std::string payload;
            handle->collection->get(hit.id, &payload);  // best-effort
            if (!first) {
                json += ",";
            }
            first = false;
            json += "{\"id\":" + std::to_string(hit.id) +
                    ",\"distance\":" + std::to_string(hit.distance) +
                    ",\"payload\":\"" + json_escape(payload) + "\"}";
        }
        json += "]";
    }
    json += "]}";
    return alloc_c_string(json);
}

// ---- maintenance ----

int lumina_snapshot(void* h) {
    Handle* handle = lookup(h);
    if (handle == nullptr) {
        return -1;
    }
    return handle->collection->snapshot().ok() ? 0 : -2;
}

char* lumina_stats(void* h) {
    Handle* handle = lookup(h);
    if (handle == nullptr) {
        return alloc_c_string("{\"error\":\"invalid handle\"}");
    }
    const auto stats = handle->collection->stats();
    std::string json = "{\"live_entries\":" + std::to_string(stats.live_entries) +
                       ",\"wal_bytes\":" + std::to_string(stats.wal_bytes) + "}";
    return alloc_c_string(json);
}

void lumina_free_string(char* ptr) {
    std::free(ptr);
}

}  // extern "C"
