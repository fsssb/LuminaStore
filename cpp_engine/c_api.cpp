#include "lumina/vector/hnsw_index.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct LuminaBridgeState {
    size_t dim = 0;
    std::unique_ptr<lumina::HNSWIndex> index;
    std::unordered_map<uint64_t, std::string> payloads;
    std::mutex mutex;
};

LuminaBridgeState g_state;

std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char c : input) {
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

char* alloc_c_string(const std::string& text) {
    char* buffer = static_cast<char*>(std::malloc(text.size() + 1));
    if (!buffer) {
        return nullptr;
    }
    std::memcpy(buffer, text.c_str(), text.size() + 1);
    return buffer;
}

}  // namespace

extern "C" {

int lumina_init(int dim) {
    try {
        if (dim <= 0) {
            std::cerr << "[Lumina C API] 初始化失败：维度必须大于 0，收到 dim=" << dim << std::endl;
            return -1;
        }

        std::lock_guard<std::mutex> lock(g_state.mutex);
        g_state.dim = static_cast<size_t>(dim);
        g_state.index = std::make_unique<lumina::HNSWIndex>(g_state.dim);
        g_state.payloads.clear();
        std::cout << "[Lumina C API] 初始化成功，向量维度 dim=" << dim << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Lumina C API] 初始化异常：" << e.what() << std::endl;
        return -2;
    } catch (...) {
        std::cerr << "[Lumina C API] 初始化未知异常" << std::endl;
        return -3;
    }
}

int lumina_add_vector(uint64_t id, const float* vector, int dim, const char* payload_text) {
    try {
        if (!vector || !payload_text) {
            std::cerr << "[Lumina C API] 插入失败：vector/payload 为空指针" << std::endl;
            return -1;
        }

        std::lock_guard<std::mutex> lock(g_state.mutex);
        if (!g_state.index) {
            std::cerr << "[Lumina C API] 插入失败：请先调用 lumina_init" << std::endl;
            return -2;
        }
        if (dim <= 0 || static_cast<size_t>(dim) != g_state.dim) {
            std::cerr << "[Lumina C API] 插入失败：维度不匹配，expect=" << g_state.dim
                      << " got=" << dim << std::endl;
            return -3;
        }

        const lumina::Status status = g_state.index->add_item(id, vector);
        if (!status.ok()) {
            std::cerr << "[Lumina C API] 插入失败：HNSW add_item 错误，id=" << id
                      << " status=" << status.ToString() << std::endl;
            return -4;
        }

        g_state.payloads[id] = payload_text;
        std::cout << "[Lumina C API] 插入成功，id=" << id << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Lumina C API] 插入异常：" << e.what() << std::endl;
        return -5;
    } catch (...) {
        std::cerr << "[Lumina C API] 插入未知异常" << std::endl;
        return -6;
    }
}

char* lumina_search_vector(const float* query_vector, int dim, int top_k) {
    try {
        if (!query_vector) {
            std::cerr << "[Lumina C API] 检索失败：query_vector 为空指针" << std::endl;
            return alloc_c_string("{\"error\":\"query_vector is null\",\"results\":[]}");
        }

        std::lock_guard<std::mutex> lock(g_state.mutex);
        if (!g_state.index) {
            std::cerr << "[Lumina C API] 检索失败：请先调用 lumina_init" << std::endl;
            return alloc_c_string("{\"error\":\"engine not initialized\",\"results\":[]}");
        }
        if (dim <= 0 || static_cast<size_t>(dim) != g_state.dim) {
            std::cerr << "[Lumina C API] 检索失败：维度不匹配，expect=" << g_state.dim
                      << " got=" << dim << std::endl;
            return alloc_c_string("{\"error\":\"dimension mismatch\",\"results\":[]}");
        }

        const size_t k = std::max(1, top_k);
        const auto hits = g_state.index->search_top_k(query_vector, k);

        std::string json = "{\"results\":[";
        bool first = true;
        for (const auto& hit : hits) {
            const auto it = g_state.payloads.find(hit.id);
            const std::string payload = (it == g_state.payloads.end()) ? "" : it->second;
            if (!first) {
                json += ",";
            }
            first = false;
            json += "{\"id\":";
            json += std::to_string(hit.id);
            json += ",\"distance\":";
            json += std::to_string(hit.distance);
            json += ",\"payload\":\"";
            json += json_escape(payload);
            json += "\"}";
        }
        json += "]}";
        std::cout << "[Lumina C API] 检索完成，top_k=" << k << " 命中条数=" << hits.size() << std::endl;
        return alloc_c_string(json);
    } catch (const std::exception& e) {
        std::cerr << "[Lumina C API] 检索异常：" << e.what() << std::endl;
        return alloc_c_string("{\"error\":\"search exception\",\"results\":[]}");
    } catch (...) {
        std::cerr << "[Lumina C API] 检索未知异常" << std::endl;
        return alloc_c_string("{\"error\":\"unknown exception\",\"results\":[]}");
    }
}

void lumina_free_string(char* ptr) {
    if (!ptr) {
        return;
    }
    std::free(ptr);
}

}  // extern "C"
