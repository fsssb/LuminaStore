#include "lumina/vector/hnsw_index.h"

#include "lumina/vector/aligned_alloc.h"
#include "lumina/vector/vector_math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace lumina {

namespace {

struct Node {
    uint64_t id = 0;
    AlignedFloatVector vec;
    std::vector<std::vector<size_t>> neighbors;
    int max_layer = 0;
};

struct InternalImpl {
    size_t dim = 0;
    size_t M = 16;
    size_t ef_construction = 200;
    int entry_point = -1;
    int max_layer = -1;
    std::vector<Node> nodes;
    std::unordered_map<uint64_t, size_t> id_to_idx;
    std::mt19937 rng{42};
    double ml = 1.0;
};

using Candidate = std::pair<float, size_t>;

struct CandidateMinCompare {
    bool operator()(const Candidate& a, const Candidate& b) const { return a.first > b.first; }
};

int random_level(InternalImpl& impl) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double x = std::max(dist(impl.rng), 1e-12);
    return static_cast<int>(-std::log(x) * impl.ml);
}

std::vector<Candidate> search_layer(const InternalImpl& impl,
                                    const float* query,
                                    size_t entry_idx,
                                    size_t ef,
                                    int layer) {
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateMinCompare> candidates;
    std::priority_queue<Candidate> result;
    std::unordered_set<size_t> visited;

    const float d = VectorMath::l2_distance(query, impl.nodes[entry_idx].vec.data(), impl.dim);
    candidates.push({d, entry_idx});
    result.push({d, entry_idx});
    visited.insert(entry_idx);

    while (!candidates.empty()) {
        const auto [cand_dist, cand_idx] = candidates.top();
        candidates.pop();

        if (!result.empty() && cand_dist > result.top().first) {
            break;
        }

        if (layer > impl.nodes[cand_idx].max_layer) {
            continue;
        }

        for (size_t nb_idx : impl.nodes[cand_idx].neighbors[layer]) {
            if (visited.count(nb_idx) != 0U) {
                continue;
            }
            visited.insert(nb_idx);

            const float nd = VectorMath::l2_distance(query, impl.nodes[nb_idx].vec.data(), impl.dim);
            if (result.size() < ef || nd < result.top().first) {
                candidates.push({nd, nb_idx});
                result.push({nd, nb_idx});
                if (result.size() > ef) {
                    result.pop();
                }
            }
        }
    }

    std::vector<Candidate> out;
    out.reserve(result.size());
    while (!result.empty()) {
        out.push_back(result.top());
        result.pop();
    }
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
        return a.first < b.first;
    });
    return out;
}

void connect_bidirectional(InternalImpl& impl, size_t a, size_t b, int layer, size_t max_m) {
    auto& na = impl.nodes[a].neighbors[layer];
    auto& nb = impl.nodes[b].neighbors[layer];

    if (std::find(na.begin(), na.end(), b) == na.end()) {
        na.push_back(b);
    }
    if (std::find(nb.begin(), nb.end(), a) == nb.end()) {
        nb.push_back(a);
    }

    const auto prune = [&](size_t center_idx, std::vector<size_t>& conn) {
        if (conn.size() <= max_m) {
            return;
        }
        std::sort(conn.begin(), conn.end(), [&](size_t x, size_t y) {
            const float dx = VectorMath::l2_distance(impl.nodes[center_idx].vec.data(),
                                                     impl.nodes[x].vec.data(), impl.dim);
            const float dy = VectorMath::l2_distance(impl.nodes[center_idx].vec.data(),
                                                     impl.nodes[y].vec.data(), impl.dim);
            return dx < dy;
        });
        conn.resize(max_m);
    };

    prune(a, na);
    prune(b, nb);
}

}  // namespace

struct HNSWIndex::Impl {
    InternalImpl data;
};

HNSWIndex::HNSWIndex(size_t dim, size_t M, size_t ef_construction)
    : impl_(std::make_unique<Impl>()), dim_(dim) {
    auto& d = impl_->data;
    d.dim = dim;
    d.M = M;
    d.ef_construction = ef_construction;
    d.ml = 1.0 / std::log(static_cast<double>(std::max<size_t>(M, 2)));
}

HNSWIndex::~HNSWIndex() = default;

Status HNSWIndex::add_item(uint64_t id, const float* vec) {
    auto& d = impl_->data;
    if (vec == nullptr) {
        return Status::InvalidArgument("vector is null");
    }
    if (d.id_to_idx.count(id) != 0U) {
        return Status::InvalidArgument("duplicate id");
    }

    const size_t new_idx = d.nodes.size();
    const int new_layer = random_level(d);

    Node node;
    node.id = id;
    node.vec.assign(vec, vec + d.dim);
    node.max_layer = new_layer;
    node.neighbors.resize(static_cast<size_t>(new_layer + 1));
    for (int l = 0; l <= new_layer; ++l) {
        node.neighbors[l].reserve(l == 0 ? 2 * d.M : d.M);
    }

    d.nodes.push_back(std::move(node));
    d.id_to_idx[id] = new_idx;

    if (d.entry_point < 0) {
        d.entry_point = static_cast<int>(new_idx);
        d.max_layer = new_layer;
        return Status::OK();
    }

    size_t ep = static_cast<size_t>(d.entry_point);

    for (int l = d.max_layer; l > new_layer; --l) {
        auto res = search_layer(d, vec, ep, 1, l);
        if (!res.empty()) {
            ep = res.front().second;
        }
    }

    for (int l = std::min(new_layer, d.max_layer); l >= 0; --l) {
        auto neighbors = search_layer(d, vec, ep, d.ef_construction, l);

        const size_t max_m = (l == 0) ? 2 * d.M : d.M;
        if (neighbors.size() > max_m) {
            neighbors.resize(max_m);
        }

        if (!neighbors.empty()) {
            ep = neighbors.front().second;
        }

        for (const auto& item : neighbors) {
            connect_bidirectional(d, new_idx, item.second, l, max_m);
        }
    }

    if (new_layer > d.max_layer) {
        d.max_layer = new_layer;
        d.entry_point = static_cast<int>(new_idx);
    }

    return Status::OK();
}

std::vector<SearchResult> HNSWIndex::search_top_k(const float* query, size_t k, size_t ef_search) const {
    const auto& d = impl_->data;
    std::vector<SearchResult> out;
    if (query == nullptr || k == 0 || d.entry_point < 0) {
        return out;
    }

    size_t ep = static_cast<size_t>(d.entry_point);
    for (int l = d.max_layer; l > 0; --l) {
        auto res = search_layer(d, query, ep, 1, l);
        if (!res.empty()) {
            ep = res.front().second;
        }
    }

    auto candidates = search_layer(d, query, ep, std::max(ef_search, k), 0);
    if (candidates.size() > k) {
        candidates.resize(k);
    }

    out.reserve(candidates.size());
    for (const auto& item : candidates) {
        out.push_back({d.nodes[item.second].id, item.first});
    }
    return out;
}

size_t HNSWIndex::size() const {
    return impl_->data.nodes.size();
}

Status HNSWIndex::save(const std::string& path) const {
    const auto& d = impl_->data;
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return Status::IOError("failed to open hnsw file for write");
    }

    const uint64_t dim_u64 = static_cast<uint64_t>(d.dim);
    const uint64_t m_u64 = static_cast<uint64_t>(d.M);
    const int32_t max_layer_i32 = d.max_layer;
    const int32_t entry_point_i32 = d.entry_point;
    const uint64_t nodes_u64 = static_cast<uint64_t>(d.nodes.size());

    ofs.write(reinterpret_cast<const char*>(&dim_u64), sizeof(dim_u64));
    ofs.write(reinterpret_cast<const char*>(&m_u64), sizeof(m_u64));
    ofs.write(reinterpret_cast<const char*>(&max_layer_i32), sizeof(max_layer_i32));
    ofs.write(reinterpret_cast<const char*>(&entry_point_i32), sizeof(entry_point_i32));
    ofs.write(reinterpret_cast<const char*>(&nodes_u64), sizeof(nodes_u64));

    for (const auto& node : d.nodes) {
        const int32_t node_layer = node.max_layer;
        ofs.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
        ofs.write(reinterpret_cast<const char*>(&node_layer), sizeof(node_layer));
        ofs.write(reinterpret_cast<const char*>(node.vec.data()),
                  static_cast<std::streamsize>(node.vec.size() * sizeof(float)));

        for (int l = 0; l <= node.max_layer; ++l) {
            const uint32_t ncount = static_cast<uint32_t>(node.neighbors[l].size());
            ofs.write(reinterpret_cast<const char*>(&ncount), sizeof(ncount));
            for (size_t idx : node.neighbors[l]) {
                const uint64_t idx_u64 = static_cast<uint64_t>(idx);
                ofs.write(reinterpret_cast<const char*>(&idx_u64), sizeof(idx_u64));
            }
        }
    }

    if (!ofs.good()) {
        return Status::IOError("failed to write hnsw file");
    }
    return Status::OK();
}

Status HNSWIndex::load(const std::string& path) {
    auto& d = impl_->data;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return Status::IOError("failed to open hnsw file for read");
    }

    uint64_t dim_u64 = 0;
    uint64_t m_u64 = 0;
    int32_t max_layer_i32 = -1;
    int32_t entry_point_i32 = -1;
    uint64_t nodes_u64 = 0;

    ifs.read(reinterpret_cast<char*>(&dim_u64), sizeof(dim_u64));
    ifs.read(reinterpret_cast<char*>(&m_u64), sizeof(m_u64));
    ifs.read(reinterpret_cast<char*>(&max_layer_i32), sizeof(max_layer_i32));
    ifs.read(reinterpret_cast<char*>(&entry_point_i32), sizeof(entry_point_i32));
    ifs.read(reinterpret_cast<char*>(&nodes_u64), sizeof(nodes_u64));

    if (!ifs.good()) {
        return Status::Corruption("invalid hnsw header");
    }
    if (dim_u64 != d.dim) {
        return Status::InvalidArgument("dimension mismatch while loading hnsw");
    }

    d.M = static_cast<size_t>(m_u64);
    d.max_layer = max_layer_i32;
    d.entry_point = entry_point_i32;
    d.nodes.clear();
    d.id_to_idx.clear();
    d.nodes.reserve(static_cast<size_t>(nodes_u64));

    for (size_t i = 0; i < static_cast<size_t>(nodes_u64); ++i) {
        Node node;
        int32_t node_layer = 0;
        ifs.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
        ifs.read(reinterpret_cast<char*>(&node_layer), sizeof(node_layer));
        if (!ifs.good() || node_layer < 0) {
            return Status::Corruption("invalid node metadata in hnsw file");
        }

        node.max_layer = node_layer;
        node.vec.resize(d.dim);
        ifs.read(reinterpret_cast<char*>(node.vec.data()),
                 static_cast<std::streamsize>(node.vec.size() * sizeof(float)));
        if (!ifs.good()) {
            return Status::Corruption("invalid vector payload in hnsw file");
        }

        node.neighbors.resize(static_cast<size_t>(node_layer + 1));
        for (int l = 0; l <= node_layer; ++l) {
            uint32_t ncount = 0;
            ifs.read(reinterpret_cast<char*>(&ncount), sizeof(ncount));
            if (!ifs.good()) {
                return Status::Corruption("invalid neighbor count in hnsw file");
            }
            node.neighbors[l].resize(ncount);
            for (uint32_t j = 0; j < ncount; ++j) {
                uint64_t idx_u64 = 0;
                ifs.read(reinterpret_cast<char*>(&idx_u64), sizeof(idx_u64));
                if (!ifs.good()) {
                    return Status::Corruption("invalid neighbor list in hnsw file");
                }
                node.neighbors[l][j] = static_cast<size_t>(idx_u64);
            }
        }

        d.id_to_idx[node.id] = d.nodes.size();
        d.nodes.push_back(std::move(node));
    }

    return Status::OK();
}

}  // namespace lumina
