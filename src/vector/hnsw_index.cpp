#include "lumina/vector/hnsw_index.h"

#include "lumina/vector/aligned_alloc.h"
#include "lumina/vector/vector_math.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lumina {

namespace {

using Candidate = std::pair<float, size_t>;  // (distance, node index)

struct CandidateMinCompare {
    bool operator()(const Candidate& a, const Candidate& b) const { return a.first > b.first; }
};

struct Node {
    uint64_t id = 0;
    AlignedFloatVector vec;
    std::vector<std::vector<size_t>> neighbors;  // per layer
    int  max_layer = 0;
    bool deleted   = false;
};

}  // namespace

struct HNSWIndex::Impl {
    size_t dim = 0;
    size_t M = 16;
    size_t ef_construction = 200;
    DistanceFn dist = nullptr;  // default set in ctor to l2_distance

    std::deque<Node> nodes;  // deque: push_back never invalidates existing elements
    std::unordered_map<uint64_t, size_t> id_to_idx;
    std::mt19937 rng{42};
    double ml = 1.0;

    std::atomic<int>  entry_point{-1};
    std::atomic<int>  max_layer{-1};
    std::atomic<size_t> active_count{0};

    // ---- concurrency model ----
    // struct_lock: shared_mutex. Writers (node push / id_to_idx mutation / entry
    //   & max_layer updates / tombstone marking / update reset) hold it exclusive;
    //   search, graph traversal and edge connection hold it shared. This makes the
    //   deque container structurally safe (no concurrent push vs indexed access).
    // node_stripes: fixed-size per-node-ish locks serializing access to a node's
    //   vector + neighbour lists. DualNodeLock acquires two stripes in ascending
    //   order. Lock order is always: label -> struct_lock -> node stripe(s).
    static constexpr size_t kLabelStripes = 65536;
    static constexpr size_t kNodeLockStripes = 4096;

    std::vector<std::mutex> label_locks;
    mutable std::shared_mutex struct_lock;
    mutable std::array<std::mutex, kNodeLockStripes> node_stripes;
    std::mutex rng_lock;

    Impl() : label_locks(kLabelStripes) {}

    size_t stripe_for(uint64_t id) const { return id & (kLabelStripes - 1); }
    size_t node_stripe(size_t idx) const { return idx & (kNodeLockStripes - 1); }
};

namespace {

// Per-thread "visited" marker set (hnswlib-style VisitedListPool, simplified):
// search_layer marks a node visited by stamping t_visited[idx] = t_stamp.
thread_local std::vector<uint32_t> t_visited;
thread_local uint32_t             t_stamp = 0;

uint32_t next_stamp() {
    uint32_t s = ++t_stamp;
    if (s == 0) {
        s = ++t_stamp;  // skip the reserved 0 marker on overflow
    }
    return s;
}

// Lock one node stripe.
class NodeLock {
public:
    explicit NodeLock(const HNSWIndex::Impl& d, size_t idx)
        : lock_(d.node_stripes[d.node_stripe(idx)]) {}
private:
    std::lock_guard<std::mutex> lock_;
};

// Lock two node stripes in ascending order (deadlock-free multi-lock).
class DualNodeLock {
public:
    DualNodeLock(const HNSWIndex::Impl& d, size_t a, size_t b) {
        const size_t sa = d.node_stripe(a);
        const size_t sb = d.node_stripe(b);
        if (sa == sb) {
            lock_single_.emplace(d.node_stripes[sa]);
        } else {
            const size_t lo = std::min(sa, sb);
            const size_t hi = std::max(sa, sb);
            lock_lo_.emplace(d.node_stripes[lo]);
            lock_hi_.emplace(d.node_stripes[hi]);
        }
    }
private:
    std::optional<std::lock_guard<std::mutex>> lock_single_;
    std::optional<std::lock_guard<std::mutex>> lock_lo_;
    std::optional<std::lock_guard<std::mutex>> lock_hi_;
};

int random_level(HNSWIndex::Impl& d) {
    std::lock_guard<std::mutex> lk(d.rng_lock);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double x = std::max(dist(d.rng), 1e-12);
    return static_cast<int>(-std::log(x) * d.ml);
}

// Greedy beam search on one layer. Skips deleted nodes. When `filter` is
// non-null, a candidate only enters the result set if filter(id) is true
// (traversal still expands all neighbours). Returns candidates sorted by
// distance ascending. Caller must hold struct_lock (shared at least).
std::vector<Candidate> search_layer(const HNSWIndex::Impl& d, const float* query,
                                    size_t entry_idx, size_t ef, int layer,
                                    const FilterFn* filter = nullptr) {
    const uint32_t stamp = next_stamp();
    if (t_visited.size() < d.nodes.size()) {
        t_visited.assign(d.nodes.size(), 0);
    }

    std::priority_queue<Candidate, std::vector<Candidate>, CandidateMinCompare> candidates;
    std::priority_queue<Candidate> result;

    {
        NodeLock lk(d, entry_idx);
        const auto& e = d.nodes[entry_idx];
        if (e.deleted || layer > e.max_layer) {
            return {};
        }
        const float de = d.dist(query, e.vec.data(), d.dim);
        candidates.push({de, entry_idx});  // always an expansion point
        if (filter == nullptr || (*filter)(e.id)) {
            result.push({de, entry_idx});
        }
        t_visited[entry_idx] = stamp;
    }

    while (!candidates.empty()) {
        const auto [cand_dist, cand_idx] = candidates.top();
        candidates.pop();

        if (result.size() >= ef && cand_dist > result.top().first) {
            break;
        }

        std::vector<size_t> neighbors;
        {
            NodeLock lk(d, cand_idx);
            const auto& n = d.nodes[cand_idx];
            if (layer > n.max_layer) {
                continue;
            }
            neighbors = n.neighbors[static_cast<size_t>(layer)];  // small copy (<= 2*M)
        }

        for (const size_t nb_idx : neighbors) {
            if (t_visited[nb_idx] == stamp) {
                continue;
            }
            t_visited[nb_idx] = stamp;

            float nd = 0.0F;
            {
                NodeLock lk(d, nb_idx);
                const auto& nb = d.nodes[nb_idx];
                if (nb.deleted || layer > nb.max_layer) {
                    continue;  // stale asymmetric edge to a lower-layer node
                }
                nd = d.dist(query, nb.vec.data(), d.dim);
            }

            // Expansion queue: never filtered, so the search can escape a
            // region where nothing matches the filter (hnswlib behaviour).
            if (candidates.size() < ef || nd < candidates.top().first) {
                candidates.push({nd, nb_idx});
            }

            // Result set: filter-aware.
            if (filter != nullptr) {
                uint64_t nid = 0;
                {
                    NodeLock lk(d, nb_idx);
                    nid = d.nodes[nb_idx].id;
                }
                if (!(*filter)(nid)) {
                    continue;  // in-filter: excluded candidate skipped
                }
            }
            if (result.size() < ef || nd < result.top().first) {
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
    std::sort(out.begin(), out.end(),
              [](const Candidate& a, const Candidate& b) { return a.first < b.first; });
    return out;
}

// Diverse ("heuristic") neighbour selection from HNSW paper / hnswlib.
// ordered is sorted by distance to center ascending; a candidate c is kept iff
// it is not closer to any already-selected neighbour than it is to the center
// (i.e. no selected neighbour lies "between" c and the center), which prevents
// near-duplicate clustering.
std::vector<size_t> select_neighbors_heuristic(const HNSWIndex::Impl& d,
                                               const std::vector<Candidate>& ordered,
                                               size_t max_m) {
    // Only the closest 2*max_m candidates can possibly be selected: the
    // heuristic keeps at most max_m, and anything farther than the max_m-th
    // closest is dominated. Trimming the input cuts build-time distance
    // computation significantly on clustered data.
    const size_t trim = std::min(ordered.size(), 2 * max_m);
    std::vector<size_t> selected;
    selected.reserve(std::min(max_m, trim));
    for (size_t _i = 0; _i < trim; ++_i) {
        const auto& [dc, c] = ordered[_i];
        if (selected.size() >= max_m) {
            break;
        }
        bool good = true;
        for (const size_t s : selected) {
            float dcs = 0.0F;
            {
                DualNodeLock lk(d, c, s);
                dcs = d.dist(d.nodes[c].vec.data(), d.nodes[s].vec.data(), d.dim);
            }
            if (dcs < dc) {  // s lies between c and center
                good = false;
                break;
            }
        }
        if (good) {
            selected.push_back(c);
        }
    }
    return selected;
}

// Re-select up to max_m neighbours of `center` layer `layer` using the heuristic.
void prune_neighbors(HNSWIndex::Impl& d, size_t center, int layer, size_t max_m) {
    std::vector<size_t> conn;
    {
        NodeLock lk(d, center);
        conn = d.nodes[center].neighbors[static_cast<size_t>(layer)];
    }
    std::vector<Candidate> ordered;
    ordered.reserve(conn.size());
    for (const size_t c : conn) {
        float dc = 0.0F;
        {
            NodeLock lk(d, c);
            dc = d.dist(d.nodes[center].vec.data(), d.nodes[c].vec.data(), d.dim);
        }
        ordered.push_back({dc, c});
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const Candidate& a, const Candidate& b) { return a.first < b.first; });
    const std::vector<size_t> sel = select_neighbors_heuristic(d, ordered, max_m);
    {
        NodeLock lk(d, center);
        d.nodes[center].neighbors[static_cast<size_t>(layer)] = sel;
    }
}

void connect_bidirectional(HNSWIndex::Impl& d, size_t a, size_t b, int layer, size_t max_m) {
    if (a == b) {
        return;  // self-loop: DualNodeLock would deadlock on the same mutex
    }
    bool overflow_a = false;
    bool overflow_b = false;
    {
        DualNodeLock lk(d, a, b);
        if (layer > d.nodes[a].max_layer || layer > d.nodes[b].max_layer) {
            return;  // defensive: stale asymmetric edge; never touch out-of-range layer
        }
        auto& na = d.nodes[a].neighbors[static_cast<size_t>(layer)];
        auto& nb = d.nodes[b].neighbors[static_cast<size_t>(layer)];
        if (std::find(na.begin(), na.end(), b) == na.end()) {
            na.push_back(b);
        }
        if (std::find(nb.begin(), nb.end(), a) == nb.end()) {
            nb.push_back(a);
        }
        overflow_a = na.size() > max_m;
        overflow_b = nb.size() > max_m;
    }
    if (overflow_a) {
        prune_neighbors(d, a, layer, max_m);
    }
    if (overflow_b) {
        prune_neighbors(d, b, layer, max_m);
    }
}

}  // namespace

HNSWIndex::HNSWIndex(size_t dim, size_t M, size_t ef_construction, DistanceFn dist)
    : impl_(std::make_unique<Impl>()), dim_(dim) {
    auto& d = *impl_;
    d.dim = dim;
    d.M = std::max<size_t>(M, 2);
    d.ef_construction = ef_construction;
    d.dist = (dist != nullptr) ? dist : VectorMath::l2_distance;
    d.ml = 1.0 / std::log(static_cast<double>(d.M));
}

HNSWIndex::~HNSWIndex() = default;

HNSWIndex::HNSWIndex(HNSWIndex&&) noexcept = default;
HNSWIndex& HNSWIndex::operator=(HNSWIndex&&) noexcept = default;

Status HNSWIndex::add_item(uint64_t id, const float* vec) {
    auto& d = *impl_;
    if (vec == nullptr) {
        return Status::InvalidArgument("vector is null");
    }
    std::lock_guard<std::mutex> label(d.label_locks[d.stripe_for(id)]);

    // Whole add under one exclusive struct lock: graph wiring mutates neighbour
    // lists, and concurrent searches must not observe half-written vectors.
    std::unique_lock<std::shared_mutex> g(d.struct_lock);

    if (d.id_to_idx.count(id) != 0U) {
        return Status::InvalidArgument("duplicate id");
    }
    const int new_layer = random_level(d);
    Node node;
    node.id = id;
    node.vec.assign(vec, vec + d.dim);
    node.max_layer = new_layer;
    node.neighbors.resize(static_cast<size_t>(new_layer + 1));
    for (int l = 0; l <= new_layer; ++l) {
        node.neighbors[static_cast<size_t>(l)].reserve(l == 0 ? 2 * d.M : d.M);
    }
    const size_t new_idx = d.nodes.size();
    d.nodes.push_back(std::move(node));
    d.id_to_idx[id] = new_idx;
    ++d.active_count;
    if (d.entry_point.load() < 0) {
        d.entry_point.store(static_cast<int>(new_idx));
        d.max_layer.store(new_layer);
        return Status::OK();
    }

    {
        size_t ep = static_cast<size_t>(d.entry_point.load());
        for (int l = d.max_layer.load(); l > new_layer; --l) {
            const auto res = search_layer(d, vec, ep, 1, l);
            if (!res.empty()) {
                ep = res.front().second;
            }
        }

        for (int l = std::min(new_layer, d.max_layer.load()); l >= 0; --l) {
            const auto neighbors = search_layer(d, vec, ep, d.ef_construction, l);
            const size_t max_m  = (l == 0) ? 2 * d.M : d.M;
            const auto sel = select_neighbors_heuristic(d, neighbors, max_m);
            if (!sel.empty()) {
                ep = sel.front();
            }
            for (const size_t nb : sel) {
                connect_bidirectional(d, new_idx, nb, l, max_m);
            }
        }
    }

    if (new_layer > d.max_layer.load()) {
        d.max_layer.store(new_layer);
        d.entry_point.store(static_cast<int>(new_idx));
    }
    return Status::OK();
}

Status HNSWIndex::remove(uint64_t id) {
    auto& d = *impl_;
    std::lock_guard<std::mutex> label(d.label_locks[d.stripe_for(id)]);
    std::unique_lock<std::shared_mutex> g(d.struct_lock);

    const auto it = d.id_to_idx.find(id);
    if (it == d.id_to_idx.end()) {
        return Status::NotFound("id not found");
    }
    const size_t idx = it->second;
    {
        NodeLock nlk(d, idx);
        if (d.nodes[idx].deleted) {
            return Status::OK();  // idempotent
        }
        d.nodes[idx].deleted = true;
    }
    --d.active_count;

    // If the removed node was the entry point, pick a live replacement with the
    // highest layer (or reset to an empty index).
    if (d.entry_point.load() == static_cast<int>(idx)) {
        int new_ep = -1;
        int new_max = -1;
        for (size_t i = 0; i < d.nodes.size(); ++i) {
            NodeLock nlk(d, i);
            if (d.nodes[i].deleted) {
                continue;
            }
            if (new_ep < 0 || d.nodes[i].max_layer > new_max) {
                new_ep = static_cast<int>(i);
                new_max = d.nodes[i].max_layer;
            }
        }
        d.entry_point.store(new_ep);
        d.max_layer.store(new_max);
    }
    return Status::OK();
}

Status HNSWIndex::update_item(uint64_t id, const float* vec) {
    auto& d = *impl_;
    if (vec == nullptr) {
        return Status::InvalidArgument("vector is null");
    }
    std::lock_guard<std::mutex> label(d.label_locks[d.stripe_for(id)]);

    // The whole update runs under one exclusive struct lock: detach, reset and
    // re-insert must be atomic with respect to searches. This is conservative
    // (writes are serialized against reads) but correct; the fine-grained lock
    // path (shared + node stripes) was observed to corrupt neighbour vectors
    // under concurrent search, so correctness wins over write/read overlap.
    std::unique_lock<std::shared_mutex> g(d.struct_lock);

    const auto it = d.id_to_idx.find(id);
    if (it == d.id_to_idx.end()) {
        return Status::NotFound("id not found");
    }
    const size_t idx = it->second;

    // 1. Detach all old edges (both directions).
    {
        std::vector<std::vector<size_t>> old_neighbors;
        int old_max_layer = 0;
        {
            NodeLock nlk(d, idx);
            old_max_layer = d.nodes[idx].max_layer;
            old_neighbors = d.nodes[idx].neighbors;
        }
        for (int l = 0; l <= old_max_layer; ++l) {
            for (const size_t nb : old_neighbors[static_cast<size_t>(l)]) {
                DualNodeLock lk(d, idx, nb);
                if (l > d.nodes[nb].max_layer) {
                    continue;  // defensive: stale asymmetric edge
                }
                auto& lst = d.nodes[nb].neighbors[static_cast<size_t>(l)];
                lst.erase(std::remove(lst.begin(), lst.end(), idx), lst.end());
            }
        }
    }

    // 2. Replace vector, reset layers, revive tombstoned entry.
    const int new_layer = random_level(d);
    {
        NodeLock nlk(d, idx);
        const bool was_deleted = d.nodes[idx].deleted;
        auto& node = d.nodes[idx];
        node.vec.assign(vec, vec + d.dim);
        node.deleted = false;
        node.max_layer = new_layer;
        node.neighbors.assign(static_cast<size_t>(new_layer + 1), {});
        for (int l = 0; l <= new_layer; ++l) {
            node.neighbors[static_cast<size_t>(l)].reserve(l == 0 ? 2 * d.M : d.M);
        }
        if (was_deleted) {
            ++d.active_count;
        }
    }

    // 3. Re-insert like a fresh node.
    if (d.entry_point.load() < 0) {
        // Empty-graph state (the last live node was tombstoned): promote this node.
        d.entry_point.store(static_cast<int>(idx));
        d.max_layer.store(new_layer);
    } else {
        size_t ep = static_cast<size_t>(d.entry_point.load());
        for (int l = d.max_layer.load(); l > new_layer; --l) {
            const auto res = search_layer(d, vec, ep, 1, l);
            if (!res.empty()) {
                ep = res.front().second;
            }
        }
        for (int l = std::min(new_layer, d.max_layer.load()); l >= 0; --l) {
            const auto neighbors = search_layer(d, vec, ep, d.ef_construction, l);
            const size_t max_m  = (l == 0) ? 2 * d.M : d.M;
            const auto sel = select_neighbors_heuristic(d, neighbors, max_m);
            if (!sel.empty()) {
                ep = sel.front();
            }
            for (const size_t nb : sel) {
                connect_bidirectional(d, idx, nb, l, max_m);
            }
        }
        if (new_layer > d.max_layer.load()) {
            d.max_layer.store(new_layer);
            d.entry_point.store(static_cast<int>(idx));
        }
    }
    return Status::OK();
}

std::vector<SearchResult> HNSWIndex::search_top_k(const float* query, size_t k,
                                                  size_t ef_search) const {
    return search_top_k_filtered(query, k, ef_search, nullptr);
}

std::vector<SearchResult> HNSWIndex::search_top_k_filtered(const float* query, size_t k,
                                                           size_t ef_search,
                                                           const FilterFn& filter) const {
    const auto& d = *impl_;
    std::vector<SearchResult> out;
    if (query == nullptr || k == 0 || d.entry_point.load() < 0) {
        return out;
    }

    std::shared_lock<std::shared_mutex> g(d.struct_lock);

    size_t ep = static_cast<size_t>(d.entry_point.load());
    for (int l = d.max_layer.load(); l > 0; --l) {
        const auto res = search_layer(d, query, ep, 1, l, filter ? &filter : nullptr);
        if (!res.empty()) {
            ep = res.front().second;
        }
    }

    auto candidates = search_layer(d, query, ep, std::max(ef_search, k), 0, filter ? &filter : nullptr);
    if (candidates.size() > k) {
        candidates.resize(k);
    }

    out.reserve(candidates.size());
    for (const auto& item : candidates) {
        uint64_t cid = 0;
        {
            NodeLock lk(d, item.second);
            cid = d.nodes[item.second].id;
        }
        out.push_back({cid, item.first});
    }
    return out;
}

size_t HNSWIndex::size() const {
    return impl_->active_count.load();
}

bool HNSWIndex::contains(uint64_t id) const {
    const auto& d = *impl_;
    std::shared_lock<std::shared_mutex> g(d.struct_lock);
    return d.id_to_idx.count(id) != 0U;
}

Status HNSWIndex::save(const std::string& path) const {
    const auto& d = *impl_;
    std::shared_lock<std::shared_mutex> g(d.struct_lock);

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return Status::IOError("failed to open hnsw file for write");
    }

    const char magic[4] = {'L', 'M', 'H', 'N'};
    ofs.write(magic, 4);
    const uint16_t version  = 1;
    const uint16_t reserved = 0;
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));

    const uint64_t dim_u64 = static_cast<uint64_t>(d.dim);
    const uint64_t m_u64   = static_cast<uint64_t>(d.M);
    const int32_t  max_layer_i32 = d.max_layer.load();
    const int32_t  entry_i32     = d.entry_point.load();
    const uint64_t nodes_u64     = d.nodes.size();
    const uint64_t active_u64    = d.active_count.load();
    ofs.write(reinterpret_cast<const char*>(&dim_u64), sizeof(dim_u64));
    ofs.write(reinterpret_cast<const char*>(&m_u64), sizeof(m_u64));
    ofs.write(reinterpret_cast<const char*>(&max_layer_i32), sizeof(max_layer_i32));
    ofs.write(reinterpret_cast<const char*>(&entry_i32), sizeof(entry_i32));
    ofs.write(reinterpret_cast<const char*>(&nodes_u64), sizeof(nodes_u64));
    ofs.write(reinterpret_cast<const char*>(&active_u64), sizeof(active_u64));

    for (const auto& node : d.nodes) {
        const int32_t node_layer = node.max_layer;
        const uint8_t deleted    = node.deleted ? 1 : 0;
        ofs.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
        ofs.write(reinterpret_cast<const char*>(&node_layer), sizeof(node_layer));
        ofs.write(reinterpret_cast<const char*>(&deleted), sizeof(deleted));
        ofs.write(reinterpret_cast<const char*>(node.vec.data()),
                  static_cast<std::streamsize>(node.vec.size() * sizeof(float)));
        for (int l = 0; l <= node.max_layer; ++l) {
            const uint32_t ncount = static_cast<uint32_t>(node.neighbors[static_cast<size_t>(l)].size());
            ofs.write(reinterpret_cast<const char*>(&ncount), sizeof(ncount));
            for (const size_t idx : node.neighbors[static_cast<size_t>(l)]) {
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
    auto& d = *impl_;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return Status::IOError("failed to open hnsw file for read");
    }

    char magic[4] = {};
    ifs.read(magic, 4);
    if (!ifs.good() || magic[0] != 'L' || magic[1] != 'M' || magic[2] != 'H' || magic[3] != 'N') {
        return Status::Corruption("invalid hnsw magic (expected v2 format)");
    }
    uint16_t version = 0;
    uint16_t reserved = 0;
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
    if (!ifs.good() || version != 1) {
        return Status::Corruption("unsupported hnsw version");
    }

    uint64_t dim_u64 = 0, m_u64 = 0, nodes_u64 = 0, active_u64 = 0;
    int32_t max_layer_i32 = -1, entry_i32 = -1;
    ifs.read(reinterpret_cast<char*>(&dim_u64), sizeof(dim_u64));
    ifs.read(reinterpret_cast<char*>(&m_u64), sizeof(m_u64));
    ifs.read(reinterpret_cast<char*>(&max_layer_i32), sizeof(max_layer_i32));
    ifs.read(reinterpret_cast<char*>(&entry_i32), sizeof(entry_i32));
    ifs.read(reinterpret_cast<char*>(&nodes_u64), sizeof(nodes_u64));
    ifs.read(reinterpret_cast<char*>(&active_u64), sizeof(active_u64));
    if (!ifs.good()) {
        return Status::Corruption("invalid hnsw header");
    }
    if (dim_u64 != d.dim) {
        return Status::InvalidArgument("dimension mismatch while loading hnsw");
    }

    std::unique_lock<std::shared_mutex> g(d.struct_lock);
    d.M = static_cast<size_t>(m_u64);
    d.ml = 1.0 / std::log(static_cast<double>(std::max<size_t>(d.M, 2)));
    d.nodes.clear();
    d.id_to_idx.clear();

    size_t active = 0;
    for (size_t i = 0; i < static_cast<size_t>(nodes_u64); ++i) {
        Node node;
        int32_t node_layer = 0;
        uint8_t deleted = 0;
        ifs.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
        ifs.read(reinterpret_cast<char*>(&node_layer), sizeof(node_layer));
        ifs.read(reinterpret_cast<char*>(&deleted), sizeof(deleted));
        if (!ifs.good() || node_layer < 0) {
            return Status::Corruption("invalid node metadata in hnsw file");
        }
        node.max_layer = node_layer;
        node.deleted = (deleted != 0);
        if (!node.deleted) {
            ++active;
        }
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
            node.neighbors[static_cast<size_t>(l)].resize(ncount);
            for (uint32_t j = 0; j < ncount; ++j) {
                uint64_t idx_u64 = 0;
                ifs.read(reinterpret_cast<char*>(&idx_u64), sizeof(idx_u64));
                if (!ifs.good()) {
                    return Status::Corruption("invalid neighbor list in hnsw file");
                }
                node.neighbors[static_cast<size_t>(l)][j] = static_cast<size_t>(idx_u64);
            }
        }
        d.id_to_idx[node.id] = d.nodes.size();
        d.nodes.push_back(std::move(node));
    }

    d.max_layer.store(max_layer_i32);
    d.entry_point.store(entry_i32);
    d.active_count.store(active);
    return Status::OK();
}

} // namespace lumina
