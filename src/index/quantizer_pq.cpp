#include "lumina/index/quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace lumina {

namespace {

// Mini k-means on a set of sub-vectors. Returns per-point cluster ids.
std::vector<uint32_t> kmeans(const std::vector<std::vector<float>>& points, size_t k, size_t iters,
                             size_t subdim, std::mt19937* rng,
                             std::vector<std::vector<float>>* out_centroids) {
    std::vector<std::vector<float>> centroids;
    if (points.size() <= k) {
        centroids = points;  // fewer points than clusters: use them as-is
    } else {
        std::vector<size_t> idx(points.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), *rng);
        for (size_t i = 0; i < k; ++i) {
            centroids.push_back(points[idx[i]]);
        }
    }

    std::vector<uint32_t> assign(points.size(), 0);
    for (size_t it = 0; it < iters; ++it) {
        for (size_t p = 0; p < points.size(); ++p) {
            float best = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (size_t c = 0; c < centroids.size(); ++c) {
                float sum = 0.0F;
                for (size_t j = 0; j < subdim; ++j) {
                    const float d = points[p][j] - centroids[c][j];
                    sum += d * d;
                }
                if (sum < best) {
                    best = sum;
                    best_c = static_cast<uint32_t>(c);
                }
            }
            assign[p] = best_c;
        }
        std::vector<std::vector<float>> sums(centroids.size(), std::vector<float>(subdim, 0.0F));
        std::vector<size_t> counts(centroids.size(), 0);
        for (size_t p = 0; p < points.size(); ++p) {
            for (size_t j = 0; j < subdim; ++j) {
                sums[assign[p]][j] += points[p][j];
            }
            ++counts[assign[p]];
        }
        for (size_t c = 0; c < centroids.size(); ++c) {
            if (counts[c] > 0) {
                for (size_t j = 0; j < subdim; ++j) {
                    centroids[c][j] = sums[c][j] / static_cast<float>(counts[c]);
                }
            }
        }
    }
    *out_centroids = std::move(centroids);
    return assign;
}

}  // namespace

ProductQuantizer::ProductQuantizer(size_t m, size_t k)
    : m_(std::max<size_t>(m, 1)), k_(std::max<size_t>(k, 2)) {}

Status ProductQuantizer::train(const std::vector<const float*>& samples, size_t count, size_t dim) {
    if (count == 0 || dim == 0 || dim < m_) {
        return Status::InvalidArgument("bad pq dims");
    }
    dim_ = dim;
    subdim_ = dim_ / m_;
    if (subdim_ == 0 || dim_ % m_ != 0) {
        return Status::InvalidArgument("dim must be divisible by subspaces");
    }
    std::vector<std::vector<std::vector<float>>> per_sub(m_);
    for (size_t p = 0; p < count; ++p) {
        for (size_t s = 0; s < m_; ++s) {
            std::vector<float> sub(subdim_);
            for (size_t j = 0; j < subdim_; ++j) {
                sub[j] = samples[p][s * subdim_ + j];
            }
            per_sub[s].push_back(std::move(sub));
        }
    }
    std::mt19937 rng(7);
    codebooks_.resize(m_);
    for (size_t s = 0; s < m_; ++s) {
        kmeans(per_sub[s], k_, 8, subdim_, &rng, &codebooks_[s]);
    }
    return Status::OK();
}

void ProductQuantizer::encode(const float* vec, QuantCode* code) const {
    code->bytes.resize(m_);
    for (size_t s = 0; s < m_; ++s) {
        float best = std::numeric_limits<float>::max();
        uint8_t best_c = 0;
        for (size_t c = 0; c < codebooks_[s].size(); ++c) {
            float sum = 0.0F;
            for (size_t j = 0; j < subdim_; ++j) {
                const float d = vec[s * subdim_ + j] - codebooks_[s][c][j];
                sum += d * d;
            }
            if (sum < best) {
                best = sum;
                best_c = static_cast<uint8_t>(c);
            }
        }
        code->bytes[s] = best_c;
    }
}

float ProductQuantizer::distance_to_query(const QuantCode& code, const float* query,
                                          size_t dim) const {
    (void)dim;
    float sum = 0.0F;
    for (size_t s = 0; s < m_; ++s) {
        const uint8_t c = code.bytes[s];
        for (size_t j = 0; j < subdim_; ++j) {
            const float d = query[s * subdim_ + j] - codebooks_[s][c][j];
            sum += d * d;
        }
    }
    return sum;
}

float ProductQuantizer::distance(const QuantCode& a, const QuantCode& b) const {
    float sum = 0.0F;
    for (size_t s = 0; s < m_; ++s) {
        const auto& ca = codebooks_[s][a.bytes[s]];
        const auto& cb = codebooks_[s][b.bytes[s]];
        for (size_t j = 0; j < subdim_; ++j) {
            const float d = ca[j] - cb[j];
            sum += d * d;
        }
    }
    return sum;
}

size_t ProductQuantizer::code_bytes() const { return m_; }

Status ProductQuantizer::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    os.write(reinterpret_cast<const char*>(&m_), sizeof(m_));
    os.write(reinterpret_cast<const char*>(&k_), sizeof(k_));
    os.write(reinterpret_cast<const char*>(&subdim_), sizeof(subdim_));
    for (const auto& cb : codebooks_) {
        for (const auto& centroid : cb) {
            os.write(reinterpret_cast<const char*>(centroid.data()),
                     static_cast<std::streamsize>(centroid.size() * sizeof(float)));
        }
    }
    return os.good() ? Status::OK() : Status::IOError("pq save failed");
}

Status ProductQuantizer::load(std::istream& is) {
    is.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    is.read(reinterpret_cast<char*>(&m_), sizeof(m_));
    is.read(reinterpret_cast<char*>(&k_), sizeof(k_));
    is.read(reinterpret_cast<char*>(&subdim_), sizeof(subdim_));
    if (!is.good() || dim_ == 0 || m_ == 0 || k_ == 0 || subdim_ == 0) {
        return Status::Corruption("pq load bad header");
    }
    codebooks_.assign(m_, std::vector<std::vector<float>>(k_, std::vector<float>(subdim_)));
    for (auto& cb : codebooks_) {
        for (auto& centroid : cb) {
            is.read(reinterpret_cast<char*>(centroid.data()),
                    static_cast<std::streamsize>(centroid.size() * sizeof(float)));
        }
    }
    return is.good() ? Status::OK() : Status::Corruption("pq load failed");
}

} // namespace lumina
