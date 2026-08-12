#pragma once

// Minimal reader for the standard "fvecs" binary format used by ann-benchmarks
// / corpus-texmex: each vector is [int32 dim][dim * float32].
//
// Reads the whole file into a flat vector (data) and exposes vector pointers.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lumina {
namespace tools {

struct FvecsData {
    size_t dim = 0;
    size_t count = 0;
    std::vector<float> flat;  // count * dim, row-major

    const float* vec(size_t i) const { return flat.data() + static_cast<size_t>(i) * dim; }
    float* vec(size_t i) { return flat.data() + static_cast<size_t>(i) * dim; }
};

inline FvecsData read_fvecs(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("cannot open " + path);
    }

    FvecsData out;
    std::vector<float> buf;

    // Read vector-by-vector (dim int32 header per vector).
    int32_t dim = 0;
    if (!ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        throw std::runtime_error("empty fvecs file: " + path);
    }
    if (dim <= 0 || dim > 65536) {
        throw std::runtime_error("invalid fvecs dim: " + std::to_string(dim));
    }
    out.dim = static_cast<size_t>(dim);

    // First pass: count vectors.
    size_t count = 0;
    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    const std::streamoff per = static_cast<std::streamoff>(4 + dim * 4);
    if (size <= 0 || size % per != 0) {
        throw std::runtime_error("fvecs file size not a multiple of vector size");
    }
    count = static_cast<size_t>(size / per);
    out.count = count;
    out.flat.resize(count * out.dim);

    ifs.seekg(0, std::ios::beg);
    buf.resize(out.dim);
    for (size_t i = 0; i < count; ++i) {
        int32_t d = 0;
        if (!ifs.read(reinterpret_cast<char*>(&d), sizeof(d))) {
            throw std::runtime_error("truncated fvecs header");
        }
        if (d != static_cast<int32_t>(out.dim)) {
            throw std::runtime_error("inconsistent fvecs dim");
        }
        if (!ifs.read(reinterpret_cast<char*>(out.vec(i)), static_cast<std::streamsize>(out.dim * 4))) {
            throw std::runtime_error("truncated fvecs vector");
        }
    }
    return out;
}

} // namespace tools
} // namespace lumina
