#include "lumina/index/quantizer.h"
#include "lumina/vector/quantized_distance.h"

#include <cstdint>

namespace lumina {

namespace {

inline uint64_t popcount64(uint64_t x) {
    return static_cast<uint64_t>(__builtin_popcountll(x));
}

}  // namespace

Status BinaryQuantizer::train(const std::vector<const float*>& /*samples*/, size_t /*count*/,
                              size_t dim) {
    if (dim == 0) {
        return Status::InvalidArgument("zero dim");
    }
    dim_ = dim;
    return Status::OK();
}

void BinaryQuantizer::encode(const float* vec, QuantCode* code) const {
    const size_t nbytes = (dim_ + 7) / 8;
    code->bytes.assign(nbytes, 0);
    for (size_t j = 0; j < dim_; ++j) {
        if (vec[j] >= 0.0F) {
            code->bytes[j >> 3] |= static_cast<uint8_t>(1u << (j & 7u));
        }
    }
}

float BinaryQuantizer::distance_to_query(const QuantCode& code, const float* query,
                                         size_t dim) const {
    (void)dim;
    return quant::binary_hamming_query(code.bytes.data(), query, dim_);
}

float BinaryQuantizer::distance(const QuantCode& a, const QuantCode& b) const {
    return quant::binary_hamming(a.bytes.data(), b.bytes.data(), code_bytes());
}

size_t BinaryQuantizer::code_bytes() const { return (dim_ + 7) / 8; }

Status BinaryQuantizer::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    return os.good() ? Status::OK() : Status::IOError("binary save failed");
}

Status BinaryQuantizer::load(std::istream& is) {
    is.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    return is.good() ? Status::OK() : Status::Corruption("binary load failed");
}

float BinaryQuantizer::hamming(const std::vector<uint8_t>& code, const float* query) const {
    const size_t nbytes = (dim_ + 7) / 8;
    uint64_t h = 0;
    for (size_t i = 0; i < nbytes; ++i) {
        uint8_t qbits = 0;
        for (size_t b = 0; b < 8; ++b) {
            const size_t j = i * 8 + b;
            if (j < dim_ && query[j] >= 0.0F) {
                qbits |= static_cast<uint8_t>(1u << b);
            }
        }
        h += popcount64(code[i] ^ qbits);
    }
    return static_cast<float>(h);
}

} // namespace lumina
