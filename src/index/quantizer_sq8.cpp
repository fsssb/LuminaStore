#include "lumina/index/quantizer.h"
#include "lumina/vector/quantized_distance.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lumina {

Status ScalarQuantizer8::train(const std::vector<const float*>& samples, size_t count, size_t dim) {
    if (count == 0 || dim == 0) {
        return Status::InvalidArgument("no samples or zero dim");
    }
    dim_ = dim;
    min_ = std::numeric_limits<float>::max();
    max_ = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            min_ = std::min(min_, samples[i][j]);
            max_ = std::max(max_, samples[i][j]);
        }
    }
    if (max_ <= min_) {
        max_ = min_ + 1.0F;  // degenerate: keep a unit range
    }
    return Status::OK();
}

void ScalarQuantizer8::encode(const float* vec, QuantCode* code) const {
    code->bytes.resize(dim_);
    const float scale = 255.0F / (max_ - min_);
    for (size_t j = 0; j < dim_; ++j) {
        float v = (vec[j] - min_) * scale;
        v = std::max(0.0F, std::min(255.0F, v));
        code->bytes[j] = static_cast<uint8_t>(std::lround(v));
    }
}

float ScalarQuantizer8::distance_to_query(const QuantCode& code, const float* query,
                                          size_t dim) const {
    (void)dim;
    return quant::sq8_l2_query(code.bytes.data(), query, dim_, min_, max_);
}

float ScalarQuantizer8::distance(const QuantCode& a, const QuantCode& b) const {
    const float scale = (max_ - min_) / 255.0F;
    return quant::sq8_l2_bytes(a.bytes.data(), b.bytes.data(), dim_, scale);
}

size_t ScalarQuantizer8::code_bytes() const { return dim_; }

Status ScalarQuantizer8::save(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    os.write(reinterpret_cast<const char*>(&min_), sizeof(min_));
    os.write(reinterpret_cast<const char*>(&max_), sizeof(max_));
    return os.good() ? Status::OK() : Status::IOError("sq8 save failed");
}

Status ScalarQuantizer8::load(std::istream& is) {
    is.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    is.read(reinterpret_cast<char*>(&min_), sizeof(min_));
    is.read(reinterpret_cast<char*>(&max_), sizeof(max_));
    return is.good() ? Status::OK() : Status::Corruption("sq8 load failed");
}

float ScalarQuantizer8::dequant(const uint8_t* bytes, size_t j) const {
    return min_ + (static_cast<float>(bytes[j]) / 255.0F) * (max_ - min_);
}

float ScalarQuantizer8::l2_dequant(const uint8_t* bytes, const float* other, size_t dim) const {
    float sum = 0.0F;
    for (size_t j = 0; j < dim; ++j) {
        const float d = dequant(bytes, j) - other[j];
        sum += d * d;
    }
    return sum;
}

float ScalarQuantizer8::l2_dequant(const uint8_t* a, const uint8_t* b, size_t dim) const {
    float sum = 0.0F;
    for (size_t j = 0; j < dim; ++j) {
        const float d = dequant(a, j) - dequant(b, j);
        sum += d * d;
    }
    return sum;
}

} // namespace lumina
