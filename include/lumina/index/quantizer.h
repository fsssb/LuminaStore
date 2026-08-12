#pragma once

#include "lumina/common/types.h"

#include <cstddef>
#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <vector>

namespace lumina {

// Quantized code for one vector: opaque bytes produced by a Quantizer.
struct QuantCode {
    std::vector<uint8_t> bytes;
    size_t code_bytes() const { return bytes.size(); }
};

// Quantizer abstraction: train on a sample set, then encode vectors and
// compute approximate distances between a code and a raw query vector (and
// between two codes). Implementations: SQ8 (int8 scalar), Binary (1-bit),
// PQ (product quantization with ADC).
//
// Supported metrics:
//   SQ8    -> L2 (approx via dequantized L2)
//   Binary -> Hamming (suited for cosine/IP embeddings)
//   PQ     -> L2 via asymmetric distance computation (ADC)
class Quantizer {
public:
    virtual ~Quantizer() = default;

    // Train on `count` vectors (each dim floats). Call before encode.
    virtual Status train(const std::vector<const float*>& samples, size_t count, size_t dim) = 0;

    // Encode a raw vector into a code.
    virtual void encode(const float* vec, QuantCode* code) const = 0;

    // Approximate distance between an encoded vector and a raw query vector.
    virtual float distance_to_query(const QuantCode& code, const float* query, size_t dim) const = 0;

    // Approximate distance between two encoded vectors.
    virtual float distance(const QuantCode& a, const QuantCode& b) const = 0;

    // Encoded size in bytes per vector.
    virtual size_t code_bytes() const = 0;

    // Serialize the quantizer (codebooks etc.). No vectors are stored.
    virtual Status save(std::ostream& os) const = 0;
    virtual Status load(std::istream& is) = 0;
};

// SQ8: global min/max normalization to uint8, L2 on dequantized vectors.
class ScalarQuantizer8 : public Quantizer {
public:
    Status train(const std::vector<const float*>& samples, size_t count, size_t dim) override;
    void encode(const float* vec, QuantCode* code) const override;
    float distance_to_query(const QuantCode& code, const float* query, size_t dim) const override;
    float distance(const QuantCode& a, const QuantCode& b) const override;
    size_t code_bytes() const override;
    Status save(std::ostream& os) const override;
    Status load(std::istream& is) override;

private:
    size_t dim_ = 0;
    float  min_ = 0.0F;
    float  max_ = 1.0F;

    float dequant(const uint8_t* bytes, size_t j) const;
    float l2_dequant(const uint8_t* bytes, const float* other, size_t dim) const;
    float l2_dequant(const uint8_t* a, const uint8_t* b, size_t dim) const;
};

// Binary: sign bit per float, Hamming distance via popcount.
class BinaryQuantizer : public Quantizer {
public:
    Status train(const std::vector<const float*>& samples, size_t count, size_t dim) override;
    void encode(const float* vec, QuantCode* code) const override;
    float distance_to_query(const QuantCode& code, const float* query, size_t dim) const override;
    float distance(const QuantCode& a, const QuantCode& b) const override;
    size_t code_bytes() const override;
    Status save(std::ostream& os) const override;
    Status load(std::istream& is) override;

private:
    size_t dim_ = 0;
    float hamming(const std::vector<uint8_t>& code, const float* query) const;
};

// PQ: split dim into m subspaces, k-means per subspace, encode each subspace
// as the nearest centroid index (1 byte if k <= 256). distance_to_query uses
// ADC; distance between codes uses the centroid approximation.
class ProductQuantizer : public Quantizer {
public:
    ProductQuantizer(size_t m, size_t k);

    Status train(const std::vector<const float*>& samples, size_t count, size_t dim) override;
    void encode(const float* vec, QuantCode* code) const override;
    float distance_to_query(const QuantCode& code, const float* query, size_t dim) const override;
    float distance(const QuantCode& a, const QuantCode& b) const override;
    size_t code_bytes() const override;
    Status save(std::ostream& os) const override;
    Status load(std::istream& is) override;

private:
    size_t dim_ = 0;
    size_t m_ = 16;
    size_t k_ = 256;
    size_t subdim_ = 0;
    std::vector<std::vector<std::vector<float>>> codebooks_;  // [subspace][centroid][subdim]
};

// Create a quantizer from the config.
std::unique_ptr<Quantizer> make_quantizer(const QuantConfig& cfg);

} // namespace lumina
