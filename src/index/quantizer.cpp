#include "lumina/index/quantizer.h"

namespace lumina {

std::unique_ptr<Quantizer> make_quantizer(const QuantConfig& cfg) {
    switch (cfg.mode) {
        case QuantConfig::Mode::kSQ8:
            return std::make_unique<ScalarQuantizer8>();
        case QuantConfig::Mode::kBinary:
            return std::make_unique<BinaryQuantizer>();
        case QuantConfig::Mode::kPQ:
            return std::make_unique<ProductQuantizer>(cfg.pq_subspaces, cfg.pq_centroids);
        case QuantConfig::Mode::kNone:
        default:
            return nullptr;
    }
}

} // namespace lumina
