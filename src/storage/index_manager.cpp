#include "lumina/storage/index_manager.h"

namespace lumina {

void IndexManager::put(const std::string& key, uint64_t offset) {
    map_[key] = offset;
}

void IndexManager::remove(const std::string& key) {
    map_.erase(key);
}

std::optional<uint64_t> IndexManager::get(const std::string& key) const {
    const auto it = map_.find(key);
    if (it == map_.end()) {
        return std::nullopt;
    }
    return it->second;
}

bool IndexManager::contains(const std::string& key) const {
    return map_.find(key) != map_.end();
}

}  // namespace lumina
