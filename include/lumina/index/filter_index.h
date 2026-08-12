#pragma once

#include "lumina/common/types.h"
#include "lumina/engine/filter.h"

#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lumina {

// Simple bitset indexed by uint64_t id. Grows on demand.
class BitSet {
public:
    void set(uint64_t id) {
        ensure(id);
        words_[id >> 6] |= (uint64_t{1} << (id & 63u));
    }

    void reset(uint64_t id) {
        if (id >> 6 < words_.size()) {
            words_[id >> 6] &= ~(uint64_t{1} << (id & 63u));
        }
    }

    bool test(uint64_t id) const {
        if (id >> 6 >= words_.size()) {
            return false;
        }
        return (words_[id >> 6] & (uint64_t{1} << (id & 63u))) != 0;
    }

    size_t word_count() const { return words_.size(); }
    const std::vector<uint64_t>& words() const { return words_; }

private:
    std::vector<uint64_t> words_;

    void ensure(uint64_t id) {
        const size_t need = (id >> 6) + 1;
        if (words_.size() < need) {
            words_.resize(need, 0);
        }
    }
};

// FilterIndex maps scalar filter fields to id-bitsets for O(1) per-id predicate
// evaluation. Storage: for each field, one bitset per distinct value (int /
// double / string). Range predicates (<, <=, >, >=) scan the field's values.
//
// Thread safety: writers (add/update/remove) hold an exclusive lock; readers
// (matches) hold a shared lock.
class FilterIndex {
public:
    // Register/replace the scalar fields of an id.
    void add(uint64_t id, const std::vector<ScalarField>& scalars);

    // Remove an id from all field value sets.
    void remove(uint64_t id);

    // Check whether an id satisfies every clause (AND semantics).
    bool matches(uint64_t id, const FilterExpr& expr) const;

    // Whether any id has a value for the field.
    bool has_field(const std::string& field) const;

    size_t field_count() const;

private:
    struct FieldIndex {
        std::unordered_map<int64_t, BitSet>   ints;
        std::unordered_map<double, BitSet>    doubles;
        std::unordered_map<std::string, BitSet> strs;
    };

    std::unordered_map<std::string, FieldIndex> fields_;
    BitSet all_ids_;
    mutable std::shared_mutex mutex_;

    FieldIndex& field(const std::string& name);
    const FieldIndex* find_field(const std::string& name) const;

    // Whether `id` is in the bitset for (field, value).
    static bool value_contains(const FieldIndex& f, const ScalarValue& v, uint64_t id);

    bool clause_matches(uint64_t id, const Clause& c) const;
};

} // namespace lumina
