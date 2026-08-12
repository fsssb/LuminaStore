#include "lumina/index/filter_index.h"

#include <utility>

namespace lumina {

FilterIndex::FieldIndex& FilterIndex::field(const std::string& name) {
    return fields_[name];
}

const FilterIndex::FieldIndex* FilterIndex::find_field(const std::string& name) const {
    const auto it = fields_.find(name);
    return (it == fields_.end()) ? nullptr : &it->second;
}

void FilterIndex::add(uint64_t id, const std::vector<ScalarField>& scalars) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (const auto& s : scalars) {
        auto& f = field(s.name);
        switch (s.value.index()) {
            case 0:
                f.ints[std::get<int64_t>(s.value)].set(id);
                break;
            case 1:
                f.doubles[std::get<double>(s.value)].set(id);
                break;
            case 2:
                f.strs[std::get<std::string>(s.value)].set(id);
                break;
        }
    }
    all_ids_.set(id);
}

void FilterIndex::remove(uint64_t id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (auto& [name, f] : fields_) {
        (void)name;
        for (auto& [v, bs] : f.ints) {
            bs.reset(id);
        }
        for (auto& [v, bs] : f.doubles) {
            bs.reset(id);
        }
        for (auto& [v, bs] : f.strs) {
            bs.reset(id);
        }
    }
    all_ids_.reset(id);
}

bool FilterIndex::value_contains(const FieldIndex& f, const ScalarValue& v, uint64_t id) {
    switch (v.index()) {
        case 0: {
            const auto it = f.ints.find(std::get<int64_t>(v));
            return it != f.ints.end() && it->second.test(id);
        }
        case 1: {
            const auto it = f.doubles.find(std::get<double>(v));
            return it != f.doubles.end() && it->second.test(id);
        }
        case 2: {
            const auto it = f.strs.find(std::get<std::string>(v));
            return it != f.strs.end() && it->second.test(id);
        }
    }
    return false;
}

bool FilterIndex::clause_matches(uint64_t id, const Clause& c) const {
    const FieldIndex* f = find_field(c.field);
    if (f == nullptr) {
        return false;  // id without the field does not satisfy an equality filter
    }

    switch (c.op) {
        case FilterOp::kEq:
            return value_contains(*f, c.value, id);
        case FilterOp::kNe:
            // Must have the field set AND not equal the value.
            if (!all_ids_.test(id)) {
                return false;
            }
            return !value_contains(*f, c.value, id);
        case FilterOp::kLt:
        case FilterOp::kLe:
        case FilterOp::kGt:
        case FilterOp::kGe: {
            // Range over numeric values: OR the bitsets of matching values.
            const auto test_num = [&](auto& map, auto thresh, auto cmp) {
                bool any = false;
                for (const auto& [v, bs] : map) {
                    if (cmp(v, thresh) && bs.test(id)) {
                        any = true;
                        break;
                    }
                }
                return any;
            };
            if (std::holds_alternative<int64_t>(c.value)) {
                const int64_t thresh = std::get<int64_t>(c.value);
                switch (c.op) {
                    case FilterOp::kLt: return test_num(f->ints, thresh, [](auto a, auto b) { return a < b; });
                    case FilterOp::kLe: return test_num(f->ints, thresh, [](auto a, auto b) { return a <= b; });
                    case FilterOp::kGt: return test_num(f->ints, thresh, [](auto a, auto b) { return a > b; });
                    case FilterOp::kGe: return test_num(f->ints, thresh, [](auto a, auto b) { return a >= b; });
                    default: return false;
                }
            }
            if (std::holds_alternative<double>(c.value)) {
                const double thresh = std::get<double>(c.value);
                switch (c.op) {
                    case FilterOp::kLt: return test_num(f->doubles, thresh, [](auto a, auto b) { return a < b; });
                    case FilterOp::kLe: return test_num(f->doubles, thresh, [](auto a, auto b) { return a <= b; });
                    case FilterOp::kGt: return test_num(f->doubles, thresh, [](auto a, auto b) { return a > b; });
                    case FilterOp::kGe: return test_num(f->doubles, thresh, [](auto a, auto b) { return a >= b; });
                    default: return false;
                }
            }
            return false;
        }
    }
    return false;
}

bool FilterIndex::matches(uint64_t id, const FilterExpr& expr) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    for (const auto& clause : expr.clauses) {
        if (!clause_matches(id, clause)) {
            return false;
        }
    }
    return true;
}

bool FilterIndex::has_field(const std::string& field) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return fields_.count(field) != 0U;
}

size_t FilterIndex::field_count() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return fields_.size();
}

} // namespace lumina
