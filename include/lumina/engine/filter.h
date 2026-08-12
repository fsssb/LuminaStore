#pragma once

#include "lumina/common/types.h"

#include <string>
#include <vector>

namespace lumina {

// Minimal programmatic filter expression (AND of clauses), used for filtered
// ANN search. FilterIndex (index/filter_index.h) evaluates these as bitmaps.
// Full evaluation is wired in the search pipeline (P6); the types are defined
// here so the Collection API surface is stable.

enum class FilterOp : uint8_t {
    kEq = 0,   // ==
    kNe = 1,   // !=
    kLt = 2,   // <
    kLe = 3,   // <=
    kGt = 4,   // >
    kGe = 5,   // >=
};

struct Clause {
    std::string  field;
    FilterOp     op = FilterOp::kEq;
    ScalarValue  value;   // int64 / double / string
};

struct FilterExpr {
    std::vector<Clause> clauses;  // AND semantics
    bool empty() const { return clauses.empty(); }
};

} // namespace lumina
