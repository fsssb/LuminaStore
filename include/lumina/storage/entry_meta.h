#pragma once

#include "lumina/common/types.h"

#include <string>
#include <vector>

namespace lumina {

// On-disk encoding of a vector entry's metadata (value payload of WAL OpType::kVectorPutV2).
//
// Layout (little-endian, engine-internal):
//   [4B flags]            bit0: vec present, bit1: payload present, bit2: scalars present
//   [4B vec_len][vec float32 x vec_len]        // only if flags.bit0
//   [4B payload_len][payload bytes]            // only if flags.bit1
//   [2B nscalars]
//     per scalar: [1B type][2B key_len][key bytes][value]
//                 type 0 = int64   -> [8B LE]
//                 type 1 = double  -> [8B LE]
//                 type 2 = string  -> [4B len LE][bytes]
//
// All multi-byte fields are little-endian. Scalars are stored sorted by name for
// deterministic encoding. decode is strict: it validates every length and requires
// the whole buffer to be consumed; any inconsistency returns Status::Corruption.

struct EntryMeta {
    std::vector<float>    vec;      // raw vector, length must equal vector_dim
    std::string           payload;  // opaque user payload
    std::vector<ScalarField> scalars; // filter fields
};

Status encode_entry_meta(const EntryMeta& meta, std::string* out);

Status decode_entry_meta(const std::string& bytes, EntryMeta* out);

} // namespace lumina
