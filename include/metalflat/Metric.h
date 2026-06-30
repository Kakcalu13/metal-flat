// SPDX-License-Identifier: Apache-2.0
// metalflat/include/metalflat/Metric.h
//
// Distance / similarity metrics for flat (exact) search.

#pragma once

namespace mflat {

enum class Metric {
    // Squared Euclidean distance. Smaller = nearer. (Squared — no sqrt;
    // ranking is identical and callers rarely need the true distance.)
    L2,
    // Dot product. Larger = nearer. Vectors are used as-is.
    InnerProduct,
    // Cosine similarity. Larger = nearer. The index stores L2-normalized
    // copies of added vectors and normalizes queries at search time, so
    // this is InnerProduct over unit vectors.
    Cosine,
};

}  // namespace mflat
