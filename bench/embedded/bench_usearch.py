#!/usr/bin/env python3
"""usearch benchmark (embedded library, SIMD).

Usage: bench_usearch.py <base.fvecs> <query.fvecs> [--limit N] [--ef E1,..]
Emits one JSON line per ef.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import brute_truth, emit, percentile, read_fvecs, recall_at_k, rss_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("queries")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ef", default="50,100,200,400")
    ap.add_argument("--nq", type=int, default=200)
    args = ap.parse_args()

    from usearch.index import Index

    efs = [int(x) for x in args.ef.split(",")]
    base = read_fvecs(args.base, args.limit)
    queries = read_fvecs(args.queries)[: args.nq]
    n, dim = base.shape

    index = Index(ndim=dim, metric="l2sq", connectivity=32, expansion_add=400)
    ids = np.arange(n, dtype=np.uint64)

    t0 = time.perf_counter()
    index.add(ids, base)
    build_s = time.perf_counter() - t0

    truth = brute_truth(base, queries, 10)

    for ef in efs:
        index.expansion_search = ef
        lat = []
        ids_matrix = np.zeros((queries.shape[0], 10), dtype=np.int64)
        for qi in range(queries.shape[0]):
            t0 = time.perf_counter()
            res = index.search(queries[qi], 10)
            keys = np.array([m.key for m in res], dtype=np.int64)
            lat.append((time.perf_counter() - t0) * 1e6)
            ids_matrix[qi] = keys
        rec = recall_at_k(ids_matrix, truth, 10)
        qps = queries.shape[0] / (sum(lat) / 1e6)

        emit(
            engine="usearch",
            n=n,
            dim=dim,
            ef=ef,
            recall=round(rec, 4),
            p50_us=round(percentile(lat, 50), 1),
            p99_us=round(percentile(lat, 99), 1),
            qps=round(qps, 1),
            build_s=round(build_s, 2),
            rss_mb=round(rss_mb(), 1),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
