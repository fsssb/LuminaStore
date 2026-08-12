#!/usr/bin/env python3
"""hnswlib benchmark on the same dataset + methodology as tools/ann_bench.

Reads fvecs base/query files, builds hnswlib HNSW (same M/ef_construction),
and reports recall@10 vs QPS across an ef grid, using brute-force ground
truth (same methodology as LuminaStore's ann_bench).

Usage: hnswlib_bench.py <base.fvecs> <query.fvecs> [M] [ef_construction] [ef_list]
"""

import struct
import sys
import time

import hnswlib
import numpy as np


def read_fvecs(path, limit=None):
    with open(path, "rb") as f:
        raw = f.read()
    dim = struct.unpack("<i", raw[:4])[0]
    per = 4 + dim * 4
    n = len(raw) // per
    if limit:
        n = min(n, limit)
    data = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        data[i] = np.frombuffer(raw, dtype=np.float32, count=dim, offset=4 + i * per)
    return data


def brute_truth(base, queries, k=10):
    n, dim = base.shape
    truth = np.zeros((queries.shape[0], k), dtype=np.int64)
    for qi, q in enumerate(queries):
        d = ((base - q) ** 2).sum(axis=1)
        truth[qi] = np.argpartition(d, k)[:k]
    return truth


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    base_path, query_path = sys.argv[1], sys.argv[2]
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    efc = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    efs = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "50,100,200,400,800").split(",")]
    k = 10

    base = read_fvecs(base_path)
    queries = read_fvecs(query_path)
    n, dim = base.shape
    print(f"# hnswlib benchmark: n={n} dim={dim} queries={queries.shape[0]} M={M} efc={efc}")

    t0 = time.perf_counter()
    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=efc, M=M)
    index.add_items(base)
    print(f"build: {time.perf_counter() - t0:.2f}s")

    print("ground truth (brute-force top-10)...")
    t0 = time.perf_counter()
    truth = brute_truth(base, queries[:200], k)
    print(f"truth: {time.perf_counter() - t0:.2f}s")

    print("\n| ef | recall@10 | QPS |")
    print("|----|-----------|-----|")
    for ef in efs:
        index.set_ef(ef)
        t0 = time.perf_counter()
        labels, _ = index.knn_query(queries[:200], k=k)
        elapsed = time.perf_counter() - t0
        qps = 200 / elapsed

        hits = sum(1 for qi in range(200) for j in range(k) if labels[qi][j] in set(truth[qi]))
        recall = hits / (200 * k)
        print(f"| {ef:4d} | {recall:9.4f} | {qps:7.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
