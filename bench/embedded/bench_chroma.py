#!/usr/bin/env python3
"""Chroma benchmark (embedded, local persistent mode).

Usage: bench_chroma.py <base.fvecs> <query.fvecs> [--limit N] [--ef E1,..]
Emits one JSON line per ef (Chroma has no ef; the ef values are ignored and a
single line with ef=0 is emitted).
"""

import argparse
import os
import shutil
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
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--tmpdir", default="/tmp/lumina_bench_chroma")
    args = ap.parse_args()

    import chromadb

    base = read_fvecs(args.base, args.limit)
    queries = read_fvecs(args.queries)[: args.nq]
    n, dim = base.shape

    shutil.rmtree(args.tmpdir, ignore_errors=True)
    client = chromadb.PersistentClient(path=args.tmpdir)
    col = client.get_or_create_collection(
        name="bench",
        metadata={"hnsw:space": "l2", "hnsw:M": 32, "hnsw:construction_ef": 400},
    )

    t0 = time.perf_counter()
    # Chroma enforces a max batch size; chunk the adds.
    chunk = 1000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        col.add(
            ids=[str(i) for i in range(start, end)],
            embeddings=base[start:end].tolist(),
            metadatas=[{"i": i} for i in range(start, end)],
        )
    build_s = time.perf_counter() - t0

    truth = brute_truth(base, queries, 10)

    # Per-query latency.
    lat = []
    ids_matrix = np.zeros((queries.shape[0], 10), dtype=np.int64)
    for qi in range(queries.shape[0]):
        t0 = time.perf_counter()
        res = col.query(query_embeddings=[queries[qi].tolist()], n_results=10)
        lat.append((time.perf_counter() - t0) * 1e6)
        ids_matrix[qi] = [int(x) for x in res["ids"][0]]
    rec = recall_at_k(ids_matrix, truth, 10)
    qps = queries.shape[0] / (sum(lat) / 1e6)

    emit(
        engine="Chroma",
        n=n,
        dim=dim,
        ef=0,
        recall=round(rec, 4),
        p50_us=round(percentile(lat, 50), 1),
        p99_us=round(percentile(lat, 99), 1),
        qps=round(qps, 1),
        build_s=round(build_s, 2),
        rss_mb=round(rss_mb(), 1),
    )

    shutil.rmtree(args.tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
