#!/usr/bin/env python3
"""Qdrant benchmark (server-side via docker + qdrant-client).

The qdrant container must be running (see run_qdrant.sh). Latency is the full
client round-trip over localhost, which is exactly the "server vs embedded"
form-factor difference we want to show.

Usage: bench_qdrant.py <base.fvecs> <query.fvecs> [--limit N] [--ef E1,..]
Emits one JSON line per ef.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import brute_truth, emit, percentile, read_fvecs, recall_at_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("queries")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ef", default="50,100,200,400")
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--url", default="http://localhost:6333")
    args = ap.parse_args()

    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm

    efs = [int(x) for x in args.ef.split(",")]
    base = read_fvecs(args.base, args.limit)
    queries = read_fvecs(args.queries)[: args.nq]
    n, dim = base.shape

    client = QdrantClient(url=args.url)
    client.delete_collection("bench")
    client.create_collection(
        collection_name="bench",
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.EUCLID),
        hnsw_config=qm.HnswConfigDiff(m=32, ef_construct=400),
    )

    # Batch upsert in chunks of 1000.
    t0 = time.perf_counter()
    for start in range(0, n, 1000):
        end = min(start + 1000, n)
        client.upsert(
            collection_name="bench",
            points=[
                qm.PointStruct(id=int(i), vector=base[i].tolist())
                for i in range(start, end)
            ],
        )
    build_s = time.perf_counter() - t0

    truth = brute_truth(base, queries, 10)

    for ef in efs:
        lat = []
        ids_matrix = np.zeros((queries.shape[0], 10), dtype=np.int64)
        for qi in range(queries.shape[0]):
            t0 = time.perf_counter()
            res = client.query_points(
                collection_name="bench",
                query=queries[qi].tolist(),
                limit=10,
                search_params=qm.SearchParams(hnsw_ef=ef),
            )
            lat.append((time.perf_counter() - t0) * 1e6)
            ids_matrix[qi] = [p.id for p in res.points]
        rec = recall_at_k(ids_matrix, truth, 10)
        qps = queries.shape[0] / (sum(lat) / 1e6)

        emit(
            engine="Qdrant",
            n=n,
            dim=dim,
            ef=ef,
            recall=round(rec, 4),
            p50_us=round(percentile(lat, 50), 1),
            p99_us=round(percentile(lat, 99), 1),
            qps=round(qps, 1),
            build_s=round(build_s, 2),
            rss_mb=0.0,  # server process memory, not measurable from client
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
