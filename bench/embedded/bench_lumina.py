#!/usr/bin/env python3
"""LuminaStore benchmark (embedded, via the v2 C API + ctypes).

Usage: bench_lumina.py <base.fvecs> <query.fvecs> [--limit N] [--ef E1,E2,..]
                       [--M 32] [--efc 400] [--tmpdir /tmp/lumina_bench]
Emits one JSON line per ef with: engine, ef, recall, p50, p99, qps, build_s, rss_mb.
"""

import argparse
import ctypes
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import brute_truth, emit, percentile, read_fvecs, recall_at_k, rss_mb

_LIB = None


def load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))  # bench/embedded -> repo
    dylib = os.path.join(repo, "build", "libluminastore_shared.dylib")
    lib = ctypes.CDLL(dylib)
    lib.lumina_open.restype = ctypes.c_void_p
    lib.lumina_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.lumina_close.restype = ctypes.c_int
    lib.lumina_close.argtypes = [ctypes.c_void_p]
    lib.lumina_add_batch.restype = ctypes.c_int
    lib.lumina_add_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.lumina_search_batch.restype = ctypes.c_void_p
    lib.lumina_search_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.lumina_search.restype = ctypes.c_void_p
    lib.lumina_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.lumina_free_string.restype = None
    lib.lumina_free_string.argtypes = [ctypes.c_void_p]
    _LIB = lib
    return lib


def _search_one(lib, h, qptr, dim, top_k, ef):
    """Return (id list, latency seconds)."""
    t0 = time.perf_counter()
    raw = lib.lumina_search(h, qptr, dim, top_k, ef)
    lat = time.perf_counter() - t0
    import json

    hits = json.loads(ctypes.string_at(raw).decode())["results"]
    lib.lumina_free_string(raw)
    return [r["id"] for r in hits], lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("queries")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ef", default="50,100,200,400")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efc", type=int, default=400)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--tmpdir", default="/tmp/lumina_bench_embedded")
    args = ap.parse_args()

    efs = [int(x) for x in args.ef.split(",")]
    base = read_fvecs(args.base, args.limit)
    queries = read_fvecs(args.queries)[: args.nq]
    n, dim = base.shape

    shutil.rmtree(args.tmpdir, ignore_errors=True)
    lib = load_lib()
    h = lib.lumina_open(args.tmpdir.encode(), dim, 0)
    if not h:
        print("lumina_open failed", file=sys.stderr)
        return 1

    ids = np.arange(n, dtype=np.uint64)
    cids = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    cvec = base.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    payloads = (ctypes.c_char_p * n)(*([b"p"] * n))

    t0 = time.perf_counter()
    rc = lib.lumina_add_batch(h, cids, cvec, n, dim, payloads)
    build_s = time.perf_counter() - t0
    assert rc == 0, f"add_batch rc={rc}"

    # Compute ground truth only after all latency measurements: the float64
    # scratch arrays would otherwise skew the first rows (memory pressure).
    truth = brute_truth(base, queries, 10)

    # Compute ground truth only after all latency measurements: the float64
    # scratch arrays would otherwise skew the first rows (memory pressure).
    truth = brute_truth(base, queries, 10)

    for ef in efs:
        # Warm up the whole query set for this ef: the first execution of each
        # query (ctypes/JSON cold path) skews the first measured row.
        for qi in range(queries.shape[0]):
            wqp = queries[qi : qi + 1].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _search_one(lib, h, wqp, dim, 10, ef)

        lat = []
        ids_matrix = np.zeros((queries.shape[0], 10), dtype=np.int64)
        for qi in range(queries.shape[0]):
            qptr = queries[qi : qi + 1].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ids_row, t = _search_one(lib, h, qptr, dim, 10, ef)
            lat.append(t * 1e6)
            ids_matrix[qi] = ids_row
        rec = recall_at_k(ids_matrix, truth, 10)
        qps = queries.shape[0] / (sum(lat) / 1e6)

        emit(
            engine="LuminaStore",
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

    lib.lumina_close(h)
    shutil.rmtree(args.tmpdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
