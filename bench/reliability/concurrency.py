#!/usr/bin/env python3
"""Concurrency stress: N writer threads add distinct ids, M reader threads
search concurrently; then verify the graph is consistent (all ids present,
recall still high for known queries).

Usage: concurrency.py [--adds N] [--dim D] [--writers W] [--readers R]
"""

import argparse
import ctypes
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embedded"))
from bench_lumina import load_lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adds", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--writers", type=int, default=4)
    ap.add_argument("--readers", type=int, default=4)
    ap.add_argument("--dir", default="/tmp/lumina_conc_db")
    args = ap.parse_args()

    import shutil

    shutil.rmtree(args.dir, ignore_errors=True)
    lib = load_lib()
    h = lib.lumina_open(args.dir.encode(), args.dim, 0)
    assert h

    n = args.adds
    rng = np.random.default_rng(7)
    vecs = rng.uniform(-1, 1, (n, args.dim)).astype(np.float32)

    # Writers: each thread adds a disjoint slice, one vector at a time.
    errors = []
    lock = threading.Lock()

    def writer(tid):
        try:
            for i in range(tid, n, args.writers):
                one = vecs[i : i + 1]
                qp = one.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                ids = np.array([i], dtype=np.uint64)
                payloads = (ctypes.c_char_p * 1)(b"p")
                rc = lib.lumina_add_batch(
                    h,
                    ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    qp,
                    1,
                    args.dim,
                    payloads,
                )
                if rc != 0:
                    with lock:
                        errors.append((tid, i, rc))
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append((tid, -1, str(e)))

    def reader(tid):
        r = np.random.default_rng(100 + tid)
        # warmup
        for _ in range(50):
            q = rng.uniform(-1, 1, (1, args.dim)).astype(np.float32)
            qp = q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            raw = lib.lumina_search(h, qp, args.dim, 5, 50)
            lib.lumina_free_string(raw)

    wthreads = [threading.Thread(target=writer, args=(t,)) for t in range(args.writers)]
    rthreads = [threading.Thread(target=reader, args=(t,)) for t in range(args.readers)]
    for t in rthreads:
        t.start()
    for t in wthreads:
        t.start()
    for t in wthreads + rthreads:
        t.join()

    if errors:
        print(f"WRITER ERRORS: {len(errors)} e.g. {errors[:3]}")
        return 1

    # Verify: stats shows all ids.
    raw = lib.lumina_stats(h)
    stats = json.loads(ctypes.string_at(raw).decode())
    lib.lumina_free_string(raw)
    live = stats.get("live_entries", -1)
    print(f"concurrency: live={live} expected={n}")

    # Search recall on known vectors.
    hits_total = 0
    for i in range(0, min(n, 200), 3):
        q = vecs[i : i + 1]
        qp = q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        raw = lib.lumina_search(h, qp, args.dim, 1, 100)
        hits = json.loads(ctypes.string_at(raw).decode())["results"]
        lib.lumina_free_string(raw)
        if hits and hits[0]["id"] == i:
            hits_total += 1
    recall = hits_total / max(len(list(range(0, min(n, 200), 3))), 1)
    print(f"known-vector recall@1: {recall:.3f}")

    lib.lumina_close(h)
    ok = live == n and recall > 0.9 and not errors
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
