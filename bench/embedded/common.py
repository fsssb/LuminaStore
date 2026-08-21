# Shared methodology for the embedded-scenario comparison.
# Unified: fvecs data, brute-force ground truth, recall@10, p50/p99 latency,
# process RSS. Every engine script imports this and emits the same JSON schema.

import json
import resource
import struct
import time

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
    """Exact top-k indices per query (L2), computed in float64 for precision."""
    base64 = base.astype(np.float64)
    truth = np.zeros((queries.shape[0], k), dtype=np.int64)
    for qi, q in enumerate(queries):
        d = ((base64 - q.astype(np.float64)) ** 2).sum(axis=1)
        truth[qi] = np.argpartition(d, k)[:k]
    return truth


def recall_at_k(hits_ids, truth, k=10):
    """hits_ids: (nq, k) array of ids. Fraction of truth ids found."""
    total = 0
    hit = 0
    for qi in range(truth.shape[0]):
        ids = set(hits_ids[qi].tolist())
        for j in range(k):
            total += 1
            hit += truth[qi][j] in ids
    return hit / total if total else 0.0


def percentile(arr, p):
    return float(np.percentile(np.asarray(arr, dtype=np.float64), p))


def rss_mb():
    """Current process max RSS in MB (macOS ru_maxrss is in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def measure(fn, n):
    """Run fn n times, return list of latencies (us)."""
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1e6)
    return lat


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False, sort_keys=True))
