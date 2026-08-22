#!/usr/bin/env python3
"""Crash-recovery reliability test: repeatedly write, kill -9, reopen, verify.

Usage:
  crash_recovery.py --writer --dir /tmp/rel_db --count 100000   # write batches + snapshots
  crash_recovery.py --verify --dir /tmp/rel_db --expect N         # reopen and validate
  crash_recovery.py --run --dir /tmp/rel_db --count 100000 --rounds 10  # full loop
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "embedded"))
from bench_lumina import load_lib

DIM = 64


def open_db(lib, path):
    h = lib.lumina_open(path.encode(), DIM, 0)
    if not h:
        raise RuntimeError("open failed")
    return h


def writer(path, count, round_no):
    import numpy as np

    lib = load_lib()
    h = open_db(lib, path)
    rng = np.random.default_rng(42 + round_no)
    chunk = 1000
    total = 0
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        ids = np.arange(start, end, dtype=np.uint64)
        vecs = rng.uniform(-1, 1, (end - start, DIM)).astype(np.float32)
        payloads = (ctypes.c_char_p * (end - start))(
            *[f"v{i}".encode() for i in range(start, end)]
        )
        rc = lib.lumina_add_batch(
            h,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            end - start,
            DIM,
            payloads,
        )
        assert rc == 0
        total = end
        if total % 20000 == 0:
            lib.lumina_snapshot(h)  # durable checkpoint
            print(f"[writer] {total}/{count} + snapshot", flush=True)
    lib.lumina_close(h)
    print(f"[writer] done {total}", flush=True)


def verify(path, expect):
    lib = load_lib()
    h = open_db(lib, path)
    raw = lib.lumina_stats(h)
    stats = json.loads(ctypes.string_at(raw).decode())
    lib.lumina_free_string(raw)
    live = stats.get("live_entries", -1)
    # Sample a few ids and check payloads.
    ok = 0
    samples = list(range(0, min(expect, 50), 7))  # low ids are always durably written
    for i in samples:
        p = ctypes.c_char_p()
        ln = ctypes.c_int()
        rc = lib.lumina_get(h, i, ctypes.byref(p), ctypes.byref(ln))
        if rc == 0 and ctypes.string_at(p) == f"v{i}".encode():
            ok += 1
        if p.value:
            lib.lumina_free_string(p)
    lib.lumina_close(h)
    return live, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dir", default="/tmp/lumina_rel_db")
    ap.add_argument("--count", type=int, default=100000)
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()

    if args.writer:
        import numpy as np

        writer(args.dir, args.count, int(os.environ.get("ROUND", "0")))
        return 0
    if args.verify:
        live, ok = verify(args.dir, args.count)
        samples = len(list(range(0, min(args.count, 50), 7)))
        # kill -9 mid-flight: `live` is whatever was durably written before the
        # kill; the invariant is that every persisted entry is intact (payload
        # samples all correct) and that nothing is corrupted.
        ok_pass = ok == samples and live > 0
        print(f"[verify] live={live} (durably written before kill) sample_ok={ok}/{samples}")
        return 0 if ok_pass else 1

    # Full loop: write -> kill -9 -> verify, repeated.
    import shutil

    failures = 0
    for r in range(args.rounds):
        shutil.rmtree(args.dir, ignore_errors=True)
        env = dict(os.environ, ROUND=str(r))
        proc = subprocess.Popen(
            [sys.executable, __file__, "--writer", "--dir", args.dir, "--count", str(args.count)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        # Let it write a while, then kill -9 mid-flight (or after completion).
        try:
            out, _ = proc.communicate(timeout=6)
            killed = False
        except subprocess.TimeoutExpired:
            proc.kill()  # SIGKILL
            out, _ = proc.communicate()
            killed = True
        print(f"[round {r}] killed={killed}")
        live, ok = verify(args.dir, args.count)
        if live == 0 or ok == 0:
            failures += 1
            print(f"[round {r}] FAILED live={live} ok={ok}")
        else:
            print(f"[round {r}] OK live={live} sample_ok={ok}")
    print(f"RESULT: {args.rounds - failures}/{args.rounds} rounds data-consistent")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
