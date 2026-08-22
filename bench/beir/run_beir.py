#!/usr/bin/env python3
"""BEIR SciFact scenario evaluation: LuminaStore vs Chroma.

Both engines use the same BGE embeddings, same corpus, same queries, and are
evaluated with the same IR metrics (NDCG@10, MRR@10, Recall@100) against the
same qrels. Only the vector store differs.

Usage: run_beir.py --engine lumina|chroma [--limit N] [--ef E]
Model dir: bench/beir/model (bge-small-en-v1.5, local)
Data dir:  bench/beir/data/scifact
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate
from load_beir import load_corpus, load_qrels, load_queries

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "scifact")
MODEL = os.path.join(BASE, "model")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed(model, texts, batch=128):
    """BGE query prefix for queries, none for corpus (BGE docs)."""
    out = []
    for i in range(0, len(texts), batch):
        out.append(model.encode(texts[i : i + batch], normalize_embeddings=True))
    return np.concatenate(out, axis=0).astype(np.float32)


def index_lumina(vecs, dim):
    import ctypes
    import shutil

    sys.path.insert(0, os.path.join(BASE, "..", "embedded"))
    from bench_lumina import load_lib

    tmp = "/tmp/lumina_beir"
    shutil.rmtree(tmp, ignore_errors=True)
    lib = load_lib()
    h = lib.lumina_open(tmp.encode(), dim, 0)
    ids = np.arange(len(vecs), dtype=np.uint64)
    payloads = (ctypes.c_char_p * len(vecs))(*([b"p"] * len(vecs)))
    t0 = time.perf_counter()
    rc = lib.lumina_add_batch(
        h,
        ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        vecs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        len(vecs),
        dim,
        payloads,
    )
    build_s = time.perf_counter() - t0
    assert rc == 0
    return lib, h, build_s


def search_lumina(lib, h, qvecs, dim, top_k, ef):
    import ctypes
    import json

    results = {}
    lat = []
    for i in range(len(qvecs)):
        qp = qvecs[i : i + 1].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        t0 = time.perf_counter()
        raw = lib.lumina_search(h, qp, dim, top_k, ef)
        lat.append((time.perf_counter() - t0) * 1e6)
        hits = json.loads(ctypes.string_at(raw).decode())["results"]
        lib.lumina_free_string(raw)
        results[str(qids[i])] = [str(h["id"]) for h in hits]
    return results, lat


def index_chroma(vecs):
    import shutil

    import chromadb

    tmp = "/tmp/lumina_beir_chroma"
    shutil.rmtree(tmp, ignore_errors=True)
    client = chromadb.PersistentClient(path=tmp)
    col = client.get_or_create_collection(
        name="beir",
        metadata={"hnsw:space": "cosine", "hnsw:M": 32, "hnsw:construction_ef": 400},
    )
    chunk = 1000
    t0 = time.perf_counter()
    for start in range(0, len(vecs), chunk):
        end = min(start + chunk, len(vecs))
        col.add(
            ids=[str(i) for i in range(start, end)],
            embeddings=vecs[start:end].tolist(),
        )
    build_s = time.perf_counter() - t0
    return col, build_s


def search_chroma(col, qvecs, qids, top_k):
    results = {}
    lat = []
    for i in range(len(qvecs)):
        t0 = time.perf_counter()
        res = col.query(query_embeddings=[qvecs[i].tolist()], n_results=top_k)
        lat.append((time.perf_counter() - t0) * 1e6)
        results[str(qids[i])] = res["ids"][0]
    return results, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["lumina", "chroma"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ef", type=int, default=200)
    args = ap.parse_args()

    import os

    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 4))

    from sentence_transformers import SentenceTransformer

    docs = load_corpus(DATA)
    queries = load_queries(DATA)
    qrels = load_qrels(DATA)
    print(f"corpus={len(docs)} queries={len(queries)} qrels_queries={len(qrels)}")

    doc_ids = [d[0] for d in docs]
    doc_texts = [f"{d[1]}. {d[2]}" for d in docs]
    # Only evaluate queries that have ground truth (test split).
    test_qids = set(qrels.keys())
    kept = [(qid, text) for qid, text in queries if qid in test_qids]
    global qids
    qids = [q[0] for q in kept]
    qtexts = [QUERY_PREFIX + q[1] for q in kept]
    print(f"test queries with qrels: {len(kept)}")
    if args.limit:
        doc_texts = doc_texts[: args.limit]
        doc_ids = doc_ids[: args.limit]

    cache_dir = os.path.join(BASE, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    doc_cache = os.path.join(cache_dir, "doc_vecs.npy")
    q_cache = os.path.join(cache_dir, "q_vecs.npy")
    if os.path.exists(doc_cache) and os.path.exists(q_cache):
        doc_vecs = np.load(doc_cache)
        q_vecs = np.load(q_cache)
        print(f"embedding: loaded from cache dim={doc_vecs.shape[1]}")
    else:
        model = SentenceTransformer(MODEL)
        t0 = time.perf_counter()
        doc_vecs = embed(model, doc_texts)
        q_vecs = embed(model, qtexts)
        print(f"embedding: {time.perf_counter() - t0:.1f}s dim={doc_vecs.shape[1]}")
        np.save(doc_cache, doc_vecs)
        np.save(q_cache, q_vecs)

    if args.engine == "lumina":
        lib, h, build_s = index_lumina(doc_vecs, doc_vecs.shape[1])
        results, lat = search_lumina(lib, h, q_vecs, doc_vecs.shape[1], 100, args.ef)
        # Engine ids are sequential indices; map back to the corpus doc ids.
        results = {qid: [doc_ids[int(seq)] for seq in ranked] for qid, ranked in results.items()}
    else:
        col, build_s = index_chroma(doc_vecs)
        results, lat = search_chroma(col, q_vecs, qids, 100)
        results = {qid: [doc_ids[int(seq)] for seq in ranked] for qid, ranked in results.items()}

    lat = np.array(lat)
    metrics = evaluate(results, qrels, k=10)
    metrics.update(
        {
            "engine": args.engine,
            "build_s": round(build_s, 2),
            "search_p50_us": round(float(np.percentile(lat, 50)), 1),
            "search_p95_us": round(float(np.percentile(lat, 95)), 1),
            "search_p99_us": round(float(np.percentile(lat, 99)), 1),
        }
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
