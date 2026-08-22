#!/usr/bin/env python3
"""Embed SciFact corpus + queries with BGE via plain transformers (CLS pooling).

Fallback path that avoids sentence-transformers (which stalls on this host
when probing the network). Verified: transformers AutoModel loads in ~0.2s and
encodes in ~0.5s locally.

Usage: embed_transformers.py   # writes bench/beir/cache/{doc,q}_vecs.npy
"""

import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from load_beir import load_corpus, load_queries, load_qrels

MODEL = os.path.join(BASE, "model")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_texts(texts, batch=64):
    import torch

    torch.set_num_threads(6)
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True)
    model.eval()

    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            inp = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            out = model(**inp).last_hidden_state  # (B, L, 384)
            cls = out[:, 0, :]  # CLS pooling (bge)
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            outs.append(cls.numpy().astype(np.float32))
            print(f"  [{i}:{i + len(chunk)}] done", flush=True)
    return np.concatenate(outs, axis=0)


def main():
    docs = load_corpus(os.path.join(BASE, "data", "scifact"))
    queries = load_queries(os.path.join(BASE, "data", "scifact"))
    qrels = load_qrels(os.path.join(BASE, "data", "scifact"))
    doc_texts = [f"{d[1]}. {d[2]}" for d in docs]
    # Only embed the test queries that have ground truth (matches run_beir).
    test_qids = set(qrels.keys())
    kept = [(qid, text) for qid, text in queries if qid in test_qids]
    q_texts = [QUERY_PREFIX + q[1] for q in kept]
    print(f"corpus={len(doc_texts)} test_queries={len(kept)}", flush=True)

    os.makedirs(os.path.join(BASE, "cache"), exist_ok=True)
    t0 = time.perf_counter()
    dv = embed_texts(doc_texts)
    print(f"corpus embed: {time.perf_counter() - t0:.1f}s shape={dv.shape}", flush=True)
    t0 = time.perf_counter()
    qv = embed_texts(q_texts)
    print(f"query embed: {time.perf_counter() - t0:.1f}s shape={qv.shape}", flush=True)

    np.save(os.path.join(BASE, "cache", "doc_vecs.npy"), dv)
    np.save(os.path.join(BASE, "cache", "q_vecs.npy"), qv)
    print("SAVED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
