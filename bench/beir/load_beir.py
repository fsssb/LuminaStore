#!/usr/bin/env python3
"""Load BEIR SciFact dataset (corpus / queries / qrels)."""

import json
import os


def load_corpus(data_dir):
    """Returns list of (doc_id, title, text)."""
    docs = []
    with open(os.path.join(data_dir, "corpus.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            docs.append((d["_id"], d.get("title", ""), d.get("text", "")))
    return docs


def load_queries(data_dir):
    """Returns list of (qid, text)."""
    queries = []
    with open(os.path.join(data_dir, "queries.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            queries.append((q["_id"], q.get("text", "")))
    return queries


def load_qrels(data_dir):
    """Returns {qid: {doc_id: rel}} from the test split (standard BEIR eval)."""
    qrels = {}
    path = os.path.join(data_dir, "qrels", "test.tsv")
    if not os.path.exists(path):
        path = os.path.join(data_dir, "qrels.tsv")
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, rel = parts[0], parts[1], int(parts[2])
            if rel <= 0:
                continue
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels
