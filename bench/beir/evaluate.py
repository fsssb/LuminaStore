#!/usr/bin/env python3
"""Standard IR metrics: NDCG@k, MRR@k, Recall@k.

Ground truth: {qid: {doc_id: rel}} with rel in {0,1} (SciFact uses binary).
Ranks: list of doc ids per query, ordered by engine.
"""

import math


def _dcg(rels):
    dcg = 0.0
    for i, rel in enumerate(rels):
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_ids, relevant, k):
    """relevant: set of relevant doc ids. Returns NDCG@k (0..1)."""
    if not relevant:
        return 0.0
    rels = [1 if doc in relevant else 0 for doc in ranked_ids[:k]]
    dcg = _dcg(rels)
    ideal = sorted((1 for _ in relevant), reverse=True)[:k]
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_ids, relevant, k):
    for i, doc in enumerate(ranked_ids[:k]):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    hit = sum(1 for doc in ranked_ids[:k] if doc in relevant)
    return hit / len(relevant)


def evaluate(results, qrels, k=10):
    """results: {qid: [doc_id,...]}, qrels: {qid: {doc_id: rel}}.
    Returns dict of mean metrics over queries that have qrels."""
    ndcg = mrr = recall = 0.0
    count = 0
    for qid, ranked in results.items():
        if qid not in qrels or not qrels[qid]:
            continue
        relevant = set(qrels[qid].keys())
        ndcg += ndcg_at_k(ranked, relevant, k)
        mrr += mrr_at_k(ranked, relevant, k)
        recall += recall_at_k(ranked, relevant, 100)
        count += 1
    n = max(count, 1)
    return {
        "queries_evaluated": count,
        f"NDCG@{k}": round(ndcg / n, 4),
        f"MRR@{k}": round(mrr / n, 4),
        "Recall@100": round(recall / n, 4),
    }
