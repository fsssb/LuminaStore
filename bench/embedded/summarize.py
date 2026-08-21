#!/usr/bin/env python3
"""Summarize per-engine JSON lines into a markdown comparison table.

Usage: summarize.py <results.jsonl> [--out report.md]
The JSONL is produced by concatenating the outputs of the four bench_*.py
scripts (one JSON line per ef). Prints a markdown table of p50/p99 latency,
QPS, recall, build time and RSS.
"""

import argparse
import json
import sys


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load(args.jsonl)
    engines = ["LuminaStore", "Chroma", "usearch", "Qdrant"]

    lines = []
    lines.append("| engine | ef | recall@10 | p50 (us) | p99 (us) | QPS | build (s) | RSS (MB) |")
    lines.append("|--------|----|-----------|----------|----------|-----|-----------|----------|")
    for eng in engines:
        for r in sorted([x for x in rows if x["engine"] == eng], key=lambda x: x.get("ef", 0)):
            lines.append(
                f"| {r['engine']} | {r.get('ef', 0)} | {r['recall']} | {r['p50_us']} | "
                f"{r['p99_us']} | {r['qps']} | {r['build_s']} | {r['rss_mb']} |"
            )

    # Recall-aligned latency comparison (best recall point per engine).
    lines.append("\n### Recall-aligned latency (best recall >= 0.95, else best available)\n")
    lines.append("| engine | recall@10 | p50 (us) | p99 (us) | QPS |")
    lines.append("|--------|-----------|----------|----------|-----|")
    for eng in engines:
        candidates = [x for x in rows if x["engine"] == eng]
        if not candidates:
            continue
        ok = [x for x in candidates if x["recall"] >= 0.95]
        pick = min(ok, key=lambda x: x["p50_us"]) if ok else min(candidates, key=lambda x: x["p50_us"])
        lines.append(
            f"| {pick['engine']} | {pick['recall']} | {pick['p50_us']} | {pick['p99_us']} | {pick['qps']} |"
        )

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
