#!/usr/bin/env python3
"""Plot recall@10 vs QPS frontier from ann_bench results.

Data source: `./build/ann_bench 100000 128 200 10 "50,100,200,400,800" 32 400`
(M=32, ef_construction=400). Output: docs/recall_qps.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (ef, recall@10, qps)
POINTS = [
    (50, 0.3195, 7978.4),
    (100, 0.4675, 4189.0),
    (200, 0.6140, 2458.5),
    (400, 0.7695, 1355.9),
    (800, 0.8845, 755.0),
]

def main():
    recalls = [p[1] for p in POINTS]
    qps = [p[2] for p in POINTS]
    efs = [p[0] for p in POINTS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(recalls, qps, "o-", color="#1f77b4", linewidth=2, markersize=7)
    for ef, r, q in POINTS:
        ax.annotate(f"ef={ef}", (r, q), textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_xlabel("recall@10")
    ax.set_ylabel("QPS (log)")
    ax.set_yscale("log")
    ax.set_title("LuminaStore HNSW: recall@10 vs QPS\n100k x 128d random uniform, M=32, ef_construction=400")
    ax.grid(True, which="both", alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "..", "docs", "recall_qps.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(os.path.abspath(out), dpi=150, bbox_inches="tight")
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
