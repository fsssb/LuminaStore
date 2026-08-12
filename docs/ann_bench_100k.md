# LuminaStore ANN benchmark
n=100000 dim=128 queries=200 k=10 metric=L2

build time: 169120.6 ms (591 vectors/s)

ground truth: 372.0 ms

| ef | recall@10 | QPS | p50 (us) | p99 (us) |
|----|-----------|-----|----------|----------|
|   50 |    0.3245 |  5072.7 |    189.2 |    425.9 |
|  100 |    0.4675 |  4189.0 |    241.6 |    369.8 |
|  200 |    0.6140 |  2458.5 |    407.5 |    568.6 |
|  400 |    0.7695 |  1355.9 |    746.4 |    913.1 |
|  800 |    0.8845 |   755.0 |   1332.0 |   1553.6 |

# Notes
- Single thread, single CPU (no hyper-threading).
- recall@10 computed against brute-force top-10 ground truth.
- Random uniform data: relative ranking is representative; absolute numbers
  differ from SIFT-1M. Methodology follows ann-benchmarks/VIBE practices.
