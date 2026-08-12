# LuminaStore Benchmarking

> 方法学遵循 ann-benchmarks / VIBE 的通行做法（单线程、train/test 分离、
> recall@k 对暴力 top-k ground truth、报 p50/p99 延迟）。

## 环境

- 机器：Apple Silicon (arm64)，单线程，Release（-O3）

## 真实数据：MNIST（60k × 784，L2）

像素强度归一化到 [0,1]；索引 M=32、ef_construction=400；200 个测试查询；
recall@10 对暴力 top-10 ground truth。

| ef | recall@10 | QPS | p50 (us) | p99 (us) |
|----|-----------|-----|----------|----------|
|  50 | 0.9710 |  4448.6 | 214.5 | 569.7 |
| 100 | 0.9910 |  3430.2 | 298.0 | 486.3 |
| 200 | 0.9975 |  1920.5 | 541.2 | 793.3 |
| 400 | 0.9990 |  1078.6 | 957.8 | 1307.4 |
| 800 | 0.9995 |   613.2 | 1652.9 | 2472.5 |

**与 hnswlib 同机对比**（同一数据、同一方法学，M=32/efc=400）：

| ef | LuminaStore recall | LuminaStore QPS | hnswlib recall | hnswlib QPS |
|----|--------------------|-----------------|----------------|-------------|
|  50 | 0.9710 | 4448.6 | 0.9945 | 4682.3 |
| 100 | 0.9910 | 3430.2 | 0.9985 | 5148.2 |
| 200 | 0.9975 | 1920.5 | 0.9995 | 3041.1 |
| 400 | 0.9990 | 1078.6 | 1.0000 | 1951.9 |
| 800 | 0.9995 |  613.2 | 1.0000 | 1153.3 |

结论：recall 差距 ≤ 2.4pp（低 ef 时），QPS 同一数量级（低 10%-2x）。
hnswlib 在低 ef 图质量与搜索热路径（无锁、紧凑布局）上更优；
LuminaStore 的差距集中在构建细节与搜索锁开销，可作为后续优化方向。

复现：
```bash
./build/ann_bench --data data/mnist/base.fvecs --queries data/mnist/query.fvecs --M 32 --efc 400 --k 10 --ef "50,100,200,400,800"
python3 scripts/hnswlib_bench.py data/mnist/base.fvecs data/mnist/query.fvecs 32 400 "50,100,200,400,800"
```

## 随机数据对照（100k × 128d）

随机均匀数据无聚类结构（ANN 地狱难度），recall 上限显著低于真实数据；
用于验证方法学与调参趋势，不用于绝对数字。

| ef | recall@10 | QPS |
|----|-----------|-----|
|  50 | 0.3195 | 7978.4 |
| 100 | 0.4675 | 4189.0 |
| 200 | 0.6140 | 2458.5 |
| 400 | 0.7695 | 1355.9 |
| 800 | 0.8845 |  755.0 |

## 量化基准（bench/quant_bench）

| 距离方式 | 吞吐（距离/秒） | 内存/向量 | recall@10 vs 精确 |
|---|---|---|---|
| 全精度 L2 | 67.0M | 512 B | 1.0 |
| SQ8 | 24.4M | 128 B (4x) | 1.0 |
| Binary (Hamming) | 30.3M | 16 B (32x) | 1.0 |
| PQ (m=16, k=256) | 24.9M | 16 B (32x) | 1.0 |

> 注：SQ8/PQ 的吞吐低于全精度是因为当前实现是反量化/查表标量路径；
> 真正的速度收益需要 SIMD int8/byte 内核（下一步工作）。量化的核心价值
> 在**内存压缩**（4x-32x）且保持召回。

## 过滤基准（bench/filter_bench）

10k 向量，cat 字段（1% / 20% / 50% 选择性），recall@10 vs 过滤后暴力 top-k：

| 模式 | 1% 选择性 | 20% | 50% |
|---|---|---|---|
| in-filter | recall 1.0 | 1.0 | 1.0 |
| post-filter | recall 1.0 | 1.0 | 1.0 |

in-filter 在低选择性下更快（无需放大 ef 重试）；两者 recall 相同（ef 足够时）。

## 与外部库对比（待接入）

- hnswlib（同机、同数据集）：
- FAISS IndexHNSWFlat（同机、同数据集）：

> 网络受限时可用 `tools/ann_bench` 生成的随机数据做同机对比，方法学一致。

## 复现命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build                  # 全部单元测试
./build/ann_bench 100000 128 200 10 "10,50,100,200,400"   # recall-QPS 表
./build/quant_bench                     # 量化基准
./build/filter_bench                    # 过滤基准
```
