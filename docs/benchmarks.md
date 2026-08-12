# LuminaStore Benchmarking

> 方法学遵循 ann-benchmarks / VIBE 的通行做法（单线程、train/test 分离、
> recall@k 对暴力 top-k ground truth、报 p50/p99 延迟）。

## 环境

- 机器：Apple Silicon (arm64)，单线程，Release（-O3）
- 数据：随机均匀向量（100k × 128d, L2）作为 SIFT-1M 的替身数据集；
  绝对数值与 SIFT-1M 不同，但**相对趋势与方法学可复现**。
  （高维随机数据无聚类结构，召回上限低于真实 embedding 数据；SIFT-1M
  实测待接入。）

## 最近一次结果（100k × 128d, 200 查询, recall@10）

索引配置 M=32, ef_construction=400：

| ef | recall@10 | QPS | p50 (us) | p99 (us) |
|----|-----------|-----|----------|----------|
|  50 | 0.3195 |  7978.4 | 126.1 | 188.4 |
| 100 | 0.4675 |  4189.0 | 241.6 | 369.8 |
| 200 | 0.6140 |  2458.5 | 407.5 | 568.6 |
| 400 | 0.7695 |  1355.9 | 746.4 | 913.1 |
| 800 | 0.8845 |   755.0 | 1332.0 | 1553.6 |

M=16, ef_construction=200 对照（构建更快，召回较低）：

| ef | recall@10 | QPS |
|----|-----------|-----|
| 200 | 0.4415 | 4020.3 |
| 400 | 0.5750 | 1988.9 |

> 随机均匀数据下召回与 QPS 的权衡曲线完整；真实 embedding 数据
> （如 SIFT/GloVe）召回通常显著更高，待网络恢复后接入 ann-benchmarks
> 数据集与 hnswlib/FAISS 同机对比。

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
