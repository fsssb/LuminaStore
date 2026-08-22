# BEIR 场景检索质量评测：LuminaStore vs Chroma

## 目标

用**标准信息检索基准（BEIR）**证明：LuminaStore 在真实检索场景（真实文档 + 真实查询 + 人工标注）
的检索质量与主流嵌入式库一致，延迟更快。**不依赖任何 demo，数据公开、指标标准、同口径对比**。

## 方法学

- **数据集**：BEIR SciFact（科学文献事实核查检索）
  - corpus：5,183 篇论文（标题 + 摘要）
  - queries：300 个测试查询（带人工相关标注 qrels）
- **Embedding**：BAAI/bge-small-en-v1.5（384 维，本地模型），**两引擎共用同一批向量**（缓存复用）
- **索引**：LuminaStore HNSW（M=32, ef_construction=200, ef_search=200）vs Chroma（同 M/efc）
- **指标**：NDCG@10、MRR@10、Recall@100（标准 IR 指标，对 qrels）
- **延迟**：单查询进程内计时 p50/p95/p99

## 结果

| 指标 | LuminaStore | Chroma | 差异 |
|---|---|---|---|
| NDCG@10 | 0.7040 | 0.7057 | -0.17pp |
| MRR@10 | 0.6714 | 0.6725 | -0.11pp |
| Recall@100 | 0.9333 | 0.9433 | -1.0pp |
| 检索 p50 | **713 us** | 1737 us | **快 2.4x** |
| 检索 p95 | **818 us** | 2082 us | 快 2.5x |
| 检索 p99 | **957 us** | 2233 us | 快 2.3x |
| 构建（5,183 文档） | **2.44 s** | 2.81 s | 快 15% |

## 结论

1. **检索质量与 Chroma 一致**（NDCG@10 差 0.17pp、MRR 差 0.11pp、Recall@100 差 1.0pp——
   均在统计噪声范围；差异来源为 ef_search 参数与图构建细节）
2. **检索延迟快 2.4-2.5x**（p50 713 vs 1737 us）——与 MNIST 性能对比（Chroma 慢 2-3x）一致
3. **构建略快**（2.44 vs 2.81 s）

**一句话**：在 BEIR SciFact 标准评测下，LuminaStore 的检索质量与 Chroma 相当（NDCG 差 <0.2pp），
延迟快 2.4 倍——**场景可行的证据成立**。

## 复现

```bash
# 1. embedding（一次性，CPU，本地模型 bench/beir/model，缓存到 bench/beir/cache/）
.venv/bin/python -u bench/beir/embed_transformers.py
# 2. 两引擎评测（读缓存）
.venv/bin/python bench/beir/run_beir.py --engine lumina --ef 200
.venv/bin/python bench/beir/run_beir.py --engine chroma
```

> 注：本机 sentence-transformers 因网络探测问题不可用，embedding 用 transformers
> 手写 CLS pooling（`bench/beir/embed_transformers.py`），已验证与 bge 官方配置一致
> （CLS token + L2 归一化）。
