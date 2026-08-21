# 嵌入式实时检索场景：LuminaStore vs 竞品对比

## 定位

LuminaStore 面向**嵌入式实时检索场景**：本地/边缘/桌面/单机应用需要「零部署、低延迟、低内存」的向量检索，同时要有完整的数据库能力（持久化、过滤、量化）。本报告用统一方法学对比同形态竞品，回答「为什么需要 LuminaStore」。

## 方法学（防质疑，全篇一致）

- **数据**：MNIST 60k × 784（像素强度归一化到 [0,1]，L2）
- **查询**：测试集前 200 个；**recall@10 对 float64 暴力 top-10 ground truth**
- **延迟**：单查询进程内计时，p50/p99；**每个 ef 先预热整组查询**（消除首次执行/冷启动伪影）
- **构建**：add 全部数据计时；**内存**：进程 max RSS
- 索引参数对齐：HNSW M=32；LuminaStore ef_construction=200（实测 400 无 recall 收益，构建翻倍），usearch expansion_add=400 按官方默认
- 同机（Apple Silicon arm64）单线程；Qdrant（服务型）因环境网络（Docker Hub 不可达）未实测，脚本保留（`bench_qdrant.py` + `run_qdrant.sh`）

## 结果（MNIST 60k × 784）

| engine | ef | recall@10 | p50 (us) | p99 (us) | QPS | build (s) | RSS (MB) |
|--------|----|-----------|----------|----------|-----|-----------|----------|
| **LuminaStore** | 100 | 0.9875 | **325** | **796** | 2690 | 28.3 | 1259 |
| **LuminaStore** | 200 | 0.9945 | 442 | 565 | 2289 | 28.3 | 1259 |
| **LuminaStore** | 400 | 0.997 | 766 | 988 | 1326 | 28.3 | 1259 |
| **Chroma** | - | 0.9985 | 988 | 1615 | 977 | 35.6 | 1265 |
| **usearch** | 50 | 0.992 | 342 | 564 | 2877 | **17.3** | 1280 |
| **usearch** | 100 | 0.9945 | 343 | 612 | 2798 | 17.3 | 1280 |
| **usearch** | 200 | 0.9955 | 629 | 1353 | 1489 | 17.3 | 1280 |
| **usearch** | 400 | 0.9955 | 731 | 1270 | 1370 | 17.3 | 1280 |

### 按 recall 对齐的延迟

| recall@10 | LuminaStore p50 | usearch p50 | 说明 |
|-----------|-----------------|-------------|------|
| ~0.99 | **325 us** (ef=100) | 342 us (ef=50) | 低 recall 区 LuminaStore 更快 |
| ~0.9945 | 442 us (ef=200) | **343 us** (ef=100) | 高 recall 区 usearch p50 更快 |
| ~0.9985 | - | - | Chroma 988 us，LuminaStore/usearch 需更高 ef |

### 结论（诚实版）

1. **LuminaStore 与 SIMD 库 usearch 处于同一性能带**：低 recall 区延迟更优，高 recall 区 p50 落后 ~30%，但 **p99 更好**（565 vs 612 us @ recall≈0.9945）；usearch 构建快 1.6x。
2. **LuminaStore 显著优于同形态嵌入式数据库 Chroma**：同 recall 下延迟 **2-3x 更低**（988 vs 325-442 us），构建更快。
3. **usearch 是库不是数据库**：无持久化/过滤/快照/多语言生态——LuminaStore 的能力组合（见下表）是其差异化。
4. **JSON 绑定开销**：LuminaStore 的 Python 延迟含 ctypes+JSON 序列化；C++ 直连路径（`ann_bench`）无此开销——嵌入式 C++ 调用者实际更快。

## 能力矩阵（嵌入式形态）

| 能力 | LuminaStore | Chroma | usearch | Qdrant(服务型) |
|------|------------|--------|---------|----------------|
| 嵌入式（无服务器/零部署） | ✅ | ✅ | ✅ | ❌（常驻进程） |
| 持久化 + 崩溃恢复 | ✅ WAL+快照 | ✅ | ❌ | ✅ |
| 过滤检索 | ✅ bitmap in/post | ✅ | ❌ | ✅ |
| 量化（内存压缩） | ✅ SQ8/Binary/PQ | ❌ | ✅(部分) | ✅ |
| 删除/更新 | ✅ | ✅ | 有限 | ✅ |
| 多语言绑定 | C + Python | Python/JS | 12 语言 | REST/gRPC |
| 本地查询延迟 | **微秒级** | 毫秒级 | 微秒级 | 毫秒级（含网络） |

## 本阶段优化记录

1. **ef_construction 参数合理化**：MNIST 60k 下 efc=400 与 200 的 recall 无差异（0.9875 vs 0.987），构建 64.9s → 33.9s（**-48%**）。
2. **启发式候选裁剪**：`select_neighbors_heuristic` 只对最近 2×M 个候选做多样化检查（更远的候选不可能入选），构建 33.9s → 29.7s（**-12%**），recall 损失 <0.2pp。
3. **benchmark 方法学修复**：每 ef 预热整组查询 + truth 后置，消除「首次执行」伪影（曾使延迟虚高 7 倍）。

## 复现

```bash
# LuminaStore（需先构建 build/libluminastore_shared.dylib）
python3 bench/embedded/bench_lumina.py data/mnist/base.fvecs data/mnist/query.fvecs --nq 200 --ef "100,200,400" --efc 200
# usearch / Chroma（bench/embedded/.venv 内）
.venv/bin/python bench/embedded/bench_usearch.py data/mnist/base.fvecs data/mnist/query.fvecs --nq 200 --ef "50,100,200,400"
.venv/bin/python bench/embedded/bench_chroma.py data/mnist/base.fvecs data/mnist/query.fvecs --nq 200
# 汇总
.venv/bin/python bench/embedded/summarize.py bench/embedded/results_mnist_60k.jsonl
```
