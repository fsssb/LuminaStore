# LuminaStore 项目构思与实现说明

## 1. 项目一句话定义

LuminaStore 是一个基于 C++20 的高性能**嵌入式向量数据库**：把「可靠落盘（WAL + 快照）」与「向量检索（HNSW + 量化 + 过滤）」统一在一个可读、可测、可扩展的工程里。

## 2. 为什么做这个项目（背景与痛点）

- 很多向量系统的存储可靠性与检索性能由不同组件承担，链路复杂、崩溃恢复策略不一致；
- 生产级 ANN 引擎普遍在量化、过滤、磁盘分层上做文章，但多数教学项目只到「裸 HNSW」；
- SIMD 优化常停留在「编译机可用」，缺少运行时能力检测与按 ISA 分发的工程实践；
- 校招场景需要一个**每一层都能讲清楚**、且有可量化证据（benchmark / 对比表）的深度项目。

## 3. 设计目标

### 3.1 功能目标

- 向量 + payload + 过滤字段全部持久化，重启后状态一致（WAL 追加 + 快照增量恢复）；
- HNSW 近似最近邻（启发式邻居选择、删除/更新、L2/IP/Cosine）；
- 量化压缩（SQ8 / Binary / PQ）与精排重排管线；
- 过滤检索（in-filter / post-filter，bitmap 位图索引）；
- 句柄式 C API + Python 绑定（numpy 批量）。

### 3.2 非功能目标

- 性能：SIMD 距离内核（运行时派发）+ SIMD 量化距离；
- 可靠：WAL 帧级 CRC + 尾损坏自动修复 + 快照水位恢复；
- 可移植：x86（AVX2/AVX-512）/ ARM（NEON）均可编译；
- 可维护：gtest 单元测试、google-benchmark 基准、recall-QPS 评测工具。

## 4. 核心架构（v2 分层）

```
API 层        C API (cpp_engine) / Python (ctypes + numpy)
查询执行层    Collection（写路径 + 搜索管线：粗搜 → 候选 → 精排/过滤）
索引层        HNSW（启发式邻居选择 + 标记删除 + 细粒度锁）
              Quantizer（SQ8/Binary/PQ + SIMD 距离）
              FilterIndex（字段值 → bitmap）
存储层        WAL（追加、CRC、group commit、tail repair）
              Snapshot + MANIFEST（快照水位 + 增量重放）
计算层        SIMD 距离内核（L2/IP/Cosine + 量化距离）
```

模块依赖：上层依赖下层；`engine` 依赖 `index`/`storage`，`index`/`storage` 不依赖 `engine`。

## 5. 迭代历程（v1 → v2）

**v1（原型）**：WAL v2 + 内存 HNSW + SIMD 距离 + C API 全局单例。
局限：payload 不落盘、启动全量重放、无量化/过滤、并发靠全局锁。

**v2（本版）**：
- **存储闭环**：payload 与过滤字段入 WAL（`VectorPutV2`），快照 + MANIFEST 增量重放，kill -9 后重启一致；
- **索引升级**：启发式邻居选择（替换距离截断）、标记删除/更新、可注入距离函数；
- **并发**：label 条带锁 + node 条带锁 + 读写锁模型（写与读互斥，读并发）；
- **量化**：SQ8/Binary/PQ + SIMD 内核（code-code 距离快于全精度 2-4x），内存 4x-32x 压缩；
- **过滤**：bitmap FilterIndex，in-filter / post-filter 双模式（recall@10=1.0）；
- **绑定**：句柄式 C API + Python（numpy 批量）；
- **评测**：recall-QPS 工具 + 量化/过滤基准。

## 6. 关键设计决策与理由

| 决策 | 理由 |
|---|---|
| WAL 即数据，payload 不单独建文件 | 读少的小 payload，OS page cache 足够；接口不变可迁移 |
| 快照水位 = WAL 字节偏移 | append-only 天然单调；重放从水位开始 |
| update 全程持写锁 | 保证图结构原子性（并发 bug 曾导致邻居向量损坏，ASAN 定位后修正） |
| 搜索展开队列不过滤、结果集过滤 | hnswlib 语义；否则低选择性时搜索枯竭返回空 |
| 量化先做码间距离 SIMD | 图上存码后遍历用 code-code 距离（热路径），快于全精度 |

## 7. 面向秋招的定位

- **差异化**：不硬碰 FAISS/Milvus，强调「从零实现、每层可讲、量化 + 过滤 + 存储闭环覆盖生产系统核心挑战」；
- **证据**：12 组单元测试、4 组 benchmark、recall-QPS 曲线、量化/过滤对照数字（见 docs/benchmarks.md）；
- **技术栈**：C++20、CMake、SIMD（NEON/AVX2 手写）、系统编程（fsync/mmap/线程）、图索引与量化算法。
