# LuminaStore 向量存储引擎 — 技术问答（QA）

> 本文档面向技术面试、答辩与代码走读，问题与答案与当前仓库实现（`LuminaStore/`）对齐。  
> **专业术语**：正文中首次出现时用中文括号或脚注说明；文末附 **术语速查表**。

---

## 0. 术语速查表（Glossary）

| 术语 | 英文 / 缩写 | 简要注释 |
|------|-------------|----------|
| **向量存储引擎** | Vector Storage Engine | 管理向量数据的写入、索引与查询的软件组件；LuminaStore 为 **原型（prototype）** 级别实现。 |
| **ANN** | Approximate Nearest Neighbor | **近似最近邻**：不保证全局最优，用更少计算换取可接受召回率（Recall）的检索方式。 |
| **HNSW** | Hierarchical Navigable Small World | **分层可导航小世界图**：一种常用的 ANN 图索引结构，多层图 + 贪心搜索。 |
| **WAL** | Write-Ahead Log | **写前日志**：变更先顺序追加到日志再反映到内存结构；崩溃后可通过 **重放（replay）** 恢复。 |
| **fsync** | file synchronize | 将内核页缓存刷到持久设备；保证 **持久性（durability）** 但成本高。 |
| **Group Commit** | 组提交 | 批量多次写入后 **合并 fsync**，换吞吐、牺牲部分崩溃窗口内的持久性。 |
| **CRC32** | Cyclic Redundancy Check | **循环冗余校验**：检测帧/块是否在传输或落盘中损坏。 |
| **SIMD** | Single Instruction, Multiple Data | **单指令多数据**：一条指令并行处理多组标量数据（如 SSE/AVX/NEON）。 |
| **ISA** | Instruction Set Architecture | **指令集架构**：CPU 支持的指令集合；不同机器 ISA 能力不同。 |
| **Runtime Dispatch** | 运行时派发 | 程序运行时根据 **CPUID** 等选择具体核函数，避免在不支持 AVX-512 的机器上执行非法指令。 |
| **Tombstone** | 墓碑记录 | **逻辑删除**标记；重放日志时用于从内存索引中移除键，而非物理擦除历史。 |
| **ANN 参数 M / efConstruction / efSearch** | — | `M`：每层最大连边数上界相关；`efConstruction`：建图候选集规模；`efSearch`：查询候选集规模。 |
| **L2 距离** | Euclidean / L2 | **欧氏距离**；实现中 `search_top_k` 返回的 `distance` 与 `VectorMath::l2_distance` 语义一致（见代码与单测）。 |
| **余弦距离** | Cosine distance | 本项目中由 `VectorMath::cosine_distance` 提供；常用于归一化后的语义相似度。 |
| **对齐分配** | Aligned allocation | 按 **64 字节** 等边界分配，便于 SIMD `load` 对齐访问，减少 `loadu` 与额外惩罚。 |
| **Slice** | — | 零拷贝字节视图（`std::string_view` 包装），用于 `put`/`get` 的键值接口。 |
| **Status** | — | 操作结果类型：`OK`、`NotFound`、`IOError`、`Corruption` 等。 |

---

## 1. 项目定位与边界

### Q1.1 LuminaStore 是什么？不是什么？

**答：**  
LuminaStore 是一个用 **C++20** 编写的 **高性能持久化向量存储引擎原型**。它提供：

- **WAL** 持久化与启动 **recovery（恢复）**；
- 内存 **key → WAL offset** 索引，支持 `put` / `get` / `remove` / `put_vector`；
- **SIMD** 派发的向量距离（L2、余弦等路径）；
- 内存中 **HNSW** 索引的 `add_item`、`search_top_k`，以及图结构的 `save` / `load`。

它 **不是** 分布式向量数据库（无分片、副本、SQL、混合检索过滤等完整商业特性）；定位是 **单进程、可读可测** 的引擎内核演示与 RAG/Agent Memory 方向的技术验证。

---

### Q1.2 为什么要自研而不是直接用 Faiss / hnswlib？

**答：**  
常见动机包括：教学与 **可控性**（WAL 格式、恢复策略、SIMD 路径完全可见）；把 **存储可靠性（durability）** 与 **向量检索性能** 收敛到同一仓库；练习 **C++20、并发、磁盘格式、CPU 特性检测** 等工程能力。生产选型通常会评估成熟库与运维成本；本项目侧重 **原理落地与工程细节**。

---

## 2. 整体架构

### Q2.1 代码分为哪几层？各层职责？

**答：**  

1. **`lumina::storage`**：`LogManager`（WAL 帧读写、CRC、尾部修复）、`IndexManager`（**key → 最新 Put 的 WAL 字节偏移**）、`StorageEngine`（对外 API、锁、`recover`）。  
2. **`lumina::vector`**：`VectorMath`（距离核）、`HNSWIndex`（图索引）、`AlignedFloatVector`（对齐缓冲）。  
3. **`lumina::common`**：`Status`、`Slice`、`Options`、`OpType`、`CRC` 等公共类型与工具。  
4. **`cpp_engine/c_api.cpp`**：**C ABI** 桥接层，供 Python 等通过 **动态库** 调用 HNSW + payload 的简化集成。

---

### Q2.2 WAL 里存的「向量」与 HNSW 图是什么关系？

**答：**  
在当前设计中要分开理解：

- **`StorageEngine::put_vector`**：把一条 **`OpType::kVectorPut`** 记录追加到 **WAL**，并在 `IndexManager` 中记录 **key → 该帧在 WAL 中的 offset**；`get` 可按 key 从 WAL **随机读（pread）** 取出 value 字节串。这是 **日志型 KV/大 value** 语义。  
- **`HNSWIndex`**：在内存中维护 **图结构**与向量副本，支持 `save`/`load` 到 **`Options::hnsw_path`** 指定的文件；与 WAL 帧格式是 **另一条持久化路径**。

**C API**（`lumina_add_vector`）在内存里同时更新 **HNSW** 与 **`payloads` 哈希表**；payload **未**写入本文件的 `StorageEngine` WAL。面试时应诚实说明：**原型中多条路径并存**，完整产品需统一 **事务边界** 与 **崩溃一致性** 模型。

---

## 3. WAL 与恢复（LogManager）

### Q3.1 WAL v2 文件头与单帧布局是什么？

**答：**  
（与 `include/lumina/storage/log_manager.h` 注释一致。）

**文件头（8 字节，多字节整型为 big-endian / 大端序）：**

- `4B` **Magic**：ASCII `LMST`（用于识别新格式文件）。  
- `2B` **format version**：当前为 `1`。  
- `2B` **reserved**：必须为 `0`。

**每一帧（Frame）：**

- `1B` **OpType**：`0x01 Put`，`0x02 Delete`，`0x03 VectorPut`。  
- `4B` **CRC32**。  
- `4B` **PayloadLen**（大端）。  
- **Payload**：`[2B KeyLen BE][key bytes][value bytes]`；`Delete` 时 value 为空。

**CRC32** 覆盖：`1B Op` + **编码后的 4B PayloadLen（大端字节序参与 CRC 拼接）** + **完整 payload**。这样长度字段既在磁盘上可解析，又参与完整性校验。

---

### Q3.2 什么是 Legacy v0 WAL？为何保留？

**答：**  
旧格式 **无文件头**，第一帧即从 offset 0 开始；帧内长度字段为 **host-endian（主机字节序）**。打开时若魔数不匹配则按 v0 解析，以兼容旧测试文件或历史数据，直到文件被完全迁移/重写。

---

### Q3.3 启动恢复时，「尾部不完整帧」与「中间坏帧」如何处理？

**答：**  

- **尾部不完整（partial write at EOF）**：无法构成合法帧头或 payload 未写完 → 视为崩溃写入未完成 → 使用 **`ftruncate(2)`** 将文件截断到最后一个 **完整合法帧** 的边界。  
- **最后一帧完整但 CRC 失败**：按 **tail damage（尾部损坏）** 处理，可截断到该帧之前。  
- **中间某一帧 CRC 失败且其后仍有数据**：判定为 **middle corruption（中间损坏）** → `open()` 返回 **`Status::IsCorruption()`**，避免静默丢弃后续可能有效的数据。

这体现了 **日志修复（log repair）** 中「可截尾」与「不可瞎修中间」的工程分界。

---

### Q3.4 `group_commit` 与 `sync_every_n_appends` 如何改变持久性？

**答：**  
（见 `Options` 与 `storage_engine.cpp` 中 `maybe_fsync_after_append`。）

- **`sync_writes == false`**：引擎内 **从不 fsync**，仅依赖调用方 `StorageEngine::sync()`。  
- **`sync_writes == true` 且 `group_commit == false`**：每次变更写入后可 **立即 fsync**（最强持久、最慢）。  
- **`group_commit == true`**：不在每次 `put` 都 fsync；若 **`sync_every_n_appends > 0`**，则每 **N 次 append** 调一次 `log.sync()`；若为 `0` 则只靠 **`sync()`** 手动刷盘。

**注释（持久性窗口）**：两次 fsync 之间的崩溃可能丢失尚未落盘的日志尾部——这是 **用 durability 换 throughput（吞吐）** 的经典权衡。

---

## 4. StorageEngine 与索引（IndexManager）

### Q4.1 `IndexManager` 存的是什么？为什么不是存 value？

**答：**  
内存哈希表 **`std::unordered_map<std::string, uint64_t>`**：**key → WAL 中该 key 最近一次 Put/VectorPut 记录的起始字节偏移**。  
读取时通过 **pread** 从 WAL 文件按 offset 解析帧得到 value。这样内存占用随 **键数量** 增长，而不是随所有历史版本线性膨胀（新 Put 覆盖 map 中 offset，旧帧仍在文件中但不再可达，除非做 **compaction（压缩/合并）**——当前原型未实现）。

---

### Q4.2 `remove` 是物理删除吗？

**答：**  
否。`remove` 向 WAL 追加 **`OpType::kDelete`**（**tombstone / 墓碑**），并从 `IndexManager` 删除该 key。重放时遇到 Delete 则不再把该 key 加入索引。磁盘上历史 Put 帧仍存在，属于 **append-only** 设计的常见取舍。

---

### Q4.3 并发模型是怎样的？

**答：**  
`StorageEngine` 使用 **`std::shared_mutex`**：`put`/`remove`/`recover` 等写路径用 **独占锁（unique_lock）**；`get` 可用 **共享锁（shared_lock）**（若实现为读多写少优化）。  
**C API** 使用全局 **`std::mutex`** 保护 `HNSWIndex` 与 payload 表。高并发产品级引擎通常会 **分片（sharding）** 或无锁结构；此处以 **正确性与简单性** 为主。

---

## 5. HNSW 索引（HNSWIndex）

### Q5.1 本实现的 HNSW 入口参数有哪些？

**答：**  
`HNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200)`；查询 `search_top_k(const float* query, size_t k, size_t ef_search = 50)`。  
全局默认还与 `Options::hnsw_*` 对齐，供上层配置。

- **`dim`**：**embedding 维度**，所有向量长度一致。  
- **`M`**：每层邻居数上界相关（第 0 层为 `2*M` 等策略，见 `hnsw_index.cpp`）。  
- **`ef_construction`**：插入时在层内 **候选集扩展规模**，越大建图越慢、图质量通常越好。  
- **`ef_search`**：查询时 **beam / 候选列表** 规模，越大延迟越高、召回通常越好。

---

### Q5.2 插入新节点的大致步骤？（口述级）

**答：**  

1. 校验指针、**禁止 duplicate id（重复主键）**。  
2. 按指数分布采样 **随机层 `new_layer`**（实现用 `ml = 1/log(max(M,2))` 与 `-log(U)*ml`）。  
3. 若图为空，新点成为 **entry_point（入口点）**。  
4. 否则从当前 **max_layer** 向下 **greedy（贪心）** 下降到 `new_layer`，每层用 `search_layer` 找近邻。  
5. 在 `new_layer..0` 上连接双向边，并在边数过多时按距离 **prune（剪枝）** 到 `max_m`。  
6. 若 `new_layer` 超过原 `max_layer`，更新全局入口与高度。

向量在节点内使用 **`AlignedFloatVector`** 存储，距离调用 **`VectorMath::l2_distance`**（经 **runtime dispatch**）。

---

### Q5.3 `save` / `load` 持久化了什么？

**答：**  
将 **图拓扑与节点向量** 序列化到磁盘文件（具体二进制布局以 `hnsw_index.cpp` 为准），用于进程重启后恢复 **ANN 结构**。与 WAL 的 **操作日志** 是不同子系统；统一容灾需定义 **checkpoint（检查点）** 或 **单一真相源** 策略（当前文档级「待演进」项）。

---

## 6. SIMD 与 VectorMath

### Q6.1 为何需要 `VectorMath::init()`？何时调用？

**答：**  
`init()` 负责 **CPU 特性检测** 并注册 **函数指针表**（单一指针，避免双读不一致）；**线程安全、幂等**（`std::call_once`）。  
`l2_distance` / `cosine_distance` 在首次调用时会 **懒初始化**。测试或 benchmark 可显式先 `init()`。

---

### Q6.2 x86 上 AVX2 与 AVX-512 的优先级？CMake 如何关闭 AVX-512 派发？

**答：**  
README 说明：在同时构建出 AVX2 与 AVX-512 对象文件时，**运行时优先走 AVX2 + FMA**；仅当 AVX2 路径不可用再考虑 AVX-512F。原因是 **频率降频（frequency throttling）** 等 **512-bit 副作用**，默认采取 **保守策略（conservative default）**。  
若需编译期去掉 AVX-512 派发分支：`-DLUMINA_RUNTIME_USE_AVX512=OFF`（定义 `LUMINA_ALLOW_AVX512_KERNEL=0`）。

---

### Q6.3 对齐与 `load` / `loadu` 的关系？

**答：**  
AVX2 路径：**32 字节对齐** 用 `_mm256_load_ps`，否则 `_mm256_loadu_ps`。  
AVX-512：**64 字节对齐** 用 `_mm512_load_ps`，否则 `_mm512_loadu_ps`。  
NEON 路径在可用时使用对齐提示。  
**HNSW 节点向量**通过 **`AlignedFloatVector`（`kVectorBufferAlignment = 64`）** 分配，提高热路径对齐概率。

---

### Q6.4 `l2_distance` / `cosine_distance` 与 `*_naive` 的契约差异？

**答：**  
（与 `vector_math.h` 注释一致。）

- **Dispatch 入口**：`dim == 0` 时 L2 返回 `0`，余弦距离返回 `1`；`dim > 0` 若指针为 `nullptr` 返回 **quiet NaN**。  
- **`*_naive`**：`dim > 0` 时 **不检查空指针**，无效指针为 **UB（未定义行为）**；`dim == 0` 不读取元素。

面试表述：**公开 API 防御性更强，naive 供内核热路径与单测对齐参照**。

---

## 7. C API 桥接（cpp_engine）

### Q7.1 暴露了哪些符号？线程安全吗？

**答：**  
`extern "C"`：`lumina_init(dim)`、`lumina_add_vector(id, vector, dim, payload_text)`、`lumina_search_vector(query, dim, top_k)`、`lumina_free_string(ptr)`。  
全程在全局 **`mutex`** 下操作；返回的 JSON 字符串由 **`malloc`** 分配，需 **`lumina_free_string`** 释放，属于 **C 侧所有权约定（ownership）**。

---

### Q7.2 `lumina_search_vector` 返回的 JSON 含哪些字段？

**答：**  
`results` 数组中每项含 **`id`**、**`distance`**（L2 语义下的距离值）、**`payload`**（经 **JSON escape** 的文本）。错误路径返回带 **`error`** 字段的 JSON。

---

## 8. 构建、测试与基准（Benchmark）

### Q8.1 如何构建与跑测试？

**答：**  

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

可选：**ASan（AddressSanitizer，地址消毒器）** 构建：`-DLUMINA_ENABLE_ASAN=ON`；**`-march=native`**：`-DLUMINA_ENABLE_MARCH_NATIVE=ON`（可榨干本机 ISA，但 **二进制不可移植**）。

---

### Q8.2 Benchmark 二进制测什么？

**答：**  
- **`storage_bench`**：顺序写吞吐、随机读延迟等存储路径。  
- **`vector_bench`**：**naive vs SIMD dispatch** 的 L2、对齐/非对齐输入、**HNSW top-k** 查询延迟。

---

## 9. 设计取舍与开放问题

### Q9.1 若磁盘出现「中间 CRC 失败」，为什么不自动截断后面所有数据？

**答：**  
因为无法区分「真损坏」与「实现 bug / 版本不兼容 / 恶意篡改」；盲目截断可能 **静默丢数据（silent data loss）**。工程上更稳妥的是返回 **`Corruption`**，由运维 **备份恢复** 或 **人工审计**。

---

### Q9.2 下一步可演进方向？（结合 `PROJECT_NARRATIVE.md`）

**答：**  
**CI 矩阵**（GCC/Clang/MSVC、ARM64）；**`LUMINA_PREFER_AVX512`** 等策略开关；更大规模 **recall–latency 曲线** 报告；**WAL / HNSW 文件格式版本迁移手册**；**Compaction**；**快照（snapshot）**；与对象存储结合的 **冷备** 等。

---

## 10. 速记卡片（30 秒电梯陈述）

> 「LuminaStore 是 C++20 的向量引擎 **原型**：**append-only WAL** + **帧 CRC** + **尾部 ftruncate 修复** 保证崩溃恢复语义清晰；**内存 key→offset 索引** + **pread** 做读取；**HNSW** 提供 **ANN** 检索，**SIMD 运行时派发**（AVX2/AVX-512/NEON）+ **64B 对齐** 优化距离计算；配套 **GoogleTest / Benchmark** 与 **C API** 方便和 Python RAG 链路集成。」

---

## 附录 A：关键源文件索引

| 路径 | 内容 |
|------|------|
| `include/lumina/storage/log_manager.h` | WAL v2 格式与恢复语义注释 |
| `src/storage/log_manager.cpp` | CRC、读写、修复、`append` |
| `src/storage/storage_engine.cpp` | `put`/`get`/`remove`、`group_commit` |
| `include/lumina/storage/storage_engine.h` | 对外 API |
| `src/vector/hnsw_index.cpp` | HNSW 插入、查询、save/load |
| `include/lumina/vector/vector_math.h` | 距离函数契约 |
| `src/vector/vector_math_scalar.cpp` | 检测与派发 |
| `cmake/DetectSIMD.cmake` | 条件编译 SIMD 翻译单元 |
| `cpp_engine/c_api.cpp` | C 动态库接口 |
| `README.md` / `PROJECT_NARRATIVE.md` | 英文说明与中文叙事 |

---

## 附录 B：与本 QA 文档的关系

- 实现以仓库 **当前提交** 为准；若代码变更，请同步更新 **帧布局、状态码、CMake 开关** 等段落。  
- 术语表中英文缩写首次出现已在正文或表中注释；更深理论可参考 HNSW 原论文与各类 **LSM-tree（Log-Structured Merge-tree，日志结构合并树）** 资料对比 WAL 设计哲学（本引擎未实现 LSM，仅借鉴 **append-only log** 思想）。

---

*文档版本：与 LuminaStore 仓库同步维护；作者栏可填你的名字与日期。*
