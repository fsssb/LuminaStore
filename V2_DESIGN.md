# LuminaStore V2 详细设计方案

> 版本：draft v1（2026-08-12）
> 前置文档：`RESEARCH.md`（竞品调研与 v2 路线图）
> 范围：在 v1（WAL v2 + 内存 HNSW + SIMD 距离 + C API）基础上的增量设计

---

## 1. 概述

### 1.1 设计目标

1. **闭环**：v2 必须是一个「可重启恢复」的完整引擎 —— 向量 + payload + 过滤字段全部落盘，重启后状态与崩溃前一致。
2. **差异化算法深度**：落地量化压缩（SQ8/二进制/PQ + 精排重排）与过滤检索（in-filter / post-filter 对照），这是秋招简历的核心叙事。
3. **可量化评测**：接入 ann-benchmarks 方法学，产出 recall@10–QPS 前沿曲线，与 hnswlib/FAISS 同机对比。
4. **可维护**：显式四层架构（存储 / 索引 / 查询执行 / API），代码量可控（预计新增 ~3-4k 行），每个模块可单独测试。

### 1.2 非目标（明确不做）

- 分布式、多节点、分片复制（单机嵌入式定位）。
- 无锁 HNSW（hnswlib 也只做细粒度锁分层，收益/成本不成比例）。
- 完整 SQL / DSL（过滤用最小化的程序化谓词）。
- LSM 树（append-only WAL + 快照已满足单机向量场景）。
- 稀疏向量 / BM25 混合检索（列为 v3 备选，不在 v2 范围）。

### 1.3 v1 关键事实（设计依据，已读代码核实）

| 事实 | 影响 |
|---|---|
| C API 层**绕过 StorageEngine**，直接用 `HNSWIndex` + 全局 `unordered_map<uint64_t,string> payloads` | payload 不落盘是 API 层缺陷，v2 需把引擎层重新建立在 StorageEngine 之上 |
| StorageEngine 已支持 `put_vector`（WAL `kVectorPut`），`IndexManager` 维护 key→offset | 向量数据落盘路径已存在，v2 复用并补 payload/过滤字段 |
| WAL v2：逐帧 `[1B Op][4B CRC BE][4B PayloadLen BE][payload]`，tail 修复 + middle corruption 报错 | 格式已健壮；v2 保持该语义，仅扩展 OpType 与 payload 编码 |
| HNSW `connect_bidirectional` **仅按距离排序截断**，无启发式（多样化）邻居选择 | v2 必须补启发式邻居选择（论文 + hnswlib 一致的核心技巧） |
| HNSW 无删除/更新语义 | v2 补标记删除 + 更新 |
| 距离内核已支持 L2 / cosine，SIMD 派发已就绪 | v2 补内积（IP）与量化距离 |
| 并发：StorageEngine 全局 `shared_mutex` | v2 引入 HNSW 细粒度锁（label striping + per-node），支持写读并发 |

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│  API 层                                                      │
│  ┌────────────┐  ┌─────────────────┐                        │
│  │ C API (C)  │  │ Python 绑定      │  (nanobind, numpy 批量) │
│  └─────┬──────┘  └────────┬────────┘                        │
├───────┼───────────────────┼─────────────────────────────────┤
│ 查询执行层 (engine/)                                        │
│  Collection: add / search / get / remove / filter 谓词求值    │
│  SearchPipeline: 粗搜(量化距离 + in-filter) → 候选 → 精排     │
├───────┼─────────────────────────────────────────────────────┤
│ 索引层 (index/)                                             │
│  HNSW v2（启发式邻居选择 / 标记删除 / 细粒度锁）                │
│  Quantizer（SQ8 / Binary / PQ，训练 + 编码 + 距离）           │
│  FilterIndex（字段 → bitset bitmap）                         │
├───────┼─────────────────────────────────────────────────────┤
│ 存储层 (storage/)                                           │
│  WAL v3（追加、CRC、group commit、tail repair）              │
│  Manifest + Snapshot（HNSW 图 + payload 快照，增量重放）      │
│  PayloadStore（payload 与过滤字段的读写）                     │
├───────┴─────────────────────────────────────────────────────┤
│ 计算层 (vector/)：SIMD 距离内核、量化距离内核、64B 对齐分配     │
└─────────────────────────────────────────────────────────────┘
```

**模块依赖规则**：上层依赖下层，同一层内 `engine → index/storage`，`index/storage` 不依赖 `engine`；`vector` 是纯函数层，被所有层依赖。

**数据流（写入）**：
`add(id, vec, payload, filters)`
→ 序列化 value → `WAL.append(kVectorPutV2, id, value)` → HNSW `add_item`（更新图）→ FilterIndex 更新 bitset → （按 group_commit 策略 fsync）

**数据流（查询）**：
`search(query, k, ef, filter)`
→ HNSW 遍历（**图上压缩向量距离** + **遍历不过滤、入结果集时过滤**）→ 候选集 → 原向量精排（rescore）→ 排序取 top-k

**数据流（恢复）**：
`open()`
→ 读 manifest → 加载最近快照（HNSW 图 + payload）→ 从快照水位起 replay WAL 剩余记录 → 重建 FilterIndex → 完成

---

## 3. 数据模型与磁盘格式

### 3.1 数据模型

```
Collection
 ├── vector_dim: size_t
 ├── distance_metric: L2 | IP | COSINE
 ├── entries: map<id, Entry>
 └── Entry {
       id: uint64_t
       vec: float[dim]          // 原始向量（精排用）
       code: QuantCode          // 量化码（粗搜用，随量化模式可选）
       payload: bytes           // 用户负载（opaque，落盘）
       scalars: map<string, ScalarValue>  // 过滤字段（int64/float64/string）
     }
```

- **id 语义**：唯一、用户指定（v1 沿用）。
- **删除**：标记删除（tombstone），不物理断边；快照时可选压缩（跳过 tombstone 重建图）。
- **更新**：`put` 同 id = 覆盖（向量重连 + payload 覆盖 + 过滤字段覆盖），WAL 追加新记录。

### 3.2 WAL 格式 v3（扩展 v2，保持向后兼容）

**文件头（不变，8B）**：`[4B magic 'LMST'][2B version=2][2B reserved=0]`
→ v3 使用 **version=2** 标识（v1 头部 version 字段语义保留，文件头本身不变）。

**帧格式（帧头不变，扩展 OpType）**：

```
[1B OpType] 0x01=Put  0x02=Delete  0x03=VectorPut  0x04=VectorPutV2  (新增)
[4B CRC32 BE] 覆盖：Op + PayloadLen(BE 编码) + Payload
[4B PayloadLen BE]
Payload(Op=0x04) :=
  [2B KeyLen BE][key bytes]                 // key = id 十进制字符串（沿用 v1）
  [2B MetaLen BE][meta bytes]               // meta = 序列化 Entry 元数据（见下）
```

**Entry 元数据编码（meta bytes，小端，引擎内部格式）**：

```
[4B flags]            bit0: 有 vec, bit1: 有 payload, bit2: 有 scalars
[4B vec_len][vec float32×dim]              // 仅 flags.bit0
[4B payload_len][payload bytes]            // 仅 flags.bit1
[2B nscalars]
  每项: [1B type][2B key_len][key][8B value]  // type: 0=int64 1=double 2=string(4B len+bytes)
```

**约束**：
- `VecLen == dim*4` 由引擎校验，`kMaxPayloadBytes = 256MB` 沿用。
- CRC 覆盖规则、tail repair / middle corruption 语义与 v1 **完全一致**（复用现有 `repair_truncate_tail` 逻辑，只需在 `is_valid_op` 与解码处扩展）。
- 读兼容：v1 的 `kVectorPut`（0x03）仍可解码（key + raw value），作为 v2 的 payload-only 记录处理。

### 3.3 Payload 存储策略

**决策**：payload 留在 WAL 中（`read_value_at(offset)` 按需读），不单独建 payload 文件。

理由：
- v1 `IndexManager` 已存 key→offset，`LogManager::read_value_at` 已实现随机读，零新增文件格式。
- payload 通常是「小 + 读少」的元数据（RAG 场景是 chunk 文本），WAL 随机读（OS page cache 命中）足够。
- 若未来 payload 变大/读密集，可无缝迁移到独立 `payload.pack`（顺序追加 + 稀疏位图索引），接口不变。

**过滤字段**：**不落 WAL 的 payload，而是重建时从 Entry.meta.scalars 重建 FilterIndex**（scalars 已在 meta 中）。过滤字段是查询热点，重建成本 O(n·nfields)，可接受。

### 3.4 快照与 Manifest（解决「启动全量重放」）

**快照文件 `snap-<seq>.snap`**：

```
[4B magic 'LMSN'][4B version=1]
[8B wal_offset]                       // 快照覆盖到的 WAL 字节偏移（= 重放水位）
[8B node_count][8B dim]
HNSW 图：入口点 / 最大层 / 每节点(id, max_layer, 邻接表(逐层, 定长uint32))
量化码：QuantCode 逐节点（若启用量化）
payload：逐 id (id, payload_len, payload)   // 大块顺序写
scalars：逐 id 的过滤字段表
尾部: [4B CRC32]（覆盖整个快照内容）
```

**Manifest 文件 `MANIFEST`**：

```
文本行格式（简单可靠，每行一个记录）：
  version 1
  snap <seq> <wal_offset> <path>
  fsync_hint <offset>      // 可选：最后一次显式 sync 的 WAL 位置
```

**恢复流程**：
1. 读 `MANIFEST`，取最新 `snap` 记录 → 加载 `.snap`（先整文件 CRC 校验，损坏则回退到上一快照或全量重放）。
2. 从 `snap.wal_offset` 起对 WAL 做 `iterate` 增量重放（复用现有 `LogManager::iterate`）。
3. 重建 FilterIndex。
4. 若 WAL 起点损坏（middle corruption），报错并建议手动处理（语义与 v1 一致）。

**快照触发策略**：默认 `snapshot_interval_bytes`（默认 256MB WAL 增量）或显式 `snapshot()`；快照成功后**截断旧 WAL 前缀**（写新 WAL 文件 + 更新 CURRENT 式指针，参考 RocksDB 做法，但 v2 用「单 WAL + 定期 snapshot」即可，不做 WAL 轮转）。

> 简化决策：v2 不做 WAL 轮转。WAL 文件保持追加，快照仅用于加速启动。若要回收磁盘，提供 `compact()`（重写 WAL：只保留活条目，tombstone 剔除）—— 放在 M3。

---

## 4. 引擎层设计（engine/）

### 4.1 核心类

```cpp
// engine/collection.h
class Collection {
public:
    // —— 生命周期 ——
    Status open(const Options& opts);          // 加载 manifest+snapshot，增量重放 WAL
    Status snapshot();                          // 手动触发快照

    // —— 写入（线程安全：单写多读）——
    Status add(uint64_t id, const float* vec, const std::string& payload,
               const std::vector<ScalarField>& scalars);
    Status remove(uint64_t id);                 // 标记删除
    Status update(uint64_t id, const float* vec, const std::string& payload,
                  const std::vector<ScalarField>& scalars); // = remove+add 语义，图内重连

    // —— 读取 ——
    Status get(uint64_t id, std::string* payload, float* vec_out /*可选*/) const;

    // —— 查询 ——
    std::vector<SearchResult> search(const float* query, size_t k,
                                     const SearchOptions& opts) const;
    std::vector<SearchResult> search_filtered(const float* query, size_t k,
                                              const FilterExpr& filter,
                                              const SearchOptions& opts) const;

    size_t size() const;
    size_t dim() const;

private:
    std::unique_ptr<Impl> impl_;
};

// engine/query.h
struct SearchOptions {
    size_t ef_search = 50;
    bool   use_quantized_distance = true;   // 粗搜用量化距离（启用量化时）
    bool   rescore = true;                  // 候选集原向量精排
    size_t rescore_candidates = 0;          // 0 = 自动(≈ ef_search * 2)
};

// engine/filter.h — 最小化程序化谓词
using ScalarValue = std::variant<int64_t, double, std::string>;
struct FilterExpr {                          // 顶层是 AND 列表
    std::vector<Clause> clauses;             // Clause: {field, Op(=, !=, <, <=, >, >=, In), ScalarValue}
};
```

### 4.2 查询执行管线（SearchPipeline）

```
search(query, k, opts, filter?)
  ├─ 1. 距离函数选择：SIMD L2 / IP / 余弦；量化模式下选 Quantizer::distance
  ├─ 2. HNSW 遍历（含 filter 时：遍历不过滤、入结果集时按 bitmap 检查）
  ├─ 3. 候选集（ef 个）→ 若 rescore：逐候选算原向量精确距离
  ├─ 4. 排序，截取 top-k
  └─ 5. 统计（距离计算次数 / 候选数 / 耗时）→ 返回给 benchmark 层
```

**过滤三模式（作为可测实验）**：
- `FilterMode::PostFilter`：无过滤遍历 → 结果过滤 → 若不足 k 自动放大 ef（oversample）重试。
- `FilterMode::InFilter`（默认）：遍历中入结果集检查（hnswlib 风格），结果恒满足谓词。
- 两者通过 `FilterBench` 工具输出「选择性 vs recall vs 距离计算次数」对照（对应 RESEARCH.md P0-2）。

---

## 5. 索引层设计（index/）

### 5.1 HNSW v2（在 v1 上增量改造）

**5.1.1 启发式邻居选择（关键升级）**

替换 v1 的「距离排序截断」为论文/hnswlib 的多样化剪枝：

```cpp
// 输入：候选(距离升序) + 已选集合
// 规则：候选 c 入选 ⇔ c 到已选集合中任一元素的距离 < c 到中心点距离的某比例
//       （论文用 <，hnswlib 用 ≤ 以允许绑定；保留距离比系数 = 1.0 简化）
std::vector<size_t> select_neighbors_heuristic(
    const Node& center, const std::vector<std::pair<float,size_t>>& candidates,
    size_t max_m, bool keep_conns /* 反向边溢出时的替换选择 */);
```

- 高维数据召回显著提升；与论文 5.2 节、hnswlib `getNeighborsByHeuristic2` 一致。
- 用 `AlignedFloatVector` 批量距离 + SIMD 内核实现（v1 已有基础设施）。
- **新增基准用例**：SIFT-1M 上「heuristic vs 截断」recall 对比（预期 +3~8pp @ 同 QPS）。

**5.1.2 标记删除**

- Node 增加 `deleted` 位；搜索跳过已删节点（`search_layer` 遍历时标记，不物理断边）。
- 删除位随 `save/load` 持久化（v1 save 格式升级，见 §3.4）。
- `size()` 返回活跃数；提供 `compact()`（可选重建）。

**5.1.3 细粒度并发**

参考 hnswlib 三层锁（v1 是 StorageEngine 全局 shared_mutex）：

```cpp
class HNSWIndex {
private:
    static constexpr size_t kStripeCount = 65536;
    std::vector<std::mutex> label_locks_;      // id → 哈希条带锁
    std::vector<std::mutex> node_locks_;       // per-node 邻接表锁
    std::mutex              global_lock_;      // 仅入口点/最高层变更瞬间
    struct VisitedListPool { /* per-thread visited 复用 */ };
};
```

- 语义：`add` 与 `add` 并发安全；`add` 与 `search` **可并发**（搜索端对节点的读取用节点级锁保护或宽松读 + 有效负载保证）；删除与搜索并发的精确语义以单测锁定。
- StorageEngine 层移除全局 shared_mutex，仅保护 WAL append（追加天然串行）与 IndexManager。

**5.1.4 API 调整**

- 构造参数：`HNSWIndex(dim, M, ef_construction, distance_fn)` — 距离函数可注入（支持 L2/IP/余弦/量化距离）。
- 新增：`remove(id)`、`mark_deleted` 持久化、`contains(id)`、`reserve(n)`。

### 5.2 Quantizer（量化模块，P0-1 主线）

```cpp
// index/quantizer.h — 抽象 + 三种实现
class Quantizer {
public:
    virtual ~Quantizer() = default;
    virtual Status train(const std::vector<const float*>& samples, size_t count) = 0;
    virtual void encode(const float* vec, QuantCode* code) const = 0;
    virtual float distance(const QuantCode& a, const float* b) const = 0;  // 查询向量 vs 码
    virtual float distance(const QuantCode& a, const QuantCode& b) const = 0;
    virtual size_t code_bytes() const = 0;
    virtual Status save(std::ostream&) const = 0;
    virtual Status load(std::istream&) = 0;
};

class ScalarQuantizer8  : public Quantizer { /* int8，min/max 归一化，可 SIMD 化 */ };
class BinaryQuantizer   : public Quantizer { /* 1-bit 符号位 + popcount Hamming */ };
class ProductQuantizer  : public Quantizer { /* m 子空间 k-means + ADC 查表 */ };
```

| 量化器 | 压缩率 | 距离 | 适用度量 | 实现成本 |
|---|---|---|---|---|
| SQ8 | 4× | int8 点积/L2 近似（或反量化 L2） | 全部 | 低（复用 SIMD） |
| Binary | 32× | Hamming/popcount | IP、余弦（高维 embedding 最佳） | 极低（符号位截断） |
| PQ | 8–32× | ADC 查表 | L2、IP | 中（k-means + 查表） |

**搜索模式（与 HNSW 集成）**：
- 模式 A「图上压缩」：HNSW 节点存 `QuantCode`（不存原向量）→ 遍历距离全用压缩 → 候选集从原向量文件（或内存 buffer）读原向量精排。内存 4–32× 压缩，磁盘索引 V2 的自然前身。
- 模式 B「双份」：图存原向量 + 压缩码；遍历用压缩距离、精排用原向量。实现简单，内存不减。
- **v2 默认实现模式 A 的原向量放内存 + 量化码放图**，并预留 mmap 读取接口（P2 磁盘索引直接复用）。

### 5.3 FilterIndex（过滤字段索引）

```cpp
// index/filter_index.h
class FilterIndex {
public:
    Status add(uint64_t id, const std::vector<ScalarField>& scalars);   // id→内部序号
    void   remove(uint64_t id);
    // 谓词 → bitmap（node 序号位图），供搜索用
    // 返回 bitset；布尔求值：= → 直接取，范围 → 区间合并（等距桶预索引可选）
    const BitSet& bitmap_for(const Clause& c) const;
private:
    struct FieldIndex {
        std::unordered_map<int64_t, BitSet>   ints;    // 或 double→BitSet
        std::unordered_map<std::string, BitSet> strs;
        // 数值范围优化：排序值数组 + 前缀和位图（M2 可选）
    };
    std::unordered_map<std::string, FieldIndex> fields_;
};
```

- 谓词求值 = 位图集合运算（AND/OR），O(1) 判一个 node 是否满足。
- `BitSet` 用 `std::vector<uint64_t>` 手写（避免外部依赖），内存 ~n/8 字节/字段值。
- 范围过滤（`<`/`>`）：v2 第一版线性扫描该字段的排序值（字段基数通常小）；M2 可选加区间桶。

---

## 6. 存储层设计（storage/）

### 6.1 WAL v3 实现要点（在 v1 基础上增量）

- `LogManager` 保持类结构，扩展：
  - `is_valid_op` 增加 `kVectorPutV2 = 0x04`；
  - 新增 `decode_meta(WalEntry) → EntryMeta`（§3.2 编码/解码，放 `storage/entry_meta.h`）；
  - `append` 不变（帧头/CRC 规则一致）。
- **恢复水位**：`iterate()` 返回每个 entry 的 `offset`（已有），快照记录水位 = 快照时 `LogManager::size()`。

### 6.2 Manifest 与 Snapshot

新文件 `storage/manifest.h/.cpp` + `storage/snapshot.h/.cpp`：

```cpp
struct Manifest { uint64_t snap_seq; uint64_t wal_offset; std::string snap_path; };

Status write_manifest(const Manifest&);
Status read_manifest(Manifest*);                     // 读最新一条

Status save_snapshot(const HNSWIndex&, const PayloadTable&, const Quantizer&,
                     uint64_t wal_offset, const std::string& path);
Status load_snapshot(HNSWIndex*, PayloadTable*, Quantizer*,
                     uint64_t* out_wal_offset, const std::string& path);
```

- **PayloadTable**：`std::unordered_map<uint64_t, std::pair<uint64_t,uint64_t>>`（id → WAL offset + len），从 WAL 读取 payload；快照后 id→offset 仍指向原 WAL（WAL 不轮转，offset 稳定）——**简化决策**：快照只序列化图 + 量化码 + scalars，payload 不复制（仍在 WAL）。
- 写快照流程：① 加写锁暂停写入 ② `fsync(WAL)` ③ 写 `.snap.tmp` ④ fsync ⑤ rename ⑥ 更新 MANIFEST（写临时 + rename + fsync）—— 保证「MANIFEST 出现新快照 ⇔ 快照文件已完整落盘」。
- 加载失败处理：快照 CRC 失败 → 回退 MANIFEST 中上一条快照（MANIFEST 保留最近 2 条），或全量重放。

### 6.3 Options 扩展（types.h）

```cpp
struct Options {
    // —— 沿用 v1 ——
    std::string wal_path = "lumina.wal";
    std::string hnsw_path = "lumina.hnsw";          // v2 改为 snapshot 目录
    bool sync_writes = true;
    bool group_commit = false;
    size_t sync_every_n_appends = 0;
    size_t vector_dim = 128;
    size_t hnsw_M = 16;
    size_t hnsw_ef_construction = 200;
    size_t hnsw_ef_search = 50;

    // —— v2 新增 ——
    Metric metric = Metric::kL2;                     // L2 | IP | COSINE
    std::string snapshot_dir = "lumina_snap";
    size_t snapshot_interval_bytes = 256ULL * 1024 * 1024;
    QuantConfig quant;                               // {mode: none|sq8|binary|pq, ...}
    size_t concurrency = 0;                          // 0 = 硬件并发数
};
```

---

## 7. 距离度量扩展（vector/）

- 新增 `ip_distance`（内积，MIPS 用）+ `cosine` 已存在。运行时派发三选一：`Metric → DistanceFn`。
- 量化距离内核：
  - SQ8 L2：int8 向量（或反量化 f32）SIMD 内积；
  - Binary：`popcount(xor)`（`__builtin_popcountll`，AVX-512 `_mm512_popcnt_epi64` 可选）；
  - PQ：查表 + 累加（小 m 时手写循环，无需 SIMD 也能快）。
- 这些函数放 `vector/quantized_distance.h/.cpp`，与现有 `vector_math.*` 并列。

---

## 8. 并发模型总结

| 操作 | 锁 | 说明 |
|---|---|---|
| add / update | label 条带锁（同 id 串行）+ node 邻接锁 + 短持全局锁 | 不同 id 并行 |
| remove | label 条带锁 + 全局删除集合锁 | |
| search | 无写锁；visited 池 per-thread | 与 add 并发安全（宽松读） |
| snapshot | 写锁（暂停写入），fsync 后释放 | 查询不受阻（图是已提交状态） |
| WAL append | 单写者（引擎层串行化） | 追加无锁，offset 单调 |

**验收**：T1 线程并发 add + T2 线程并发 search，结果与串行等价（单测 + tsan）。

---

## 9. API 层设计

### 9.1 C API v2（c_api.cpp 重写）

```c
// 生命周期
int lumina_open(const char* dir, int dim, int metric);      // 替代 lumina_init
int lumina_close(void);

// 写入
int lumina_add(uint64_t id, const float* vec, int dim,
               const char* payload,
               const char** filter_keys, const double* filter_vals, int nfilters);
int lumina_remove(uint64_t id);

// 查询（过滤）
char* lumina_search(const float* query, int dim, int top_k,
                    const char* filter_json,   // 简化：JSON 谓词（{"field":{">=":100}}）
                    int* out_len);
int lumina_get(uint64_t id, char** payload, int* len);

// 维护
int lumina_snapshot(void);
int lumina_compact(void);          // M3
char* lumina_stats(void);          // JSON：节点数/量化模式/ef/内存占用

// 内存管理
void lumina_free_string(char*);
```

- 移除全局单例：句柄式（`lumina_open` 返回 `void*`），支持多 collection。
- 线程安全：每个句柄内部。

### 9.2 Python 绑定（nanobind，M2）

```
python/lumina/
  __init__.py
  _core.abi3.so          # nanobind 编译产物
  collection.py          # 高亮 API
```

```python
import lumina

col = lumina.open_collection("./data", dim=384, metric="cosine")
col.add(ids=[1,2,3], vectors=np.array(..., dtype=np.float32),
        payloads=["chunk-a","chunk-b","chunk-c"],
        filters=[{"cat":"税"}]*3)          # 支持批量
hits = col.search(query_vec, top_k=10, ef=100, filter={"cat":"税"})
col.get(1)          # -> (payload, vector)
col.snapshot()
```

- numpy 批量路径：`add`/`search` 走 C API 的批量扩展（`lumina_add_batch` / `lumina_search_batch`）。
- 依赖：`nanobind`（比 pybind11 编译快 ~5x，绑定声明式）——若环境不便安装，退化为 pybind11，接口不变。

---

## 10. 目录结构（v2 目标态）

```
LuminaStore/
├── CMakeLists.txt
├── include/lumina/
│   ├── engine/          collection.h  query.h  filter.h      (新增)
│   ├── storage/         log_manager.h index_manager.h storage_engine.h
│   │                    manifest.h  snapshot.h  entry_meta.h  (新增)
│   ├── index/           hnsw_index.h  quantizer.h  filter_index.h  (新增/重写)
│   ├── vector/          vector_math.h  quantized_distance.h  aligned_alloc.h
│   └── common/          types.h crc32.h simd_dispatch.h
├── src/
│   ├── engine/          collection.cpp  pipeline.cpp         (新增)
│   ├── storage/         (v1 保留 + manifest/snapshot/entry_meta.cpp)
│   ├── index/           hnsw_index.cpp  quantizer_*.cpp  filter_index.cpp
│   └── vector/          vector_math_*.cpp  quantized_distance.cpp
├── python/lumina/                                          (新增, M2)
├── bench/              storage_bench.cpp vector_bench.cpp
│                       filter_bench.cpp quant_bench.cpp      (新增)
│                       ann_benchmarks/  (接入脚本)           (M3)
├── tests/              test_hnsw_v2.cpp test_quantizer.cpp test_filter.cpp
│                       test_manifest.cpp test_recovery.cpp test_concurrency.cpp
└── tools/              lumina_bench_sift.cpp  (SIFT-1M 评测入口, M3)
```

CMake 新增 target：`lumina_core`（不变）+ `luminastore_shared`（C API）+ `lumina_py`（nanobind 扩展，M2）。

---

## 11. 测试计划

| 测试文件 | 覆盖 |
|---|---|
| `test_hnsw_v2.cpp` | heuristic vs 截断召回对比、标记删除、更新、参数可调、save/load 往返 |
| `test_quantizer.cpp` | SQ8/Binary/PQ 编解码、距离正确性（vs 暴力）、训练收敛、save/load |
| `test_filter.cpp` | 各谓词 bitmap 正确性、in-filter 结果恒满足、post-filter 对照召回 |
| `test_manifest.cpp` | 快照写入/加载、水位重放、快照损坏回退、MANIFEST 原子性 |
| `test_recovery.cpp` | 崩溃点注入：WAL 尾部截断 / 中间损坏 / 快照后新增记录，重启一致性 |
| `test_concurrency.cpp` | 多写多读 vs 串行结果一致（含 tsan 构建） |
| 回归 | v1 四个测试保持通过（WAL v1 文件仍可读） |

每个新模块要求：单元测试 + 一个 micro-benchmark。

---

## 12. Benchmark 计划（ann-benchmarks 方法学）

**固定口径**（写进 README，直接可引用）：
- 单线程、单 CPU；数据集：**SIFT-1M（128d, L2）** 与 **GloVe-100（100d, cosine）**；train/test 分离；recall@10（对 top-100 ground truth）。
- 指标：recall@10–QPS 前沿曲线、构建时间、内存占用、索引文件体积；延迟 p50/p95/p99。

**对照组**：
- LuminaStore v2 自身（ef 网格：50/100/200/400）
- hnswlib（同机、同数据集）
- FAISS（`IndexHNSWFlat`，同机）
- 量化模式对比：none vs SQ8 vs Binary（GloVe cosine 上）vs PQ——输出「量化 × recall」损失曲线（RESEARCH.md P0-1 叙事证据）

**产出**：`docs/benchmarks.md` 更新 + 表格 + 前沿曲线图（Python matplotlib 脚本入库）。

---

## 13. 里程碑与验收标准

### M1 引擎闭环 + HNSW 升级（目标 2-3 周）
- [ ] Collection 完整读写闭环：add/get/remove/update + payload 与 scalars 落盘 + 重启恢复（快照 + 增量重放）
- [ ] HNSW v2：启发式邻居选择、标记删除、update、细粒度锁
- [ ] WAL v3（Op 0x04 + entry_meta 编解码），v1 文件向后兼容
- [ ] 单测：test_hnsw_v2 / test_manifest / test_recovery / test_concurrency 绿
- 验收：`kill -9` 后重启数据一致；SIFT-1M heuristic vs 截断 recall 提升可复现

### M2 量化 + 过滤（目标 2-3 周）
- [ ] Quantizer 三实现（SQ8/Binary/PQ）+ 精排管线（模式 A 内存版）
- [ ] FilterIndex + in-filter / post-filter 双模式 + FilterBench 对照
- [ ] C API v2 全部接口 + Python 绑定（nanobind）
- [ ] 单测：test_quantizer / test_filter 绿
- 验收：GloVe-100 上 Binary 量化内存 32× 压缩、recall 损失 ≤5pp @ 同 QPS；过滤高选择性场景 in-filter vs post-filter 召回对比表

### M3 评测 + 文档（目标 1-2 周）
- [ ] ann-benchmarks 接入脚本 + SIFT-1M / GloVe-100 前沿曲线 + 与 hnswlib/FAISS 对比
- [ ] README 升级（架构图 + 曲线 + 对比表 + 快速开始）、`docs/benchmarks.md`
- [ ] 简历 bullet 复核（RESEARCH.md §7 的量化指标填入真实数字）
- 验收：三个 benchmark 表格与曲线入库；`ctest` 全绿

---

## 14. 风险与权衡

| 风险 | 缓解 |
|---|---|
| 量化召回损失 | 精排 rescore 兜底；提供 none 模式对照；量化只影响粗搜 |
| 启发式邻居选择引入复杂度 | 先以「距离比系数=1.0 + keep_conns」最小实现，再对齐 hnswlib 细节 |
| 细粒度锁与搜索并发正确性 | tsan 单测 + 语义文档；必要时退化为「add 与 search 不并发」声明（hnswlib 同款） |
| Python 绑定环境（nanobind 依赖） | 接口与实现解耦，可降级 pybind11；纯 C API 也可先跑通（ctypes） |
| 快照 + WAL 双写的一致性 | 快照前 fsync WAL + MANIFEST 原子 rename；损坏回退路径单测覆盖 |
| 范围过滤（<、>）性能 | v2 第一版线性扫描字段排序值（字段基数小），M2 加区间桶优化 |

---

## 附录 A：与 v1 的接口差异速览

| 项 | v1 | v2 |
|---|---|---|
| 入口 | `lumina_init(dim)` 全局单例 | `lumina_open(dir, dim, metric)` 句柄式，多实例 |
| payload | 内存 map，不落盘 | WAL meta 落盘 + 快照恢复 |
| 启动恢复 | 全量重放 WAL | 快照 + 增量重放 |
| HNSW 邻居选择 | 距离截断 | 启发式多样化剪枝 |
| 删除/更新 | 不支持 | 标记删除 + update |
| 距离度量 | L2 | L2 / IP / COSINE |
| 量化 | 无 | SQ8 / Binary / PQ + rescore |
| 过滤 | 无 | FilterIndex bitmap + in/post filter |
| 并发 | StorageEngine 全局 shared_mutex | HNSW 条带锁 + per-node 锁 |
| Python | 手写 ctypes | nanobind 原生绑定（numpy 批量） |
| Benchmark | 自建 | ann-benchmarks 方法学 + 外部对比 |
