# WAL + 快照：嵌入式向量库怎么做崩溃一致性

> LuminaStore 设计笔记 #2 —— 嵌入式向量库的「kill -9 后数据不丢」是怎么做到的

## 问题

嵌入式向量库（进程内运行）经常被 `kill -9`、断电、容器 OOM 杀掉。用户把文档向量写进去，
崩溃后必须**一条都不能少**（或少掉的必须是「用户还没收到确认」的那些）。

这比服务端数据库更苛刻：没有独立的 WAL 进程，崩溃恢复发生在**同一个进程下一次启动时**。

## 设计：三层保证

### 1. Append-only WAL（唯一真相源）

每次写入（向量 + payload + 过滤字段）编码成一条记录追加到 WAL：

```
[OpType: 1B][CRC32: 4B][PayloadLen: 4B][Payload]
```

- 帧级 **CRC32** 覆盖 Op + 长度 + 载荷——任何半写/损坏都能检测
- **group commit**：`fsync` 可批量（N 次写一次落盘），由调用方权衡吞吐与持久性
- **打开时修复**：文件尾部的半写帧被截断（tail repair）；**中间损坏**（后面还有合法数据）直接报 `Corruption`——绝不静默丢数据

### 2. Snapshot + MANIFEST（启动提速）

纯 WAL 的恢复是**全量重放**（10 万条启动要重放 10 万次）。引入快照：

```
snapshot = 当时的完整 key→offset 索引表 + WAL 水位（字节偏移）
MANIFEST 记录：snap <seq> <wal_offset> <file>
```

恢复 = **加载快照 + 只重放水位之后的 WAL**。水位安全的关键：

1. `snapshot()` **先 fsync WAL**——水位永远 ≤ 已落盘数据
2. 快照写临时文件 → fsync → rename；MANIFEST 同样原子更新
3. 快照损坏（CRC 失败）→ 自动回退全量重放

### 3. 幂等重建

HNSW 图和过滤位图**不是真相源**，是 WAL 的派生视图。恢复时从 WAL 重建它们——
索引损坏不可怕，重放一遍就是。

## 实测：kill -9 循环 6 次

脚本 `bench/reliability/crash_recovery.py`：每轮写入 10 万条（分批 + 周期快照），
中途 **SIGKILL**，重启校验「已持久化数据完好」。

```
[round 0] killed=True OK live=10129 sample_ok=8/8
[round 1] killed=True OK live=18022 sample_ok=8/8
...
[round 5] killed=True OK live=9872  sample_ok=8/8
RESULT: 6/6 rounds data-consistent
```

- `live` = kill 时刻**已持久化**的条数（WAL 恢复）——每次崩溃点不同，数据量不同
- `sample_ok = 8/8`：随机抽样的 payload 全部完整正确——**没有半写、没有错乱**
- 并发压力（4 写 4 读）：5000 条全部入库，已知向量 recall@1 = 1.0

## 设计取舍

| 决策 | 理由 |
|---|---|
| WAL 即数据（payload 留在 WAL 里） | 读少的小 payload，OS page cache 足够；少一套文件格式 |
| 中间损坏报错而非截断 | 无法区分「真损坏」与「实现 bug/篡改」，截断可能静默丢数据 |
| 快照不复制 payload | WAL 不轮转，offset 稳定；快照只存索引表 + 水位 |
| 索引可重建 | 图/位图是派生数据，崩溃一致性以 WAL 为锚 |

## 相关代码

- `src/storage/log_manager.cpp`（WAL 帧格式 / 修复）
- `src/storage/snapshot.cpp` + `manifest.cpp`（快照 + MANIFEST）
- `bench/reliability/crash_recovery.py`（kill -9 测试）
