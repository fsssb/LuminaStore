# LuminaStore 项目构思与实现说明

## 1. 项目一句话定义

LuminaStore 是一个基于 C++20 的高性能持久化向量存储引擎原型，目标是把「可靠落盘」与「向量检索性能」放在同一个工程里统一解决。

---

## 2. 为什么要做这个项目（构思与背景）

在很多向量系统里，常见痛点是：

- 存储可靠性与检索性能分别由不同组件负责，链路复杂；
- 崩溃恢复策略不一致，尾部损坏/中间损坏处理不清晰；
- SIMD 优化常停留在“编译机可用”，缺少真正的运行时派发，存在非法指令风险；
- 工程化细节（对齐分配、编译选项、测试基线）不完善，导致“看起来能跑、线上不稳”。

本项目的出发点就是：在一个可读、可测、可扩展的代码库里，把这些问题系统性打通。

---

## 3. 设计目标

### 3.1 功能目标

- 提供基础 KV/向量写入能力（WAL 持久化）；
- 提供向量距离计算与 ANN（HNSW）能力；
- 支持索引保存/加载；
- 支持崩溃恢复。

### 3.2 非功能目标

- 性能：向量距离路径支持 SIMD 加速；
- 可靠：WAL 帧级校验 + 启动恢复修复；
- 可移植：x86/ARM 下均可编译，SIMD 采用按能力选择；
- 可维护：测试、benchmark、CMake 选项清晰。

---

## 4. 核心架构思路

系统分成三层：

1. `storage`：日志写入、索引重建、崩溃恢复；
2. `vector`：距离函数、HNSW 索引；
3. `common`：状态码、CRC、公共类型等。

其中向量路径采取“编译期构建多 ISA 目标 + 运行时选择核函数”的策略：

- 标量实现：`src/vector/vector_math_scalar.cpp`
- 可选 SIMD 实现：`vector_math_neon.cpp` / `vector_math_avx2.cpp` / `vector_math_avx512.cpp`
- 运行时根据 CPU 能力选择（并非只看构建机器）。

---

## 5. 这次具体做了什么（按主题）

## 5.1 路径与结构规范化

- 统一向量数学实现路径到 `src/vector/`；
- 清理旧单文件实现，改成按 ISA 拆分，更利于按文件单独加编译选项与维护。

## 5.2 SIMD 路径工程化升级

- AVX2/AVX512 增加对齐 fast path：
  - AVX2：32B 对齐走 `_mm256_load_ps`，否则 `_mm256_loadu_ps`；
  - AVX512：64B 对齐走 `_mm512_load_ps`，否则 `_mm512_loadu_ps`；
- NEON 路径保留通用加载，同时在可用编译器上提供对齐提示。

## 5.3 对齐分配器与热路径应用

- 新增 `include/lumina/vector/aligned_alloc.h`：
  - `kVectorBufferAlignment = 64`
  - `AlignedAllocator<T, Alignment>`
  - `AlignedFloatVector`
- HNSW 节点向量与 benchmark 大向量路径接入对齐分配。

## 5.4 CMake 与可移植编译策略

- `DetectSIMD.cmake` 改为编译能力检查（`check_cxx_source_compiles`）；
- AVX2 TU 使用 `-mavx2 -mfma`（或 MSVC `/arch:AVX2`）；
- AVX512 TU 使用 `-mavx512f`（或 MSVC `/arch:AVX512`）；
- 新增：
  - `LUMINA_ENABLE_MARCH_NATIVE`（默认 OFF）
  - `LUMINA_RUNTIME_USE_AVX512`（默认 ON）

## 5.5 运行时派发与稳定性

- 运行时核函数选择改为线程安全初始化（`call_once`）；
- 使用单一 kernel table 指针，避免双函数指针不一致读取；
- x86 检测补齐 OSXSAVE/XGETBV 条件；
- `xgetbv` 采用可移植封装（MSVC intrinsic + GCC/Clang inline asm），规避部分 Clang 环境编译/链接问题。

## 5.6 AVX512 策略

- 当前默认策略是保守的：x86 上优先 AVX2，AVX2 不可用时再考虑 AVX512；
- `LUMINA_RUNTIME_USE_AVX512=OFF` 时，可在编译时屏蔽 AVX512 派发分支；
- 对应代码也做了条件编译，避免 `-Wextra` 下 unused 警告。

## 5.7 测试与基准补齐

- 单测覆盖：
  - SIMD 与 naive 一致性；
  - 维度：127/128/129/512/1024；
  - 空指针与 0 维行为；
- benchmark 覆盖：
  - aligned 与 misaligned 输入；
  - naive vs dispatched；
  - HNSW 查询性能。

---

## 6. 当前行为约定（对外需讲清）

- `l2_distance` / `cosine_distance`（dispatch 入口）：
  - `dim == 0`：分别返回 `0` / `1`；
  - 其余情况下若指针为空：返回 `quiet NaN`。
- `*_naive`：
  - `dim > 0` 要求入参可解引用，否则属于未定义行为；
  - `dim == 0` 不读取元素。

---

## 7. 我们做这个方案的收益

- 降低“编译通过但运行崩”的 SIMD 风险（非法指令、OS 状态不满足）；
- 提高热路径吞吐（对齐 + ISA 专用核）；
- 提升代码可维护性（多 ISA 文件拆分 + 可控 CMake 开关）；
- 提升可信度（单测+bench 支撑，不靠口头性能）。

---

## 8. 还有哪些可继续推进（下一步）

- 增加 CI 矩阵（x86_64 GCC/Clang、MSVC、ARM64）；
- 增加 `LUMINA_PREFER_AVX512`（策略可配置）；
- 完善更大规模数据集下的 recall/latency 曲线与报告；
- WAL/HNSW 文件格式兼容策略文档化（版本演进与迁移手册）。

---

## 9. 对外汇报时“怎么讲”（建议 10 分钟版本）

### 9.1 叙事结构（推荐）

1. **问题**：存储可靠与向量性能难兼顾；
2. **目标**：同一引擎里把 WAL 可靠性 + SIMD/HNSW 性能打通；
3. **方案**：分层架构 + 运行时 SIMD 派发 + 对齐分配；
4. **结果**：可恢复、可测、可 benchmark、可跨平台构建；
5. **展望**：CI 与策略开关进一步产品化。

### 9.2 演示清单（建议现场命令）

```bash
# 1) 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2) 单测
ctest --test-dir build --output-on-failure

# 3) benchmark
./build/vector_bench
./build/storage_bench
```

如需强调 AVX512 策略可控，可补充：

```bash
cmake -S . -B build-no512 -DCMAKE_BUILD_TYPE=Release -DLUMINA_RUNTIME_USE_AVX512=OFF
cmake --build build-no512 -j
```

---

## 10. 一句话总结

LuminaStore 这轮工作的价值，不只是“把功能写出来”，而是把向量系统最容易踩坑的工程化细节（可靠性、SIMD 派发、对齐、构建与测试）做成了可落地、可验证、可继续演进的基础版本。
