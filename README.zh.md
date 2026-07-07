# CMF — Cortiq 模型格式

**用一个自描述文件承载量化后的 LLM——权重、分词器、对话模板、任务掩码以及按技能划分的覆盖层——并配备可移植、零依赖的 Rust 运行时，可在 CPU 和 GPU（Vulkan · Metal · DX12）上运行。**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infosave2007/cmf/blob/master/LICENSE)

English: [README.md](README.md) · Русский: [README.ru.md](README.ru.md)

---

## 为什么选择 CMF

- **单一文件。** 权重、分词器、对话模板、按任务划分的掩码以及按技能划分的增量记录都作为单个 `.cmf` 容器分发——一个分发单元，没有任何附带文件。
- **自描述。** 一个固定的 128 字节信封定位每一个区段；二进制张量目录是布局的唯一可信来源。文件经过校验，而非靠猜测——一个 `.cmf` 要么有效，要么 `open()` 返回错误。
- **mmap、零拷贝、随处运行。** 权重块按页对齐，每个张量都按 64 字节对齐以适配 SIMD。运行时对文件进行内存映射并就地读取权重——冷（被掩码屏蔽）权重不占用任何 RSS。零依赖的 CPU 核心无需任何额外组件；可选的 GPU 后端面向 Vulkan、Metal 和 DX12。
- **量化。** 每个张量的数据类型包括 `q8_row`、`q4_block`、双字段 `q8_2f` 以及可变位宽 `vbit`（3–8 位），可在同一个模型中自由混用。
- **多技能覆盖层。** 一个共享的主干加上按技能划分的全形状替换张量。在前向计算时，运行时读取所选技能的张量*以替代*主干——无需实体化出一个独立的模型。存储规模按 `|backbone| + Σ|deltas|` 增长。

## 容器布局

```
 .cmf file
 ┌──────────────────────────────────────────────────────────┐
 │ Envelope        128 bytes, fixed                          │
 │   magic "CMF\x01" · version · feature bits · section       │
 │   offsets+lengths (header, dir, data, masks, vocab, index)│
 ├──────────────────────────────────────────────────────────┤
 │ Header JSON     arch, quant defaults, chat bundle,        │
 │                 skill registry, provenance                │
 ├──────────────────────────────────────────────────────────┤
 │ Tensor directory   binary 56-byte records:                │
 │                 name · dtype · shape · offset · nbytes ·  │
 │                 hash64  (read without parsing)            │
 ├──────────────────────────────────────────────────────────┤
 │ Weight blob     page-aligned (4096); every tensor 64-byte │
 │                 aligned; quantized; mmap zero-copy        │
 ├──────────────────────────────────────────────────────────┤
 │ Masks / Skills  bit-packed per-task masks (1 bit/neuron)  │
 │                 + per-skill replacement tensors           │
 ├──────────────────────────────────────────────────────────┤
 │ Tokenizer       HF tokenizer.json, verbatim               │
 ├──────────────────────────────────────────────────────────┤
 │ Sparse index    precomputed mask → active groups/heads    │
 └──────────────────────────────────────────────────────────┘
```

读取方**只**通过信封来定位各个区段——绝不假定其排列顺序。

## 特性

- 单文件、可内存映射、自校验的二进制容器。
- 二进制张量目录，张量名称与源模型 1:1 对应，并为每个张量提供 64 位哈希以检测损坏。
- 每个张量可采用混合量化：`f32`、`f16`、`bf16`、`q8_row`、`q4_block`、`q8_2f`、`vbit`。
- 内嵌分词器（与 HF 字节级 BPE 对齐）和对话模板（Jinja，遵循 HF 语义）——对话行为由文件定义，而非由二进制程序定义。
- 按任务划分的掩码（位打包）以及预先计算好的稀疏索引。
- 多技能集群：一个主干 + 按技能划分的全形状替换张量，在前向计算时通过张量来源间接寻址进行叠加；仅追加式增长与压实。
- 可选的多词元预测（MTP）头以及专家混合（MoE）FFN 层。
- 分片：一个模型可拆分为 `N` 个各自独立有效的 `.cmf` 文件。
- 零依赖的 Rust 运行时，可在 **CPU 和 GPU** 上运行（可选的 `gpu` 特性：wgpu → Vulkan / DX12 / Metal）。
- 提供 Rust（读取器 + 运行时）和 Python（写入器 + 基于 stdlib+numpy 的读取器）参考实现。

## 安装

安装命令行工具：

```sh
cargo install cortiq-cli
```

在你自己的 Rust 项目中使用该格式：

```sh
cargo add cortiq-core
```

## 快速上手

检视一个 `.cmf`——架构、张量、量化、掩码和技能：

```sh
cortiq info model.cmf
cortiq masks model.cmf
```

将一个 Hugging Face 检查点转换为 `.cmf`：

```sh
python converter/convert_dtgma_to_cmf.py \
    --model  ./my-hf-checkpoint \
    --quant  Q8_ROW \
    --output model.cmf
```

导入一个 GGUF 模型：

```sh
python converter/import_gguf.py --input model.gguf --output model.cmf
```

运行推理：

```sh
# Interactive chat
cortiq run model.cmf

# Single prompt, greedy decoding
cortiq run model.cmf --prompt "Write a haiku about memory-mapped files." --greedy

# Overlay a specific skill (replacement tensors read in place of the backbone)
cortiq run model.cmf --prompt "SELECT ..." --skill sql
```

## 格式概览

完整的规范性说明——信封、头部 JSON、张量目录、量化布局、掩码、分词器捆绑包、稀疏索引、`hash64`、技能与分片——见 [docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md)。

## 对比

CMF 与 safetensors、GGUF 以及原始 HF 检查点的关系——单文件、自描述、mmap 量化、多技能之间的权衡——见 [docs/COMPARISON.md](docs/COMPARISON.md)。

## 从源码构建

```sh
cargo build --release --workspace
```

可选的跨平台 GPU 后端（wgpu → Vulkan / DX12 / Metal）：

```sh
cargo build --release --workspace --features gpu
```

## 项目结构

```
crates/
  cortiq-core     format reader: envelope, directory, quant, masks, mmap
  cortiq-engine   portable CPU/GPU inference runtime, tokenizer, chat, skill overlay
  cortiq-server   OpenAI-compatible HTTP serving
  cortiq-cli      the `cortiq` command-line tool (inspect/convert/run/serve)
converter/        Python writers: HF → .cmf, GGUF → .cmf
python/           dependency-free reader (stdlib + numpy)
docs/             format specification and comparison
```

## 许可证与专利

依据 **Apache License, Version 2.0** 授权——见 [LICENSE](LICENSE)。

本软件实现了作为三项美国专利申请标的的方法；详情见 [PATENTS.md](PATENTS.md)。Apache-2.0 第 3 条的专利授予适用于这三项被引用的申请，为每位用户提供一份免版税许可，覆盖本软件按其分发形式必然会侵犯的专利权利要求。
