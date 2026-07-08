English: [README.md](README.md) · Русский: [README.ru.md](README.ru.md)

# CMF — Cortiq Model Format

**用一个共享模型托管众多专精 LLM——全部装进单个文件，可在 CPU 或 GPU 上运行。**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![downloads](https://img.shields.io/crates/d/cortiq-cli.svg)](https://crates.io/crates/cortiq-cli)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![stars](https://img.shields.io/github/stars/infosave2007/cmf?style=flat)](https://github.com/infosave2007/cmf)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infosave2007/cmf/blob/master/LICENSE)

---

## 问题所在

交付一整批面向特定任务的专精模型代价高昂。*N* 个微调版本通常意味着磁盘和内存里各存着 *N* 份完整副本——此外还有松散的 `config.json` / `tokenizer.json` / 适配器附属文件需要保持同步，而且没有任何内建手段能区分损坏文件与完好文件。

## 核心思路

**CMF 只保留一个骨干模型，并在其之上叠加轻量的按技能覆盖层。** 每个技能只存储它实际改动的那些张量；推理时，运行时会用选定技能的张量*取代*骨干模型的对应张量——从不需要单独组装出一个完整模型。于是一整套专家都装进**一个自描述文件**里，能在笔记本电脑上运行，权重直接从磁盘读取（`mmap`，零拷贝），未使用的技能不占用任何内存。

而且专家不仅更省成本——它在自己的任务上还*更强*：在留出数据上实测，叠加在骨干之上的技能相比单用骨干把**任务困惑度降低了 24.9%**（见[规范 §9](docs/CMF_V2_SPEC.md)）。

## 适用人群

- **智能体 / 插件开发者**——一个模型承载 20 个技能（SQL、代码、翻译……），而不必去存储、加载 20 个模型并在它们之间路由。
- **边缘 / 本地部署**——把一个带路由的多技能模型塞进单个模型的内存预算内；权重按需从磁盘分页调入。
- **任何交付量化 LLM 的人**——一个经过完整性校验的文件同时承载权重 **+ 分词器 + 聊天模板**，因此没有附属文件会丢失，损坏也能由逐张量哈希捕获。

## 实际运行效果

```console
$ cortiq run model.cmf --prompt "What is the capital of France?" --greedy
Ready: qwen2 | Task: general | Sparsity: 0%
Prompt: What is the capital of France?
 The capital of France is Paris.
[8 tokens, 33.6 tok/s, finish: stop]
```

## 为什么选 CMF——你能得到什么

- **无需拷贝模型即可添加技能。** 一个骨干 + 若干小的按技能增量：存储开销是 `|backbone| + Σ|deltas|`，而不是 `N × |model|`。
- **启动即刻、内存占用轻。** 权重经内存映射并就地读取；被掩码屏蔽或未使用的权重从不进入内存。
- **磁盘占用更小，且如实标注。** 按张量混合量化——`q8`、`q4`、双字段 `q8_2f`、可变位宽（3–8 bit）——可低至 ~1 byte/param 乃至更低。双字段与可变位宽编解码器在相同文件大小下弥补了 int8→fp16 的大部分质量差距，而且精度取舍都是*实测*得出的，绝非空口宣称。
- **单个文件，无附属文件。** HF 分词器（byte-level BPE）与聊天模板（Jinja）都随模型一同携带——是文件本身定义聊天行为，而非你的运行时二进制。
- **信任这个文件。** 固定的 128-byte 信封加上每张量一个 64-bit 哈希，意味着一个 `.cmf` 要么有效、要么 `open()` 直接返回错误；`cortiq verify` 会校验整条链路。
- **随处可运行。** 一个无依赖的 Rust 核心跑在 CPU 上，另有可选的 GPU 后端（wgpu → Vulkan · Metal · DX12）。
- **一条命令完成转换。** `cortiq convert --model <hf-repo>`——纯 Rust，无需 Python/numpy/torch；模型会（并行）下载并一步量化。

## 横向对比

托管 **N 个任务专家**：

| | N 个完整微调 | 基座 + N 个外部 LoRA | **CMF——一个骨干 + N 个技能** |
|---|---|---|---|
| 磁盘占用 | N × 完整模型 | 基座 + N 个适配器（附属文件） | 一个骨干 + N 个小增量，**一个文件** |
| 分词器 + 聊天模板 | 每份副本各带 / 附属文件 | 附属文件 | **内嵌** |
| 逐张量完整性哈希 | — | — | **有** |
| 冷 / 未使用技能占用的内存 | 已加载 | 已加载 | **0**（用到时才分页调入） |

完整、如实的逐格式对比——GGUF、safetensors、ONNX、PyTorch、GGML、TensorRT，并把各自的取舍讲清楚——见 [docs/COMPARISON.md](docs/COMPARISON.md)。

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
cortiq info  model.cmf
cortiq masks model.cmf
cortiq verify model.cmf     # envelope, sections, per-tensor hashes
```

把模型转换为 `.cmf`——**纯 Rust，无需 Python/numpy/torch**。传入一个
Hugging Face 仓库 id（并行下载）或一个本地模型目录：

```sh
cortiq convert --model Qwen/Qwen2.5-0.5B-Instruct --quant q8    --output model.cmf
cortiq convert --model ./my-hf-checkpoint         --quant q8_2f --output model.cmf
```

或直接导入 GGUF——本地文件，或一个 Hugging Face 的 GGUF **仓库 id**（自动挑选并
下载最合适的 `.gguf`）。所有常见 ggml 量化类型都原生解码（`Q4_0/1`、`Q5_0/1`、
`Q8_0`、`Q2_K`…`Q6_K`、`IQ4_NL/XS`、`BF16`）——无需 Python：

```sh
cortiq import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF --output model.cmf --quant q8
cortiq import-gguf model.gguf                      --output model.cmf --quant q8
```

量化：`q8` · `q8_2f`（双字段，质量/体积最佳）· `q4` · `f16`。
dense、**MoE** 及 **GatedDeltaNet** 模型（qwen2 / qwen3 / qwen3.5 / llama /
mistral / qwen-moe）均可原生转换——包括融合的 qwen3_next / AgentWorld 布局。内置的
Python 转换器（`converter/`）现在仅用于研究性的 v-bit / 校准选项。

运行推理：

```sh
# Interactive chat
cortiq run model.cmf

# Single prompt, greedy decoding, capped length
cortiq run model.cmf --prompt "Write a haiku about memory-mapped files." --greedy --max-tokens 64

# Overlay a specific skill — its replacement tensors are read in place of the backbone
cortiq run model.cmf --prompt "SELECT ..." --skill sql
```

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

读取方**只**通过信封来定位各个区段——绝不靠假设它们的排列顺序。

## 功能特性

- 单文件、可内存映射、自校验的二进制容器。
- 二进制张量目录，张量名与源模型 1:1 对应，并为每个张量附带 64-bit 哈希以检测损坏。
- 按张量混合量化：`f32`、`f16`、`bf16`、`q8_row`、`q4_block`、`q8_2f`、`vbit`。
- 内嵌分词器（与 HF byte-level BPE 对齐）和聊天模板（Jinja，遵循 HF 语义）。
- 按任务掩码（位打包）和预计算的稀疏索引。
- 多技能蜂群：一个骨干 + 按技能的全形状替换张量，在前向传播时叠加；仅追加式增长与压实。
- 可选的多词元预测（MTP）头和专家混合（MoE）FFN 层。
- 分片：一个模型拆分到 `N` 个各自独立有效的 `.cmf` 文件中。
- 无依赖的 Rust 运行时，可运行在 **CPU 和 GPU** 上（可选 `gpu` 特性：wgpu → Vulkan / DX12 / Metal）。
- 提供 Rust（读取器 + 运行时）和 Python（写入器 + 一个仅依赖 stdlib+numpy 的读取器）参考实现。

## 格式概览

完整的规范性说明——信封、header JSON、张量目录、量化布局、掩码、分词器捆绑包、稀疏索引、`hash64`、技能与分片——见 [docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md)。

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
converter/        Python converters for exotic archs (MoE / linear-attention)
python/           dependency-free reader (stdlib + numpy)
docs/             format specification and comparison
```

## 许可与专利

依据 **Apache License, Version 2.0** 授权——见 [LICENSE](LICENSE)。

本软件实现的方法属于三项美国专利申请的主题；详情见 [PATENTS.md](PATENTS.md)。Apache-2.0 第 3 条的专利授予适用于上述三项被引用的申请，从而赋予每位用户一份免版税许可，涵盖本软件按其分发形式必然涉及的专利权利要求。
