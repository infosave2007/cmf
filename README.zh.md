English: [README.md](README.md) · Русский: [README.ru.md](README.ru.md)

# CMF — Cortiq Model Format

CMF 是可审计的模型容器：一个文件可以保存权重、分词器、聊天元数据、任务掩码
和技能覆盖层，并在不依赖大型 ML 框架的情况下运行推理。

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## 状态

CMF v2 是当前磁盘格式。读取器会检查信封、区段边界、张量元数据和哈希；不兼容
的改变必须增加特性位或版本号。Rust crate 的 API 尚未到 1.0，仍可能调整。

项目适合本地推理和格式实验。质量与速度数字取决于模型和测试平台；每项数据的
测量方法请见对应的专题文档。

## 快速开始

安装 CLI，并转换一个小型公开检查点：

```sh
cargo install cortiq-cli
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
cortiq run qwen.cmf --prompt "法国的首都是哪里？" --greedy --no-think
```

已有 GGUF 时直接导入：

```sh
cortiq import-gguf model.gguf --output model.cmf
cortiq verify model.cmf
```

CLI 还提供 `info`、`bench`、`ppl`、`serve`、`skill`、`moe-mask`、`moe-defrag`、
`requant`、`compact`、`sign`、`imagine`、`animate` 和 `ltx-video`。使用
`cortiq <command> --help` 查看参数与限制。

## CMF 文件内容

- 固定的 128 字节信封，负责寻址所有区段。
- 包含架构、量化默认值、聊天元数据和来源的 JSON 文件头。
- 二进制张量目录：dtype、形状、偏移、长度和 `hash64`。
- 按页对齐的权重块，可通过 mmap 映射并就地读取。
- 可选的任务掩码、技能替换张量、分词器字节和稀疏索引。

`cortiq verify` 会在边界错误或哈希不匹配时失败；`python/cmf_reader.py` 是一个
独立的小型读取器，便于检查文件。[CMF v2 规范](docs/CMF_V2_SPEC.zh.md)给出
规范布局。

## 量化

量化按张量选择，因此敏感张量可以保持较高精度，大型矩阵则使用紧凑编码。

| 编码 | 用途 |
|---|---|
| `f16`、`f32` | 归一化、嵌入和精确控制张量 |
| `q8`、`q8_2f` | 高保真权重；`q8_2f` 额外保存输入通道缩放 |
| `q4`、`q4t`、`q4tp` | 常规稠密与 MoE 权重 |
| `q2tp`、`vbit`、`vbit_ro` | 受限体积或混合比特配置 |
| `q1`、`q1t`、`q1s` | 训练二值或实验性的三值/PTQ 路径 |

实验性低比特路径见 [Q1T/PTQ](docs/Q1T_PTQ.md)，编码支持矩阵见
[量化覆盖表](docs/QUANT_COVERAGE.ru.md)。

## 运行时能力

- **CPU 推理：** 可移植的 Rust 实现，权重使用 mmap。
- **GPU 推理：** macOS 原生 Metal，以及在构建和设备支持时使用 wgpu（Vulkan/DX12）。
  设置 `CMF_GPU=1` 请求 GPU。
- **长上下文：** `--o1` 使用固定大小的状态，以可测量的质量变化换取较慢的内存增长；
  请在自己的模型上验证。
- **技能：** 一个共享骨干可携带任务掩码和替换张量覆盖层，未使用技能不需要第二份
  完整模型。
- **MoE：** 任务掩码和物理碎片整理可以减少活动专家；发布前应在留出数据上检查 PPL。
- **投机解码：** 仅在模型元数据和实测接受率表明有益时启用 MTP/draft 路径。
- **服务：** `cortiq serve` 提供本地 OpenAI 兼容 HTTP API。
- **媒体：** `imagine`、`animate` 和 `ltx-video` 在适用的模型指南下运行图像、视频和音频流程。

## 平台与模型系列

| 目标 | 支持 |
|---|---|
| Linux/macOS/Windows | CPU；GPU 通过可用的 wgpu 后端 |
| Apple Silicon | CPU 与 Metal；内存与系统共享 |
| NVIDIA/AMD | 在已测试路径上通过 wgpu 使用 Vulkan 或 DX12 |
| Android/iOS | 参见配套移动项目和 split 指南 |

原生转换覆盖 Qwen、Llama、Mistral、Gemma、Phi、DeepSeek、Kimi Linear、MiniCPM
以及若干 MoE/视频系列。模型限制和已发布文件列在[模型指南](docs/hf/README.md)。
不支持的检查点可以尝试 `import-gguf`；失败时请提交可复现的 issue。

## 专题文档

- [CMF v2 规范](docs/CMF_V2_SPEC.zh.md) — 文件布局规范。
- [格式对比](docs/COMPARISON.zh.md) — 评分标准与证据。
- [技能](docs/SKILLS.zh.md) — 掩码、覆盖层和路由流程。
- [GPU 内核配方](docs/GPU_KERNEL_RECIPES.md) — 后端测量。
- [多 GPU](docs/MULTI_GPU.md) 与[移动端切分](docs/MOBILE_SPLIT.ru.md)。
- [FCD 恢复](docs/RUST_FCD.md) 与[低比特 PTQ](docs/Q1T_PTQ.md)。
- [模型卡片和转换说明](docs/hf/README.md)。
- [Cortiq Spectra](docs/SPECTRA.ru.md) — 双能 X 射线的确定性 CPU 流式着色、
  扫描仪配置、拒绝掩码与测量限制（俄文）。

## 从源码构建

```sh
cargo build --release --workspace
cargo build --release -p cortiq-cli --no-default-features  # 仅 CPU CLI
cargo test --workspace
```

本工作区采用 Apache-2.0。重新分发或报告漏洞前，请阅读 [LICENSE](LICENSE)、
[PATENTS.md](PATENTS.md)、[CONTRIBUTING.md](CONTRIBUTING.md) 和 [SECURITY.md](SECURITY.md)。
预编译版本和校验和发布在 [GitHub releases](https://github.com/infosave2007/cmf/releases)。
