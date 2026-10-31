English: [README.md](README.md) · Русский: [README.ru.md](README.ru.md)

# CMF — Cortiq Model Format

CMF 是可审计的模型容器：一个文件可以保存权重、分词器、聊天元数据、任务掩码
和技能覆盖层，并在不依赖大型 ML 框架的情况下运行推理。

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## CMF 与其他格式对比

下表以百分制为文件格式及其参考运行时打分。每项标准得 0–100 分并有一个权重。
权重由我们设定并列在表中，你可以按自己的需求重新计算总分。生态规模（已发布多少
模型、有多少第三方工具）不是文件本身的属性，因此不计分。

| 标准 | 权重 | CMF | GGUF | safetensors | ONNX | MLX | EXL3 | PyTorch `.pt` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **量化范围**：格式本身定义的编码，从 16 位到低于 1 位 | 15 | **100** | 90 | 30 | 50 | 60 | 70 | 25 |
| **一个文件内混用量化**：任意张量用任意编码，原地重新量化 | 10 | **100** | 90 | 50 | 60 | 70 | 75 | 30 |
| **空权重不占空间**：剪掉的神经元和专家不存储，每层保留自己的宽度 | 15 | **100** | 45 | 30 | 60 | 25 | 25 | 30 |
| **灵活性，一个底座承载多个任务**：任务掩码、技能、专家路由都在一个文件里 | 15 | **100** | 40 | 35 | 20 | 30 | 20 | 35 |
| **自包含文件**：分词器、聊天模板、多组件流水线 | 10 | **100** | 80 | 30 | 45 | 30 | 25 | 25 |
| **完整性与真实性**：内容哈希、校验、签名 | 10 | **100** | 30 | 55 | 35 | 50 | 50 | 10 |
| **零拷贝、按需加载**：mmap，GPU 直接读取映射文件，专家按需调入 | 10 | **100** | 90 | 85 | 40 | 80 | 60 | 40 |
| **安全演进**：旧读取器拒绝它无法执行的文件 | 5 | **100** | 70 | 60 | 85 | 50 | 40 | 40 |
| **独立运行时**：不需要 Python 或 ML 框架，支持 CPU 和 GPU | 10 | 90 | **100** | 30 | 70 | 50 | 15 | 0 |
| **加权总分** | **100** | **99** | 69 | 42 | 49 | 48 | 42 | 26 |

CMF 一列的依据：

- **一个张量目录里有 17 种权重编码。**
  - 浮点：f32、f16、bf16。
  - 8 位：`q8_row`，以及带行、列双尺度的 `q8_2f`。
  - 4 位：`q4_block`、`q4t`、`q4tp`（预测尺度，每权重 4.17 位），以及 `mxfp4`（无损重打包 OCP MXFP4 检查点）。
  - 可变 3–8 位：`vbit` 和 `vbit_ro`。
  - 2 位：`q2tp`。
  - 三值：`q1t`。
  - 二值：`q1`、`q1g`（每权重 1.125 位）和 `q1s`（二值加离群值）。
  - 每权重低于 1 位：二值低秩 `blr`。
- **任意张量用任意编码。** 目录中每个条目都有自己的数据类型。转换时用
  `--tensor-quant 模式=编码` 选择编码。`cortiq requant` 可直接修改现有文件中的
  编码，无需回到原始检查点。
- **空权重不占空间。** 剪掉的 FFN 神经元或 MoE 专家既不存储也不计算。每层有自己
  的宽度和专家数，因为读取器从张量形状读取尺寸，而不是依赖一个全局数字
  （[规范 §11](docs/CMF_V2_SPEC.zh.md)）。当前任务用不到的专家和技能不会从磁盘读取。
- **一个底座承载多个任务。**
  - 任务掩码对每个 FFN 神经元、注意力头、层和专家各只占 1 位。
  - 技能是同一文件中的替换张量，只在使用时才占内存。
  - 技能也可以作为独立的小文件分发，通过哈希绑定到它的底座。
- **一个文件，流水线也不例外。** 分词器和聊天模板存放在文件内。图像编辑模型
  （Transformer、文本编码器、VAE 和调度器）、视频模型和语音模型也都是单个文件。
- **完整性。** 每个张量、每个段都有哈希。`cortiq verify` 校验整条链，可选的
  Ed25519 签名由 `cortiq sign` 生成。损坏的文件会拒绝打开，而不是输出垃圾。
- **加载。**
  - 权重按页对齐，张量按 64 字节对齐，文件通过内存映射读取。
  - 在 Apple silicon 上，GPU 直接从映射读取张量，不做拷贝。
  - MoE 专家在路由器选中时才调入。
- **安全演进。** 每项新能力都对应一个必需特性位。读取器遇到不认识的位会拒绝该
  文件，而不是去猜。
- **运行时。** 用 Rust 编写，不需要 Python，可在 CPU、Metal 和 Vulkan/DX12 上
  运行。另有一个独立的 Python 读取器（stdlib + numpy）。GGUF 文件可用
  `cortiq import-gguf` 直接转换。CMF 在最后一行得 90 分而不是 100 分，因为
  llama.cpp 还有 CUDA、ROCm 和 SYCL 后端。

各项标准以及其他格式得分的依据见
[docs/COMPARISON.zh.md](docs/COMPARISON.zh.md)。

## Metal 与 Vulkan 决策加速

**Cortiq 0.8.0：** 一个 CMF 文件封装可训练技能，通过重建误差在本地做出决策，
无需生成 token。默认使用 CPU；通过 `CORTIQ_DECISION_DEVICE=metal` 或 `vulkan`
选择 GPU。crates.io 默认安装已包含 GPU 支持（需要 Rust 1.88+）。

![Full local CPU / GPU path; each comparison uses the same host](docs/decision/hf/figures/gpu.svg)

[快速入门、自定义技能与 API（英语）](docs/decision/hf/README.md) ·
[测试结果与限制（英语）](docs/decision/hf/GPU.md)

## 状态

CMF v2 是当前磁盘格式。读取器会检查信封、区段边界、张量元数据和哈希；不兼容
的改变必须增加特性位或版本号。Rust crate 的 API 尚未到 1.0，仍可能调整。

项目适合本地推理和格式实验。质量与速度数字取决于模型和测试平台；每项数据的
测量方法请见对应的专题文档。

## 快速开始

安装 CLI，并转换一个小型公开检查点：

```sh
cargo install cortiq-cli --locked
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
cortiq run qwen.cmf --prompt "法国的首都是哪里？" --greedy --no-think
```

已有 GGUF 时直接导入：

```sh
cortiq import-gguf model.gguf --output model.cmf
cortiq verify model.cmf
```

## Liquid AI LFM3（0.8.49）

[LFM3-1.7B](https://huggingface.co/infosave/LFM3-1.7B-cmf) 是 Liquid AI 面向本地
智能体与工具调用的三混合模型（短卷积、全注意力与滑动窗口注意力，262k 上下文），
另提供其预训练 [Base](https://huggingface.co/infosave/LFM3-1.7B-Base-cmf) 与支持图像和视频的
[LFM3-VL-2B](https://huggingface.co/infosave/LFM3-VL-2B-cmf)：

```sh
hf download infosave/LFM3-1.7B-cmf lfm3-1.7b-q8_2f.cmf --local-dir .
cortiq run lfm3-1.7b-q8_2f.cmf --prompt "写一个合并两个有序列表的 Python 函数。" --max-tokens 2048
cortiq run lfm3-vl-2b-q8_2f.cmf --image photo.jpg --prompt "描述这张图片。" --max-tokens 1024
```

转换结果与 Transformers float32 逐位置一致。在 NVIDIA A40 上，8 位文件生成
166 token/s，4 位文件 231 token/s；4096 token 提示词读取约 600 token/s。同一版本
还向对话模板传入 `bos_token`，此前 LFM2/LFM2.5/LFM3 的提示词缺少它。

## Apex-35B-V1.0 与更快的 Qwen3.5-MoE 图（0.8.48）

[Apex-35B-V1.0](https://huggingface.co/infosave/Apex-35B-V1.0-cmf) 是基于
Qwen3.6-35B-A3B 的智能体微调模型，支持工具调用与按需思考：

```sh
hf download infosave/Apex-35B-V1.0-cmf apex-35b-v1.0-q4tp-v2.cmf --local-dir .
cortiq run apex-35b-v1.0-q4tp-v2.cmf --prompt "写一个合并两个有序列表的 Python 函数。" --max-tokens 4096
```

在 NVIDIA A40 上，4 位文件生成 80 token/s，读取 4096 token 提示词约 390 token/s
（首 token 10.5 s）。同一版本加速了 Vulkan 上所有 Qwen3.5/3.6-MoE：q4tp MoE 的
提示词按 256 行分块读取；当 MoE 草稿块只能在 CPU 上运行时默认关闭推测解码；
token 图每层专家库只构建一次（Apex q4tp-v2 相比 0.8.45：生成 52.9 → 80.3 token/s，
4096 token 提示词 172 → 390 token/s）。

## 一条命令按任务压缩模型（0.8.39）

把通用模型变成针对你的提示的专用模型：

```sh
cortiq compress model.cmf prompts.jsonl model-task.cmf --ratio 0.9
```

教师模型回答提示（模型本身、另一个 `.cmf`，或任何 OpenAI 兼容 API：
`--teacher openrouter:qwen/qwen3.8-27b`、`openai:MODEL`、
`api:MODEL --api-base URL`）；任务不使用的 FFN 神经元被移除（按激活照明，或用
`--select dtgma` 训练 DTG-MA 掩码），其贡献以闭式解重拟合进保留的神经元，然后
模型在这些回答上修复；留出回答对结果评分，并由质量门槛决定是否写出文件。MoE
模型会移除任务不会路由到的专家。输出保持源模型的编码格式，量化是单独的一步
（也可以把 `q4tp` 等编码格式作为第四个参数）。GPU 阶段在 NVIDIA（Vulkan）和
Apple Silicon（Metal）上运行。详见[压缩指南](docs/COMPRESS.zh.md)；方法与测量见
[面向任务压缩的文章](docs/research/TASK_COMPRESSION_ARTICLE.zh.md)，大模型上的
完整运行见[Qwen3.8-27B → 14B 逐步记录](docs/research/QWEN38_27B_TASK_COMPRESSION.zh.md)
（得到的模型及其提示和教师数据集：
[infosave/Qwen3.8-27B-Code-14B-Experimental-cmf](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf)）。

## 原生 Qwen Image 编辑

0.6.8 提供 Qwen-Image-Edit-2509 的原生 Rust 流程。默认发布形态是一个
自包含的 `qwen-image-edit-2509-q4tp.cmf`；独立的 transformer、Qwen2.5-VL
文本/视觉编码器、VAE 和 scheduler 文件仍可单独暂存或显式选择。使用下面的
命令合并为一个文件：

```sh
cortiq imagine-pack --bundle qwen-image \
  --out qwen-image/qwen-image-edit-2509-q4tp.cmf
```

Bundle 内含 transformer、文本/视觉编码器、tokenizer、processor、配置、VAE 和
调度器；运行 bundle 不需要旁边再放其它 CMF：

```sh
cortiq imagine qwen-image/qwen-image-edit-2509-q4tp.cmf \
  --image docs/media/fox-512.png \
  --prompt "为狐狸围上一条鲜艳的蓝色针织围巾，同时保留狐狸姿态、雪景和日落背景。" \
  --height 1024 --width 1024 --steps 40 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf.png
```

需要至少一个参考图像；使用多个参考图像时重复 `--image`。如需独立组件，可用
以下显式覆盖：

```sh
cortiq imagine qwen-image/transformer.cmf \
  --text-encoder qwen-image/text_encoder.cmf \
  --vae qwen-image/vae.cmf \
  --scheduler qwen-image/scheduler_config.json \
  --image docs/media/fox-512.png \
  --prompt "将场景变成水彩插画。" \
  --height 1024 --width 1024 --steps 40 --cfg 4 --seed 7 \
  --reference-size 1024 --out watercolor.png
```

`--reference-size 1024` 是 VAE 参考区域 1024² 的平方根，与输出尺寸无关。
在满足设备和内存预算的 Vulkan 设备上，`cortiq imagine` 会自动选择 Qwen
transformer 的 resident forward；各 transformer block 之间保留设备端 hidden
state，每次 forward 末尾只回读一次。CPU 和 Metal fallback 仍可用，预算不满足
时会选择兼容路径。固定获取、打包和配置细节请参阅 [Qwen Image 指南](docs/QWEN_IMAGE.md)。

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
[量化覆盖表](docs/QUANT_COVERAGE.zh.md)。

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
- [按任务压缩](docs/COMPRESS.zh.md) — `cortiq compress` 指南；
  [方法文章](docs/research/TASK_COMPRESSION_ARTICLE.zh.md)与
  [Qwen3.8-27B → 14B 运行记录](docs/research/QWEN38_27B_TASK_COMPRESSION.zh.md)。
- [GPU 内核配方](docs/GPU_KERNEL_RECIPES.md) — 后端测量。
- [多 GPU](docs/MULTI_GPU.md) 与[移动端切分](docs/MOBILE_SPLIT.zh.md)。
- [FCD 恢复](docs/RUST_FCD.md) 与[低比特 PTQ](docs/Q1T_PTQ.md)。
- [MTP FFN 压缩](docs/MTP_COMPACTION.zh.md) — MTP 草稿 FFN 的独立校准与物理
  缩小。
- [Qwen Image 编辑](docs/QWEN_IMAGE.md) — 单文件运行和独立组件暂存。
- [模型卡片和转换说明](docs/hf/README.md)。
- [Cortiq Spectra](docs/SPECTRA.zh.md) — 双能 X 射线的确定性 CPU 流式着色、
  扫描仪配置、拒绝掩码与测量限制。

## 从源码构建

```sh
cargo build --release --workspace
cargo build --release -p cortiq-cli --no-default-features  # 仅 CPU CLI
cargo test --workspace
```

本工作区采用 Apache-2.0。重新分发或报告漏洞前，请阅读 [LICENSE](LICENSE)、
[PATENTS.md](PATENTS.md)、[CONTRIBUTING.md](CONTRIBUTING.md) 和 [SECURITY.md](SECURITY.md)。
预编译版本和校验和发布在 [GitHub releases](https://github.com/infosave2007/cmf/releases)。
