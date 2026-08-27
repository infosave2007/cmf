English: [README.md](README.md) · Русский: [README.ru.md](README.ru.md)

# CMF — Cortiq Model Format

**一个文件，装着权重、分词器和聊天模板，能自检完整性，且不需要任何 ML 框架就能运行。**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![downloads](https://img.shields.io/crates/d/cortiq-cli.svg)](https://crates.io/crates/cortiq-cli)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

不需要 torch、BLAS、ONNX，不需要装 CUDA，也不需要 C++ 工具链。一个小巧的 Rust
内核：CPU 到处能跑，GPU 走原生 Metal 和 wgpu（Vulkan / DX12）。权重内存映射、
就地读取。一个开关就能把模型的注意力换成常量内存算子——不重训，也不改动任何一个
权重。

## 上手试试

```sh
cargo install cortiq-cli          # 或者从发布页拿一个预编译二进制

cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
cortiq run qwen.cmf --prompt "法国的首都是哪里？" --greedy --no-think
```

```console
Ready: qwen3 | Task: general | Sparsity: 0%
法国的首都是 **巴黎**。
[10 tokens, 40.1 tok/s, finish: stop]
```

已经有 GGUF？`cortiq import-gguf <文件或仓库> --output model.cmf`。

什么都不用装就能试：[转换模型](https://huggingface.co/spaces/infosave/cmf-converter) ·
[生成图片](https://huggingface.co/spaces/infosave/cmf-imagine) ·
[看看视频](https://huggingface.co/spaces/infosave/cmf-animate)

## 与其他格式的百分制对比

八项标准，各自加权，每个格式按 0–100 打分。完整矩阵和每一格的理由见
[docs/COMPARISON.zh.md](docs/COMPARISON.zh.md)。

| | CMF | GGUF | safetensors | ONNX | PyTorch | GGML | TensorRT |
|---|---:|---:|---:|---:|---:|---:|---:|
| **加权总分 /100** | **80** | **86** | 53 | 56 | 45 | 55 | 52 |
| *去掉「生态」这一项* | **97** | 83 | 41 | 48 | 33 | 63 | 50 |

总分是 GGUF 赢，这很应该：一百分里有二十分是生态，而那一项 CMF 只有 15 分，
GGUF 是 100 分。一个作者，2026 年 7 月才首次公开发布。

论容器本身的性质，CMF 领先，97 比 83：每个张量都带完整性哈希——别的单文件格式
都没有；专家模型住在文件内部、共用同一条哈希链；量化阶梯一路做到三值。

两个数字都是真的。如果你要的是今晚就能在别人验证过的硬件上跑起来的模型，那是
GGUF。如果你要的是一个可审计、装着 N 个专家、并且能自证完整性的文件，那才是这
个东西。

## 文件里有什么

固定的 128 字节信封，其后各段只能通过它寻址，绝不靠假定的顺序：

| 段 | 内容 |
|---|---|
| header JSON | 架构、默认量化、chat bundle、技能注册表、来源信息 |
| 张量目录 | 56 字节记录：名称、dtype、形状、偏移、字节数、`hash64` |
| 权重块 | 按页对齐，映射后就地读取 |
| 技能 | 任务掩码与每个技能的替换张量 |
| 分词器 | 原封不动的 Hugging Face 文件 |

```sh
cortiq verify model.cmf     # 信封、各段、每一个张量哈希
cortiq info   model.cmf     # 架构、张量、量化、技能
cortiq sign   model.cmf     # 基于 SHA-256 的分离式 Ed25519 签名
```

一个 `.cmf` 要么有效，要么 `open()` 大声失败——截断和位腐都能被抓住。此外还
承载：MTP 头、MoE 层、全局与滑窗混合注意力、双 RoPE/YaRN、只追加的技能增长，
以及切分成 N 个各自独立有效的文件。

**你不会被锁死。** `python/cmf_reader.py` 是一个完整的读取器，约 300 行，只用
标准库加 numpy，照着规范写成，与 Rust 运行时不共享任何代码：

```python
from cmf_reader import CmfReader
r = CmfReader("model.cmf")
w = r.tensor("model.layers.0.mlp.gate_proj.weight")   # np.ndarray，已反量化
assert r.verify() == []                               # 所有张量哈希都对得上
```

规范正文：[docs/CMF_V2_SPEC.zh.md](docs/CMF_V2_SPEC.zh.md)。

## 量化

逐张量、可混用——同一个文件里注意力保持 q8，FFN 压到 q4 完全可以。

| 量化 | 比特/参数 | 说明 |
|---|---|---|
| `f16` | 16 | 不量化 |
| `q8` | 8 | 按行的缩放 |
| `q8_2f` | 8 | 按行**并且**按列的缩放——同样大小，质量更好 |
| `q4` · `q4t` | 4.5 | 分块 / 交错瓦片 |
| `q4tp` | **4.17** | 带预测缩放的 `q4t`——小 7%，误差 +0.1% |
| `q2tp` | ~2 | MoE 的 2/4 混合档 |
| `vbit` | ~4.25 | 可变 3–8 比特 |
| `q1t` | 2.25–3.5 | 免训练三值 + 稀疏离群覆盖层（[文档](docs/Q1T_PTQ.md)） |
| `q1` | 1.5 | 面向**以二值训练**出来的检查点（Bonsai / BitNet） |

**Granite 4.2 已获得显式支持。**公开的
[3B/8B/30B Q4TP 与 Q8_2F 文件](https://huggingface.co/infosave/Granite-4.2-cmf)
携带 IBM 的精确 attention multiplier 和官方聊天模板。Q4TP 会把 100,352-token
embedding 与独立的 `lm_head` 保留为 Q8_2F；完整 Q8_2F 则作为更高质量的对照档。
当稠密模型大于显存时，Cortiq 会选择固定的 GPU 层前缀，并在 CPU 上运行 mmap
尾部，而不是让全部权重反复流经驱动。在 10 核 M4 上，3B 文件的稳定解码速度为
34.8 tok/s（Q4TP）和 11.2 tok/s（Q8_2F）。

一句话讲 `q4tp`：一个 `q4t` 瓦片为 32 个权重花掉 16 比特存 f16 缩放，占文件的
11%。把这个缩放改成按行几何阶梯上的 5 比特档位，代价是在行内中位离散度上
**多 0.1% 的相对误差**。已有文件可以就地转换，不需要原始检查点：

```sh
cortiq requant model.cmf --output model-q4tp.cmf --quant q4tp
# KAT-Coder-V2.5：12.65 → 11.80 GB（19254 个张量），2 分钟
```

## O(1) 注意力

`--o1` 把某一层的 softmax 注意力换成固定大小的状态：几个精确的锚点键、一段精确
的近期窗口，再加上对更久远内容的地标草图，全部落在同一个共享分母下。**权重不
变**——这个开关只在文件头里记一条提示。

Qwen3.5-4B（转换了 8 个 softmax 层），Apple M4：

| 上下文 | `--o1 off` | `--o1 all` | 解码 |
|---:|---:|---:|---:|
| 543 | 141.0 MB | **124.1 MB** | 15.7 → 16.5 tok/s |
| 1055 | 174.5 MB | **124.1 MB** | 15.5 → 16.5 tok/s |
| 4127 | 380.3 MB | **124.1 MB** —— 少 3.1 倍 | 8.2 → 10.7 tok/s |

任何长度都是这个常数。它替掉的 KV 每个 token 大约长 64 KiB，所以两条曲线在约
290 个 token 处相交：低于此处 `--o1` 反而多花几 MB，高于此处就只有节省。

**代价是什么。** 在留出的 wikitext 上，困惑度在 Qwen3.5-4B 上升到 **1.13 倍**，
在 Qwen3-0.6B（28 层全转）上升到 **1.30 倍**。模型里 softmax 注意力占比越高，
代价越大。这是一个「内存换质量」的旋钮，不是白捡的便宜——在你自己的模型上测：

```sh
cortiq ppl model.cmf --file wiki.txt --o1 all   # 会把精确注意力的基线并排打出来
cortiq run model.cmf --o1 all|deep12|off        # 或者加载时再决定
```

`cortiq fcd` 能用一次有界的原生训练把代价找回来一部分——不用 Python，也不用
ML 框架。

## 对比 llama.cpp

Qwen2.5-0.5B-Instruct，Apple M4，两边都用精确注意力，从全新进程交错运行，各自
取自己最好的线程数。`cortiq bench --core` 与 `llama-bench` 的口径一致。

| | `llama.cpp` (q8_0) | CMF (q8) | Δ |
|---|---|---|---|
| tg128，CPU，他们最好的 `-t 6` | 165.5 tok/s | 151–158 | −5% |
| tg128，CPU，他们的默认 `-t 4` | 129.4 tok/s | 151–158 | **+18%** |
| tg128，他们的 Metal `-ngl 99` | 150.9 tok/s | 151–158（CPU） | **CMF 的 CPU ≥ 他们的 GPU** |
| pp512，CPU | 1168 tok/s | 1017–1051 | −12% |
| pp512，GPU | 3333–3396 | 2742–3215 | 最好对最好 −5% |
| 相对自身 f16 的 PPL | 几乎无损 | +0.38% | 相当 |
| 文件大小 | 644 MB | **479 MB** | **−26%** |

复现：`cortiq bench --json --core`。

## 一个主干，很多专家

要交付 N 个微调模型，通常意味着 N 份完整副本。CMF 只保留一个主干加上每个专家一
个很小的技能：技能只存它替换掉的那些张量，运行时用它们代替主干的张量，而没被
用到的技能占用**零内存**。磁盘上是 `|主干| + Σ|技能|`，而不是 `N × |模型|`。

在自己的任务上，技能相对它所依附的主干把困惑度降低 **24.9%**（留出数据，
[规范 §9](docs/CMF_V2_SPEC.zh.md)）。

```sh
cortiq skill add ...                            # 从捐赠检查点烘焙
cortiq run model.cmf --prompt "SELECT ..." --skill sql
cortiq route model.cmf --prompt "..."           # 或者让它自己选，用 `explain` 看理由
```

把三个来自公开微调的真实技能烘进一个 0.5B 文件里，连同各种失败模式：
[docs/SKILLS.zh.md](docs/SKILLS.zh.md)。

**MoE 专家。** 专家的使用与任务强相关——代码和散文路由到几乎不相交的集合
（top-64 的 Jaccard 只有 0.25）。`cortiq moe-defrag` 会把某个任务从不使用的专家
物理丢掉：一个 34.7B 的代码模型从 **19.6 GB 降到 12.7 GB（−35%）**，代码困惑度
只涨 2.8%；在一台 24 GB 的 MacBook 上，完整模型会换页，而这个专家版装得下，解码
**快 1.8 倍**。`cortiq moe-mask` 则把同样的限制烘成可切换的任务掩码——一个文件，
`run --task coder`，与物理裁剪逐 token 一致。
[docs/KAT_CODER.md](docs/KAT_CODER.md)

## 投机解码

带 MTP 头的模型用它来打草稿，然后一次批量提交验证整条链。**对 q4tp 文件的贪心
解码默认开启。** 一个监视器会把「每轮 token 数」和普通 token 做对比，连输四轮
就停掉投机、过一阵再试——所以能占到便宜的提示词保住便宜，占不到的就按普通速度走。

- Qwen3.8-27B q4tp，RTX 5090：**76 tok/s，普通路径 48.5**，草稿接受率 90%
- 同一个模型在 M4 mini（24 GB）：普通 6.7，一段代码 **12.2**；447 token 的提示词 **11.6 秒**出首 token
- `CMF_VERIFY_I8=0` 让输出与普通路径逐位一致，`CMF_GRAPH_SPEC=0` 关闭投机

这个解码是**卡在总线上**的：同一张卡上两个进程合计 52.8 tok/s，单进程 48.8；把
matvec 的算术全部剥掉，同一个 token 的耗时相差不到 4%。所以能动的杠杆是少读字
节，或者把这次读摊到更多 token 上，而不是写更快的 kernel。
[docs/GPU_KERNEL_RECIPES.md](docs/GPU_KERNEL_RECIPES.md)

## 它还能生成画面

同一个容器，同一个二进制，推理时没有 Python。

| `cortiq animate` —— 视频**连同它的声音** | `--first-frame` —— 从一张图继续 |
|---|---|
| ![戴厨师帽的柯基在颠煎饼](docs/media/corgi.gif) | ![同一段视频，从一张静帧续出来](docs/media/keyframe.gif) |

声音不是事后配上去的：它和视频在同一个打包序列里去噪，走自己的流匹配时间表，
所以出来就是同步的。

- **`cortiq animate`** —— MiniMax-H3 + Turbo LoRA。512×288，39 帧，四步，出自一个 23.9 GB 的文件（参考实现的目录树有 124.4 GB）。一张 RTX 5090：**60.2 秒**。
- **`cortiq ltx-video`** —— LTX-2.5，21B 的 DiT，带一路联合音频流。八步，或者用 `--two-stage` 换细节。`--lora` 在运行时挂载适配器，`--ref` 最多可用五张参考图做条件。[docs/LTX.md](docs/LTX.md)
- **`cortiq imagine`** —— Lumina-Image 2.0，19 GB 的 diffusers 目录树装进 **3.2 GB** 的 `.cmf`。M4：256×256、30 步约 37 秒。

<img src="docs/media/fox-512.png" width="340" alt="雪林中的红狐">

## 在手机上

同一种格式也跑在 Android 和 iOS 上：**[Cortiq: Local AI Models](https://play.google.com/store/apps/details?id=ai.cortiq.cmf_mobile)**
把这套运行时作为原生库带上手机——在设备上直接与 `.cmf` 对话，在手机本地把
Hugging Face 仓库转换成 CMF，并通过同一套 OpenAI 兼容 API 把已加载的模型提供
给局域网。

- 功能介绍 —— **[huggingface.co/spaces/infosave/cortiq-mobile](https://huggingface.co/spaces/infosave/cortiq-mobile)**
- 源码 —— [github.com/infosave2007/cmfmobile](https://github.com/infosave2007/cmfmobile)（Apache-2.0）

再配合桌面端的 `cortiq worker`，手机就能跑比自身内存更大的模型：34.7B 的 MoE，
在仅剩 2 GB 可用内存的手机上跑到 **16.3 tok/s**。

## 服务

`cortiq serve` 说的是 OpenAI API，现有客户端不用改就能用。

```sh
cortiq serve model.cmf --port 8080     # 另外在 / 上有一个网页面板
```

`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/healthz`，支持流式。
如实划定范围：**请求是串行处理的**，而且**没有鉴权**——这是本地优先的服务器，
不是多租户网关。

## 命令

| 命令 | 作用 |
|---|---|
| `convert --model <hf 仓库\|目录>` | HF 检查点 → `.cmf`（原生 Rust） |
| `import-gguf <文件\|hf 仓库>` | GGUF → `.cmf`，覆盖常见的 ggml 量化 |
| `run` · `serve` | 对话 / 单次提问；OpenAI 兼容服务器 |
| `info` · `verify` · `masks` · `diff` | 查看、校验完整性、比较版本 |
| `bench` · `ppl` | tok/s 与内存；teacher-forced 困惑度 |
| `requant` · `compact` | 就地更换量化；压紧容器 |
| `skill add` · `list` · `route` · `explain` | 烘焙、列出并路由专家 |
| `moe-defrag` · `moe-mask` | 物理丢弃不用的专家，或把它做成可切换的 |
| `fcd` | 面向 `--o1` 模型的恢复训练器 |
| `sign` | 分离式 Ed25519 签名 |
| `imagine` · `animate` · `ltx-video` | 图片、带声音的视频、LTX-2.5 |
| `imagine-pack` · `animate-pack` · `ltx-pack` | 把参考目录树打包成一个 `.cmf` |
| `worker` · `peers` | 把层提供给另一台机器；在局域网里找到它 |

`cortiq <命令> --help` 会说明每一个开关。

## GPU

```sh
CMF_GPU=1 cortiq run model.cmf
```

后端自动选择——Linux/Windows 上是 Vulkan，DX12 作后备，macOS 上是 Metal。
**打开 GPU 绝不会让你变慢**：对每一类算子，引擎在启动时把两条路都测一遍，留下
更快的那条（`CMF_GPU_PROBE=0` 表示无条件信任设备）。

无论 Metal 还是独立显卡，整个 token 都作为一张图执行、只回读一次：隐藏状态在所
有层之间都留在设备上，注意力也在设备上算，路由式 MoE 的路由器、top-k 选择和被
选中的每一个专家都在同一次提交里跑完。实测：

| | |
|---|---|
| Bonsai-27B q1，RTX 4090 | **40 tok/s**（是其 CPU 路径的 7.7 倍），4K 上下文仍有 38 |
| KAT-Coder 34.7B-A3B，RTX 5090 | **32.8 tok/s**，其 32 核宿主 CPU 是 14.4 |
| Nanbeige4.2-3B，无风扇 MacBook Air M4 | **22.4 tok/s**，提示词吞吐 181 tok/s |

输出与 CPU 路径在分布上等价，但不是逐位相同——浮点归约的顺序不一样，任何 GPU
卸载都是如此。

**多张 GPU** —— 两个开关对应两个不同的问题。`serve --gpus N` 在每张卡上放一份
完整副本（2×RTX 5090，34.7B 的 MoE：单请求 115.3 tok/s，两个请求合计
**218.5**）。`run --gpus N` 把层栈切开分到多张卡上，是给比单卡更大的模型用的
——它买到的是空间而不是速度，对本来就装得下的模型还要倒贴几个百分点。`--peer`
把同样的切分放到网络上。[docs/MULTI_GPU.md](docs/MULTI_GPU.md)

## 你的模型能跑吗？

原生转换支持：qwen2 · qwen3 · qwen3.5（含融合后的 qwen3_next）· llama ·
mistral · qwen-moe · gemma / gemma-2 / gemma-3 · gemma-4 稠密 12B/31B 与 MoE
26B-A4B · gemma-3n E4B · phi-3 / phi-4 · DeepSeek-R1 蒸馏版 · DeepSeek-V2 MLA ·
Kimi Linear 48B-A3B · MiniCPM3 · MXFP4 打包的检查点。暂时不支持：Kimi-K3 独有的
那部分，要等它的建模代码公开。

其他的——试试 `import-gguf`。如果它拒绝了，那是一个值得提的 bug。

## 安装与构建

```sh
cargo install cortiq-cli                 # 命令行工具
cargo add cortiq-core                    # 或者在你自己的 Rust 代码里用这个格式

cargo build --release --workspace                            # 默认带 wgpu 后端
cargo build --release -p cortiq-cli --no-default-features    # 纯 CPU
```

预编译二进制覆盖 Linux x86-64、macOS（Apple Silicon 与 Intel）、Windows
（x86-64 与 ARM64）以及 `aarch64-linux-android`，每个都附 `.sha256`：
[最新发布](https://github.com/infosave2007/cmf/releases/latest)。

```
crates/cortiq-core     格式读取：信封、目录、量化、掩码、mmap
crates/cortiq-engine   可移植的 CPU/GPU 运行时、分词器、对话、技能
crates/cortiq-server   OpenAI 兼容的 HTTP 服务
crates/cortiq-cli      `cortiq` 命令
python/                参考读取器——只用标准库和 numpy
docs/                  规范、对比、模型实操
```

## 状态

**格式是已经定下来的那部分。** 它是 v2：读取器只通过信封寻址，文件头里不认识的
字段一律忽略，破坏性变更要付出一个特性位或者一次版本号提升的代价——绝不会悄悄
改变含义。今天写出的 `.cmf` 以后仍然读得出来，而 `cortiq verify` 就是这份契约。

各 crate 的 API 在 1.0 之前还可能变动。首次公开发布是 2026 年 7 月，作者一人。
所有变更都在 [CHANGELOG.md](CHANGELOG.md)。

Bug 与需求：[提 issue](https://github.com/infosave2007/cmf/issues)。
安全问题：**请不要**公开提 issue，见 [SECURITY.md](SECURITY.md)。
转换不了的模型是 bug 报告，不是用户的错。

> **Hub 集成。** `.cmf` 下载计数和 Cortiq 代码片段的上游改动已于 2026 年
> 8 月 26 日合并：
> [huggingface.js#2354](https://github.com/huggingface/huggingface.js/pull/2354)。
> 它已经进入发布的 `@huggingface/tasks` 包，但 Hub 的 production UI 与统计
> worker 尚未部署这项变更：截至 8 月 27 日，CMF 仓库仍显示 `0` 下载，也还没有
> “Use with Cortiq” 入口。

## 许可

**Apache-2.0**（[LICENSE](LICENSE)）——随便用、随便改、也可以商业分发。

本软件实践了作者四项在审美国专利申请中主张的方法（[PATENTS.md](PATENTS.md)）。
Apache-2.0 第 3 条授予你一份永久、全球、免版税的许可，覆盖本软件在分发形态下
必然触及的那些权利要求：**运行、fork 和分发本软件都在覆盖范围内**，该授权只在
你就专利起诉本项目时失效。这份授权的范围是本作品本身；如果你要用别的语言独立
实现这个容器，请发邮件到 urevich55@gmail.com——可以提供给实现者的授权。

设计的来处，并且在「已测量的」和「仅是比喻的」之间划了硬线：
[CMF 背后的 VMF/NVG 原则](VMF_principles_in_CMF.zh.md)
（[English](VMF_principles_in_CMF.md) · [Русский](VMF_principles_in_CMF.ru.md)）。
