English: [README.md](README.md) · Русский: [README.ru.md](README.ru.md)

# CMF — Cortiq Model Format

**一种单文件 LLM 格式——它的注意力内存不再随上下文增长。**

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![downloads](https://img.shields.io/crates/d/cortiq-cli.svg)](https://crates.io/crates/cortiq-cli)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/infosave2007/cmf/blob/master/LICENSE)

一个 `.cmf` 文件把权重、分词器和聊天模板装在一起，能自校验完整性，并直接从磁盘
内存映射。运行时是一个小巧的 Rust 核心，底下没有任何 ML 框架——不用 torch、不用
BLAS、不用 ONNX、不用装 CUDA、不用 C++ 工具链——在所有平台上跑 CPU，源码构建时还
能通过 wgpu（Vulkan / DX12 / Metal）跑 GPU。转换一个模型只需一条命令，无需 Python。

它的不同之处在于：**只用一个开关，你就能把模型的注意力转换成常量内存的流式
算子**——无需重训练，权重逐字节不变——于是长对话不再比短对话更费内存。

## 上手试试

```sh
# prebuilt binary: github.com/infosave2007/cmf/releases/latest
# or, with a Rust toolchain:
cargo install cortiq-cli

cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
cortiq run qwen.cmf --prompt "What is the capital of France?" --greedy --no-think
```

```console
Loading model: qwen.cmf
Ready: qwen3 | Task: general | Sparsity: 0%

Prompt: What is the capital of France?

The capital of France is **Paris**.
[10 tokens, 40.1 tok/s, finish: stop]
```

`convert` 会从 Hugging Face 拉取 checkpoint（分片并行下载）、做量化，并写出一个
自包含的文件——纯 Rust 实现，不用 torch，不用 numpy。已经有 GGUF 了？
`cortiq import-gguf <file-or-repo-id> --output model.cmf` 同样能原生读取。

`run` 会套用文件里存着的聊天模板，所以这是一次真正的对话轮次，模型会自行停止。
Qwen3 是推理模型——去掉 `--no-think`，它会先展示 `<think>` 推理过程。`--raw` 则
完全跳过模板（续写模式）。`Task` 和 `Sparsity` 反映的是技能覆盖层；没有选中技能
时它们显示 `general` / `0%`——[技能](#多个专家共用一个骨干)详见下文。

**它能跑你的模型吗？** 目前可原生转换：qwen2 · qwen3 · qwen3.5（包括融合的
qwen3_next / AgentWorld 布局）· llama · mistral · qwen-moe——涵盖 dense、MoE 和
GatedDeltaNet。尚不支持：gemma、phi、deepseek。其它模型请试 `import-gguf`——如果
它拒绝了，那就是一个值得提 issue 的 bug。

## 接入你现有的工具链

`cortiq serve` 说的是 OpenAI API，因此现有的客户端和 SDK 无需改动即可工作——把
它们指过来就行：

```sh
cortiq serve qwen.cmf --port 8080        # + a web dashboard on /
```

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "cmf",
  "messages": [{"role": "user", "content": "Explain mmap in one sentence."}]
}'
```

`/v1/models`、`/v1/completions` 和 `/healthz` 也都在，流式输出（`"stream": true`）
可用。`model` 字段是 schema 要求的必填项，但不会拿去匹配任何东西——你的客户端发
什么就发什么。

部署前请如实划定它的适用范围：**请求是串行处理的**（每个模型同一时刻只处理
一个），而且**没有任何鉴权**——这是一个本地优先的服务器，不是多租户网关。不要把
它暴露到你不信任的网络上。

## 为什么选 CMF

### 不再随上下文增长的注意力

通常，你每往对话里加一个词元，就会往 KV 缓存里永久地添一笔。`--o1` 把某一层的
softmax 注意力换成一个流式算子，它转而维护一份**固定大小的状态**：若干精确的锚点
键、一个精确的近期窗口，以及一份覆盖更早内容的地标草图，全部共用同一个 softmax
分母。转换是瞬时的，而且**权重完全不变**——这个开关只是在 header 里记下一个提示。

实测于 **Qwen3.5-4B**（24 个 GatedDeltaNet 层 + 8 个 softmax 层；`--o1 all` 转换
其中那 8 个；16 个 query head / 4 个 KV head，head_dim 256；q8_2f）。Apple M4，
每次运行之间让机器充分降温：

| 上下文 | 注意力内存，`--o1 off` | `--o1 all` | 解码，`off` → `all` |
|---:|---:|---:|---:|
| 543 | 141.0 MB | **124.1 MB** | 15.7 → 16.5 tok/s |
| 1055 | 174.5 MB | **124.1 MB** | 15.5 → 16.5 tok/s |
| 4127 | 380.3 MB | **124.1 MB** — 少 3.1× | 8.2 → 10.7 tok/s |

**任何上下文长度下都是 124.1 MB**——这正是全部意义所在。它可以拆成两块：循环层的
一个常量基底，加上顶替 softmax 层 KV 缓存的固定 **18.8 MB**。不这么做的话，那份
KV 会以约 64 KiB/token 的速度增长，因此两条曲线大约在 **290 词元** 处相交：在此
之下，`--o1` 要让你多花几 MB；在此之上，它就只有节省——4k 时少 3.1×，按外推 32k
时约少 17×（状态是常量，所以这个比值会一直往上走；我们实测到 4k——在你自己的机器
上跑 `cortiq bench model.cmf --ctx 32768`）。

**它的代价。** 草图是一种近似，代价要用质量来付：Qwen3.5-4B 上困惑度上升
**1.13×**，Qwen3-0.6B（28/28 层全部转换）上升 **1.30×**——这是在留出的 wikitext
上、经由真实的流式内核、在最苛刻的区段测得的（地标由一段 256 词元的 prefill 封
定，只对漂移行计分）。模型里 softmax 注意力占得越多，`--o1` 的代价就越大：混合
架构有循环层来承载长程状态，而纯注意力模型只能让草图独自扛下全部工作。请把
`--o1` 当作一个内存/质量的旋钮，而不是白捡的便宜。这个代价不会随上下文增长——
状态也不会。这些都别只听我们说；请测你自己的模型：

```sh
cortiq ppl model.cmf --file wiki.txt --o1 all
```

它会经由真实的流式内核给转换后的模型打分，并在旁边打印出在完全相同的词元上用
精确注意力得到的基线，因此这个比值是一次同口径的实测，而不是一句宣称。

如果这个代价对你的场景来说太高，`cortiq fcd` 能用一趟有界的原生训练把其中一部分
找回来——见 [O(1) 深入解析](#o1-深入解析)。我们还没有为它公布干净的前后对比数字。

把评价轴说清楚：CMF 并不打算在每秒词元数上赢过 `llama.cpp`，今天也确实没赢。它压
平的是开销随上下文增长的曲线：内存那一列是其中一半，解码那一列是另一半。随着上下文变长，
精确注意力从 15.7 tok/s 衰减到 8.2 tok/s，而 `--o1` 稳在约 16.5。如果短提示下的
单流吞吐是你唯一的约束，那就用 `llama.cpp`。要用 CMF，是为了长上下文那条尾巴。

### 一个文件，别无附属

分词器（HF byte-level BPE）和聊天模板（Jinja）都随模型一起装在**文件内部**——
GGUF 也是这么做的，而且这么做是对的：定义聊天行为的是文件本身，而不是你的运行时
二进制，也就没有附属文件会丢失、会悄悄失去同步。`.cmf` 在此之上加的是完整性：
固定的 128-byte 信封加上每张量一个 64-bit 哈希，意味着一个 `.cmf` 要么有效，要么
`open()` 就大声报错。它能检测截断和位腐；它不是签名。

```sh
cortiq verify model.cmf     # envelope, sections, every tensor hash
cortiq info   model.cmf     # arch, tensors, quantization, skills
```

权重经内存映射后就地读取，因此启动是瞬时的，未使用的权重从不进入内存。量化是
按张量来的，且可以混用——`q8`（1 byte/param）· `q8_2f`（int8，同时带每行和每列
两个缩放因子——相同字节数下质量更好）· `q4`（0.5）· `f16` · `vbit`（可变 3–8 bit，
均值约 4.25 ≈ 0.53）——所以你可以在同一个文件里把注意力保持在 q8，而把 FFN 压到 q4。

### 多个专家，共用一个骨干

交付 *N* 个微调版本，通常意味着磁盘和内存里各有 *N* 份完整副本。CMF 只保留**一个
骨干，外加每个专家一个小技能**：技能只存储它实际替换掉的那些张量，推理时运行时会
*取代*骨干的对应张量去读它们——从不需要单独组装出一个模型。存储开销是
`|backbone| + Σ|skills|`，而不是 `N × |model|`，而你没用到的技能**不占任何内存**。

技能不只是交付起来更便宜——在它自己的任务上，它还胜过它所依附的骨干：在留出数据
上，叠加一个技能能把任务困惑度降低 **24.9%**（见[规范 §9](docs/CMF_V2_SPEC.md)）。
骨干越弱的地方，技能的收益越大；在骨干本来就擅长的领域，预期收益要小一些。

```sh
cortiq run model.cmf --prompt "SELECT ..." --skill sql
```

不想手动挑？`cortiq route` 会根据提示词选出技能，`cortiq explain` 会告诉你它为什么
这么选。

托管 *N* 个任务专家：

| | N 个完整微调 | 基座 + N 个外部 LoRA | **CMF** |
|---|---|---|---|
| 磁盘占用 | N × 完整模型 | 基座 + N 个适配器（附属文件） | 一个骨干 + N 个小技能，**一个文件** |
| 分词器 + 聊天模板 | 每份副本各带 / 附属文件 | 基座是 GGUF 则内嵌，否则为附属文件 | **内嵌** |
| 逐张量完整性哈希 | — | — | **有** |
| 未使用的技能占用的内存 | 已加载 | 配合支持适配器分页的服务端（S-LoRA / vLLM）为 0；否则已加载 | **0**，用到时才分页调入，且不需要任何服务框架 |
| 技能随模型文件一同交付 | — | 否（适配器是独立文件） | **是，且在同一条哈希链之下** |

完整的逐格式对比——GGUF、safetensors、ONNX、PyTorch、GGML、TensorRT，并把各自的
取舍讲清楚——见 [docs/COMPARISON.md](docs/COMPARISON.md)。

## 安装

```sh
cargo install cortiq-cli                 # the `cortiq` command-line tool
cargo add cortiq-core                    # or use the format from your own Rust code
```

预编译二进制在[最新发布](https://github.com/infosave2007/cmf/releases/latest)页面
——Linux x86-64、macOS（Apple Silicon 和 Intel）、Windows（x86-64 和 ARM64）；每个
压缩包都附带 `.sha256`。它们**只支持 CPU**；要用 GPU 后端，请用 `--features gpu`
从源码构建。

## 命令

| 命令 | 作用 |
|---|---|
| `cortiq convert --model <hf-repo\|dir>` | Hugging Face checkpoint → `.cmf`（纯 Rust） |
| `cortiq import-gguf <file\|hf-repo>` | GGUF → `.cmf`，覆盖所有常见 ggml 量化类型 |
| `cortiq run model.cmf` | 对话；或用 `--prompt` 跑单次 |
| `cortiq serve model.cmf` | 兼容 OpenAI 的 HTTP 服务器 + 仪表盘 |
| `cortiq info` · `masks` · `verify` | 检视架构、张量、技能；校验完整性 |
| `cortiq bench --ctx 4096` | 给定上下文下的 tok/s 与内存 |
| `cortiq ppl --file f.txt` | teacher-forced 困惑度——质量门禁 |
| `cortiq fcd` | `--o1` 模型的修复训练器（以 KL 锚定，按生成结果把关） |
| `cortiq diff a.cmf b.cmf` | 两个模型版本之间改了什么 |
| `cortiq route` · `explain` | 路由器选了哪个技能，以及为什么 |

`cortiq <command> --help` 里有每个参数的说明。

### 转换

```sh
cortiq convert --model Qwen/Qwen2.5-0.5B-Instruct --quant q8    --output model.cmf
cortiq convert --model ./my-hf-checkpoint         --quant q8_2f --output model.cmf
cortiq import-gguf Qwen/Qwen2.5-0.5B-Instruct-GGUF --output model.cmf --quant q8
```

GGUF 导入覆盖 `Q4_0/1`、`Q5_0/1`、`Q8_0`、`Q2_K`…`Q6_K`、`IQ4_NL/XS` 和 `BF16`。

原生转换器写出的是**骨干**。上文说的那些按技能替换张量和任务掩码，目前仍由
`converter/` 里的 Python 工具链产出；需要激活 Hessian 的 GPTQ 校准 v-bit 变体也是。
仅按权重的 v-bit 路径已经是原生的。

## O(1) 深入解析

可以在转换时把提示记进文件，也可以在加载时再决定——运行时会自动读取 header 里的
提示：

```sh
# at convert time: all softmax layers, the deepest N, or an explicit list
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --o1 all    --output model.cmf
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --o1 deep12 --output model.cmf

# or override at load time, without reconverting
cortiq run   model.cmf --o1 all      # force-convert every softmax layer
cortiq run   model.cmf --o1 off      # back to exact attention
cortiq bench model.cmf --ctx 4096    # memory + tok/s, with and without
CMF_O1=deep6 cortiq serve model.cmf  # env override, same syntax

# tuning (validated defaults: 32 landmarks, window 128, 4 anchor keys)
cortiq run model.cmf --o1 all --o1-m 32 --o1-window 128 --o1-sink 4
```

在混合模型上（例如 qwen3.5：GatedDeltaNet 层中夹着 softmax 孤岛），`--o1 all` 只
转换那些 softmax 层，这就让整个模型的注意力状态在上下文长度上成为常量。

**修复。** `cortiq fcd` 是一趟有界的原生训练——不用 Python，不用 ML 框架——它只
调整被转换层的 norm/FFN 张量，对齐目标是同一个模型跑精确注意力时的输出（以 KL
锚定），并且只有在长上下文生成没有陷入循环时才保留 checkpoint：

```sh
cortiq fcd model.cmf --corpus corpus.txt --gen-check --gen-gate --out model.fcd.cmf
# knobs: --steps 300 --eval-every 25 --kl 0.7 --lr 5e-5 --o1 all|deepN|i,j,k
#        --val-corpus val.txt --gate-threshold 0.35 --gate-slack 0.10
```

## 格式

一个 `.cmf` 是一个固定的 128-byte 信封，后面跟着若干区段；读取方**只**通过这个
信封来定位它们，绝不靠假设顺序：

- **header JSON**——架构、量化默认值、聊天捆绑包、技能注册表、来源信息
- **张量目录**——56-byte 的二进制记录（name、dtype、shape、offset、nbytes、hash64），不碰 JSON 也能读
- **权重 blob**——页对齐，映射后就地读取
- **技能**——位打包的任务掩码和按技能的替换张量
- **分词器**——原封不动的 Hugging Face 文件
- **稀疏索引**——预先算好

此外还支持：多词元预测（MTP）头、MoE FFN 层、仅追加式的技能增长与压实，以及把一个
模型分片到 `N` 个各自独立有效的文件中。

**你不会被锁死。** `python/cmf_reader.py` 是一个完整的读取器，约 300 行，只用
stdlib + numpy，与 Rust 运行时不共享任何代码——它是刻意照着规范写出来的，为的是
证明这个格式活得比这份实现更久：

```python
from cmf_reader import CmfReader
r = CmfReader("model.cmf")
w = r.tensor("model.layers.0.mlp.gate_proj.weight")   # np.ndarray, dequantized
assert r.verify() == []                               # every tensor hash checks
```

就算这个项目明天消失，仅凭规范你的权重依然读得出来。完整的规范性说明见
[docs/CMF_V2_SPEC.md](docs/CMF_V2_SPEC.md)。

## 现状

CMF 目前是 **0.2.x**，还很年轻——2026 年 7 月首次公开发布，作者只有一个人。在 1.0
之前，crate 的 API 仍可能变动。已经定下来的是**格式**：它是 v2，读取方只通过信封
来定位，未知的 header 字段会被忽略（增量式演进），破坏性变更要付出一个 feature bit
或一次 `version` 递增的代价——绝不会悄悄改变含义。今天写出的 `.cmf` 以后依然读得
出来；`cortiq verify` 就是这份契约。每一处改动都记在 [CHANGELOG.md](CHANGELOG.md)。

Bug 和功能请求：[提一个 issue](https://github.com/infosave2007/cmf/issues)。
安全问题：**不要**公开提 issue——见 [SECURITY.md](SECURITY.md)。
转换不了的模型是一份 bug 报告，不是用户的错。

## 从源码构建

```sh
cargo build --release --workspace
cargo build --release --workspace --features gpu   # + wgpu → Vulkan / DX12 / Metal
```

```
crates/
  cortiq-core     format reader: envelope, directory, quant, masks, mmap
  cortiq-engine   portable CPU/GPU inference runtime, tokenizer, chat, skills
  cortiq-server   OpenAI-compatible HTTP serving
  cortiq-cli      the `cortiq` command-line tool
converter/        Python: DTG-MA skills/masks + the GPTQ-calibrated v-bit path
python/           reference reader — stdlib plus numpy, nothing else
docs/             format specification and comparison
```

欢迎贡献——见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

**Apache-2.0**（[LICENSE](LICENSE)）——随你使用、修改，也可用于商业发布。

本软件实现了作者四项美国在审专利申请所主张的方法，清单见 [PATENTS.md](PATENTS.md)。
Apache-2.0 第 3 条向你授予一份永久、全球范围、免版税的专利许可，覆盖本代码按其
分发形式必然涉及的那些权利要求：**运行、fork 和发布本软件都在覆盖范围内**，而
这份授予只有在你就专利起诉本项目时才会失效。

这份授予的范围限于本“作品”（Work），Apache-2.0 §3 向来如此——它本身并不延伸到对
该容器的独立重新实现。如果你想用另一种语言实现 CMF，或把它嵌进你自己的运行时，
请发邮件到 urevich55@gmail.com：面向实现者的授权是可以提供的，而这个格式本就希望
被广泛实现。

## 它从何而来

这些设计思路来自作者另一项独立的物理理论工作——零矢量引力（NVG）框架下的真空质量
分数（VMF）：共享骨干加扰动的模型，以及双字段量化。格式里没有任何东西依赖于那套
理论是否正确；它立足于规范和上面那些数字。完整的映射，并在*已测量*与仍属隐喻者
之间划下硬界线：[CMF 背后的 VMF/NVG 原理](VMF_principles_in_CMF.zh.md)
（[English](VMF_principles_in_CMF.md) · [Русский](VMF_principles_in_CMF.ru.md)）。
物理本身则在[它自己的仓库](https://github.com/infosave2007/vmf)里。
