# 面向任务压缩模型：`cortiq compress`

*语言：[English](COMPRESS.md) · [Русский](COMPRESS.ru.md) · **中文***

一条命令即可把通用模型变成针对你的任务的专用模型：

```bash
cortiq compress model.cmf prompts.jsonl model-task.cmf --ratio 0.75
```

- `model.cmf`：源模型（f16/bf16 或 q8；也可以是 Hugging Face 目录或仓库 id）；
- `prompts.jsonl`：任务提示（每行一个对象，含 `prompt` 或 `messages`；
  `calibration_prompts.jsonl` 的格式可直接使用）；
- `model-task.cmf`：输出结果；
- `--ratio 0.75`：压缩目标，结果为源模型大小的 75%（同一编码格式）。

内部流程：教师模型回答你的提示。默认教师就是模型本身；使用 `--teacher` 时可以
是更大的模型，包括通过 API 访问的模型（见下文）。从回答中可以看出任务使用了哪些
神经元（"照明"，illumination）。未使用的神经元被移除，其贡献转移到保留的神经元
中，然后模型在这些回答上继续训练。如果需要更小的文件，量化是单独的一步：把编码
格式作为第四个参数传入（见下文 S5 阶段），或者之后运行 `cortiq quantize-gptq`
或 `cortiq convert`。

运行过程中可以看到正在进行什么以及还剩多少：阶段编号、该阶段的目标、带速率的
计数器以及"约剩 N 分钟"的估计。中断的运行用同一条命令即可继续，已完成的阶段
不会重新计算。

输出是一个 `.cmf` 文件，可直接用于 `cortiq run` 和 `cortiq serve`。旁边有一份
质量报告（`model-task.cmf.work/S6.eval/report.md`），文件头记录了来源信息
（`provenance.compress`）。

附加选项（默认情况下不需要）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--teacher` | 模型本身 | 谁来回答提示：使用相同分词器的另一个 `.cmf`，或 OpenAI 兼容 API 背后的模型（`openrouter:…`、`openai:…`、`api:…`） |
| `--api-base`、`--api-extra`、`--api-concurrency` | —、—、8 | API 教师：服务器地址、请求附加字段、同时进行的请求数 |
| `--gen-max-tokens` | 2048；API 教师为 8192 | 单个回答最多可占用的 token 数（含推理） |
| `--gen-samples` | 1 | 每个提示的回答数 |
| `--no-think` | | 不经推理直接回答 |
| `--ffn F` / `--size 12B` | | 以 FFN 神经元比例或参数量为目标，替代 `--ratio` |

除了提示，也可以直接提供现成的教师数据（`.tdata`，见下文），例如同系列更大
模型的回答。

**MoE 模型**使用同一条命令压缩。照明统计路由器在任务回答上选择每个专家的频率；
文件中只保留达到所需路由份额的专家（阈值按 `--ratio` 拟合），路由器像原来一样
在剩余专家中选择。示例（LFM2.5-8B-A1B，80 个代码类提示，`--ratio 0.8`）：
f16 16.96 GB → 13.57 GB，各层保留 32 个专家中的 20–30 个，测试 NLL
0.280 → 0.332（只有 2 条记录的测试仅供参考，不是精确评估）。MoE 模型目前还
不做修复。

设计规范（含决策依据和实测数据，俄语）：
[`TZ_TEACHER_DATA_COMPRESS.ru.md`](TZ_TEACHER_DATA_COMPRESS.ru.md)。

## 1. 教师数据

### 生成

```bash
cortiq teacher-data generate TEACHER.cmf PROMPTS.jsonl OUT.tdata \
    [--no-think] [--max-tokens 8192] [--samples 1] [--split 90/5/5] [--seed 42]
```

- `PROMPTS.jsonl`：每行一个对象，含 `prompt` 或 `messages`，可选 `id`、
  `category`（划分按其分层）、`samples`、`max_tokens`、`meta`。
  `calibration_prompts.jsonl` 的格式无需修改即可使用。
- 教师可以是任何带对话模板的 `.cmf`。通常是同一模型的 `q8_2f` 版本（自蒸馏），
  或同系列的更大模型，即使用相同分词器的模型。教师也可以是 API 背后的模型，
  见"通过 API 的教师模型"。
- 推理模型默认会思考。推理与回答的边界严格取 `</think>` token，不做文本解析。
  `--no-think` 切换为直接回答。
- 如果生成陷入重复（最后 512 个 token 中新 5-gram 少于 5%），生成停止，该记录
  标记为 `usable=false`。
- 中断的生成用同一条命令继续：已完成的记录保留，未写完的尾部被截掉。同一目录
  下命令不接受不同的参数。
- 按提示划分：同一提示的所有回答落入同一个划分。任何压缩阶段都不使用测试集，
  只有最终评估使用。

### 通过 API 的教师模型

更大的教师模型无需部署在租用的服务器上：任何提供 OpenAI 兼容
`/chat/completions` 的模型都可以。回答以文本形式返回，并用你要压缩的模型的
分词器转为 token（`--tokenizer`；在 `compress` 中就是源模型本身）。

```bash
# OpenRouter（密钥在 OPENROUTER_API_KEY 中）
cortiq teacher-data generate openrouter:qwen/qwen3.8-27b prompts.jsonl task.tdata \
    --tokenizer model.cmf --sampling auto --no-think --api-concurrency 32
# OpenAI（OPENAI_API_KEY）
cortiq teacher-data generate openai:MODEL prompts.jsonl task.tdata --tokenizer model.cmf
# 自建服务器：vLLM、SGLang、llama.cpp、Ollama、LM Studio、`cortiq serve` 等
# （若服务器需要密钥，放在 CORTIQ_API_KEY 中）
cortiq teacher-data generate api:Qwen/Qwen3-32B prompts.jsonl task.tdata \
    --api-base http://gpu-box:8000/v1 --tokenizer model.cmf
# 同样的事情用一条压缩命令完成
cortiq compress model.cmf prompts.jsonl model-task.cmf --ratio 0.8 \
    --teacher openrouter:qwen/qwen3.8-27b --no-think --api-concurrency 32
```

- 密钥只从环境变量读取，不会写入任何地方。`.tdata` 的清单记录模型、地址和解码
  设置。
- 回答随到随写入 `OUT.tdata.answers.jsonl`。中断的运行用同一条命令继续：已完成
  的回答不会重复请求或重复付费。进度行显示速率、已花费的金额（如果服务器提供）
  以及剩余时间估计。
- 推理开关：OpenRouter 使用 `reasoning` 字段，vLLM/SGLang/llama.cpp 使用
  `chat_template_kwargs.enable_thinking`。推理文本取自 `reasoning`/
  `reasoning_content` 或回答中的 `<think>…</think>`，并放在 `</think>` 之前，
  与学生模型模板的续写方式一致。在推理中途因上限被截断的回答作为截断记录保存，
  不带结束标签，也不带结束 token。
- 不同服务器接受的可选字段不同（`top_k`、`seed`、`max_tokens` 或
  `max_completion_tokens`、推理开关）。服务器拒绝的字段会被丢弃一次并给出提示，
  工作继续进行。错误的密钥或模型会立即停止运行，而不会产生上百条相同的错误。
- OpenRouter 默认选择以 8 或 16 bit 提供该模型的最便宜提供商。同一模型的价格
  相差 2.5 倍，部分提供商以 fp4 提供。每个回答的提供商记录在 `meta.provider` 中。
  可以用 `--api-extra` 自行指定，例如 `'{"provider":{"sort":"throughput"}}'`。
- `--api-extra '{…}'` 为每个请求添加字段；值为 `null` 的字段不会发送。

实测（经 OpenRouter 调用 Qwen3.8-27B，代码类提示）：

| | 回答数 | token 数 | 价格 | 时间 |
|---|---|---|---|---|
| 开启推理，上限 8192，8 路并行 | 12 | 8.76 万（9 份达到上限） | 0.19 美元 | 4 分钟 |
| `--no-think`，12 路并行，任意提供商 | 12 | 2.20 万（全部完整） | 0.048 美元 | 55 秒 |
| `--no-think`，最便宜的 8/16 bit 提供商 | 12 | 2.08 万（全部完整） | 0.032 美元 | 2 分钟 |
| `--no-think`，32 路并行，任意提供商 | 160 | 41.9 万 | 0.91 美元 | 2 分钟 |
| `--no-think`，483 个提示 × 2，新增 806 份（160 份来自缓存） | 966（52 份达到 8192） | 135 万 | 2.38 美元 | 15 分钟 |

**分词器必须与被压缩的模型完全一致。** 同系列的模型也有差异：Qwen3.8-27B 的
`tokenizer.json` 比 Qwen3.5-0.8B 多 7 个特殊 token（音频/TTS，id
248070–248076），并有自己的 `generation_config.json`，以 `<|im_end|>` 作为轮次
结束。S0 会按词表哈希拒绝用较小模型分词的数据，回答也会以错误的 token 结束。
如果机器上没有该模型本身，分词器载体即可：同系列的一个小模型，其目录中放入目标
模型的 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 和
`generation_config.json`。用 `cortiq convert` 把它转换为 CMF，作为 `--tokenizer`
传入。缓存中的回答会被重新导入，无需新的请求。

开启推理的 27B 在四分之三的任务上思考超过 8k token。如果压缩后的模型将在不推理
的情况下运行，`--no-think` 更便宜，回答也更完整。如果大量回答达到上限，命令会在
结束时建议提高上限或关闭推理。

### 导入现有的回答缓存

```bash
cortiq teacher-data import train_combined.json OUT.tdata --tokenizer MODEL.cmf
```

接受 `[prompt, rendered ChatML]` 数组（DTG-MA 笔记本的缓存），或含
`prompt`/`messages` 和 `response` 的 JSONL。token 由 `--tokenizer`（即你要压缩
的模型）计算。同一提示的重复项成为多个样本。注意：这类回答由另一个模型写出，
因此只能作为文本目标，不能作为本模型的分布。

### 检查

```bash
cortiq teacher-data inspect OUT.tdata --verify
```

## 2. 压缩详解

```bash
cortiq compress SOURCE PROMPTS|DATA OUTPUT [QUANT] --ratio R [其他选项]
```

| 参数 | 含义 |
|---|---|
| `SOURCE` | `.cmf`（f16/bf16；q8 会给出警告）、本地 HF 目录或仓库 id |
| `PROMPTS` / `DATA` | 提示的 `.jsonl`（由模型自己回答）或现成的 `.tdata` |
| `OUTPUT` | 最终的 `.cmf`，编码格式与源模型相同 |
| `QUANT` | 可选：输出的编码格式（`q4tp`、`q2tp`、`q8_2f` 等），在 S5 中量化 |
| `--ratio` | 占源模型大小的比例（同一编码格式下的参数量） |
| `--ffn` / `--size` | `--ratio` 的替代：FFN 神经元比例或目标参数量 |

### 阶段

每个阶段作为独立进程运行，并以其输入为键缓存在 `OUTPUT.work/` 中。再次运行同一
命令会从第一个未完成的阶段继续。修改某个参数只会重新计算对应阶段及其后续阶段。

| 阶段 | 作用 |
|---|---|
| S0 prepare | 源模型 → 工作用 f16 副本 + 用于校准的 q8_2f 副本；检查数据的分词器 |
| S1 illuminate | 照明：源模型在教师回答上运行（训练集和验证集分开）；每层 `down_proj` 输入的 Gram 矩阵、神经元能量、任意宽度下的重拟合（refit）误差曲线 |
| S2 plan | 各层宽度：相同宽度（`--allocate uniform`，默认）、按重拟合误差曲线（`curve`），或按曲线并以层输出能量加权（`abs`）；首尾层可以保持全宽（`--protect-first N --protect-last M`，默认 0：保护会让其他层更窄，总体有害——2B 上折叠为 0.330，不保护为 0.320）；宽度为 128 的倍数。在 0.8B 上统一分配的 NLL 为 0.621，`curve` 为 0.629 |
| S3 fold | 按重要性 `‖W_down[:,j]‖²·⟨a_j²⟩` 选择保留的神经元；被丢弃的神经元回归到保留的神经元上，`W_down' = W_S + W_P·Πᵀ`，`Π = (Σ_SS+εI)⁻¹Σ_SP`；岭参数及是否重拟合都按验证集选择（`--refit auto`）。训练器可用时在 GPU 上计算（见下文） |
| S4 heal | 以未压缩的源模型为参照，在教师回答上做知识蒸馏修复，损失只计算生成的 token，按验证集选最佳检查点。在 NVIDIA（Vulkan，矩阵单元）和 Apple Silicon（Metal，8×8 simdgroup 矩阵）上用 `nvg-train` 内核在 GPU 上运行：尾部层的权重整体训练（f32 + Adam，f16 权重平面每步更新），对源模型 top-64 计算 KL；下面的层冻结，反向传播不会到达。否则在 CPU 上：`0.3·CE + 0.7·KL`，窗口位于记录内部，并有防止循环的生成门槛 |
| S5 quant | 仅在给出 `QUANT` 时执行（默认输出保持源模型的编码格式）。`q4tp`/`q2tp`：在同一数据上校准的 GPTQ（窗口位于记录内部），全注意力的 k/v 用 q8_2f（`--escort kv`）；其他编码格式使用转换器的编码器 |
| S6 eval | 在测试集上：教师回答的 NLL、测试提示的贪心续写（循环、相对教师的长度、自行结束的比例）、速度、大小；质量门槛；`report.md` |

### 预设（`--recipe`）

| | `fast` | `balanced`（默认） | `full` | `quant-only` |
|---|---|---|---|---|
| 照明 + 折叠 | ✓ | ✓ | ✓ | — |
| 修复 | — | FCD，2 个 epoch | FCD，3 个 epoch | — |
| GPTQ（仅在给出 `QUANT` 时），校准 token 数 | 3.2 万 | 最多 6.4 万 | 最多 6.4 万 | 最多 6.4 万 |

`quant-only` 不改变结构：它就是在任务数据上做 GPTQ。可以作为"结构压缩相对
单纯量化带来多少收益"的参照点。

### 常用参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--plan-only` | | 显示计划后停止 |
| `--until S3`、`--from S4` | | 在某阶段后停止 / 从某阶段开始重新计算 |
| `--heal-layers all\|deepN\|auto` | auto | 修复哪些层（auto：内存足够时修复全部，否则修复能放下的最后若干层） |
| `--heal-epochs`、`--heal-lr`、`--kd` | 见右列，0.7 | epoch 数：CPU 2（`full` 为 3），GPU 6（`full` 为 12）；学习率：CPU 1e-5；GPU 在尾部不超过 30% 层时为 1e-5，训练全部层时降到 3e-6（0.8B 全部层：1e-5 → 0.331，5e-6 → 0.310，3e-6 → 0.307）；`--kd` 仅用于 CPU 路径 |
| `--heal-loss ce\|kd\|mix=A` | mix=0.5 | GPU 修复损失：教师文本（CE）、源模型分布（KL），或 A·CE + (1−A)·KL（0.8B、75% FFN：kd 0.303，ce 0.292，mix=0.5 0.281） |
| `--heal-patience`、`--heal-val-windows` | 6、32 | GPU 修复在连续这么多次留出检查没有改进后停止；每次检查读取这么多个验证窗口 |
| `--heal-device auto\|gpu\|cpu` | auto | 在哪里修复：auto 表示显卡有协作矩阵（NVIDIA Vulkan 或 Apple Metal）时用 GPU，否则用 CPU |
| `--heal-params ffn\|all`、`--heal-window`、`--heal-topk` | ffn、2048、64 | GPU 修复：训练层中的哪些权重（只训练 MLP 在 0.8B 上为 0.307，训练全部权重为 0.310，内存只需三分之一）、窗口长度、每个 token 保存多少个教师概率 |
| `--loss-on answer\|generated\|all` | generated | 在哪些 token 上计算损失 |
| `--keep-grams auto\|on\|off` | auto | 把 S1 的 Gram 矩阵保存到磁盘供折叠使用（f32，每层 8·n² 字节——27B 为 155 GB）；auto 表示占用不到一半可用空间时保存。保存后，多个目标（不同 `--work-dir` 中的 `--size 20B`、`17B`、`14B`，共享 S0/S1）可以基于同一次照明折叠；否则 S3 会重新照明 |
| `--ridge` | 0.03,0.1,0.3 | 重拟合的相对岭参数网格（在 1130 个层上，0.1 胜出 86%，0.003 从未胜出） |
| `--calib-tokens`、`--escort` | 65536、kv | GPTQ 校准量，以及保持为 q8_2f 的张量 |
| `--mtp-heal auto\|on\|off` | auto | 当源模型带有草稿且 GPU 训练器可用时，在最终模型上重新训练 MTP 草稿（`cortiq mtp-heal`，3 步链） |
| `--general-eval FILE` | | 在报告中给出通用文本上的 ppl（检查遗忘） |
| `--compare MODEL.cmf` | | 在同一批测试记录上评估另一个模型（另一种压缩、单纯量化、另一位教师）；可重复 |
| `--select importance\|dtgma` | importance | 如何选择 FFN 神经元：按照明（一遍）或按训练得到的 DTG-MA 掩码（专利 2，见下文） |
| `--write-on-fail` | | 即使未通过质量门槛也写出结果（`*.rejected.cmf`） |

### 用 DTG-MA 掩码选择神经元（`--select dtgma`）

不按照明排序，而是在每个 FFN 神经元上训练掩码 σ(m)：在逐渐增大的 L1 压力下，
用任务的语言模型损失贯穿整个模型训练。每 25 步在验证集上检查硬掩码（σ > 0.5），
取第一个剪枝比例不低于目标的检查点。根据训练得到的 logits，用一个统一阈值确定
每层宽度，并选出具体的神经元。之后照常进行带重拟合的折叠和修复。

在 NVIDIA 上，掩码位于 GPU 训练器中：SwiGLU 与 `down_proj` 之间的门控，每步
0.37 秒，0.8B 的完整掩码约 3 分钟（主机上的 `skillbake` 为每步 16.7 秒，共
70 分钟）。压力每次评估乘以 1.5，直到达到目标；线性增长在 0.8B 上会停滞在 15%。

| Qwen3.5-0.8B | 折叠：照明 / DTG-MA | 修复后：照明 / DTG-MA |
|---|---|---|
| FFN 75% | 0.364 / **0.339** | 0.281 / 0.282 |
| `--ratio 0.9` | 0.394 / **0.372** | 0.305 / 0.310 |
| Qwen3.5-2B，FFN 75% | 0.320 / 0.320 | **0.255** / 0.261 |

在 0.8B 上掩码得到更好的折叠；修复后结果相同，或照明略好（2B）。因此默认仍
使用照明：它更便宜。

### GPU：NVIDIA 与 Apple Silicon

照明（S1）、重拟合曲线（S1）、DTG-MA 掩码（S2）、折叠（S3）、修复（S4）和测试
NLL（S6）都在 GPU 训练器上运行。在 NVIDIA 上使用 Vulkan 和 16×16 张量核心；在
Mac 上通过 wgpu 使用 Metal 和 Apple 的 8×8 simdgroup 矩阵。Mac 上的运行时仍使用
原生 Metal，训练器有自己的上下文。`CMF_TRAIN_GPU=0` 可在 Mac 上关闭训练器。

Qwen3.5-0.8B 的一个修复步（2 个窗口 × 2048 token，所有层的 MLP）在 M4 上的耗时：

| 启用的优化 | 秒/步 |
|---|---|
| 注意力和 GDN 的 f32 备用内核 | 47 |
| + 基于 8×8 simdgroup 矩阵的注意力 | 27.6 |
| + 基于 SIMD 组的快速 GDN 内核 | 17.4 |
| + 64×64×32 的 GEMM 分块（3.0 TFLOPS，原为 1.7） | 12.2 |

作为对比，RTX 5090 每步 0.42 秒。在 MacBook 上，机器发热后每步增至约 16 秒。
GEMM 上限约 3 TFLOPS，接近 M4 GPU 的实际极限，因此在其上完整修复 0.8B 需要数小时。
短时运行可使用 `--heal-epochs 1`。

**在 GPU 上折叠。** 在均衡化矩阵上做 f32 Cholesky 分解（单位对角线，与设备上的
GPTQ 相同：这样 f32 看到的是相关性而不是神经元能量，而 Qwen 的离群值约为中位数的
10⁴ 倍），求解以分块 GEMM 完成。修正项 `Aᵀ = (Σ_SS+εI)⁻¹·Σ_SP·W_Pᵀ` 针对 h 个
右端项求解，而不像 `Π` 那样针对 p 个。所有误差都以与 `Σ_SP·W_Pᵀ` 和 `Σ_SS·Aᵀ`
的迹来计算。结果与主机上的 f64 版本一致：在 Qwen3.5-0.8B、保留 75% FFN 时，
全部 24 层的验证/训练误差和岭参数都相同。`CMF_FOLD_ON=host` 可切回主机路径。

| 单层 | 主机（f64） | GPU，M4 |
|---|---|---|
| n 6144 → 4608，h 2048（2B 规模） | 29 秒 | 2.2 秒 |
| n 17408 → 4352，h 5120（27B，14B 目标点） | 约 11 分钟（按规模推算） | 12 秒 |
| n 17408 → 10496（27B，20B 目标点） | — | 33 秒 |
| 重拟合曲线，n 17408 | — | 14 秒 |

0.8B 上整个 S3：21 秒，主机上为 121 秒。

**大模型的照明。** 与权重放不到一起的 Gram 矩阵分多遍累积。每一遍从上一遍停止处
的残差流继续，残差流按训练集和验证集分别保存在主机上，因此每一层只计算一次。
累加器的空间取自 f16 权重和工作缓冲区剩余的部分。对于 96 GB 上的 27B，每遍约
25 层。

### 质量门槛（S6）

在同一批测试提示上把结果与源模型比较：循环次数不超过源模型加一；回答长度中位数
不短于教师的 0.7；自行结束的回答数不少于源模型减去 10% 的提示数。未通过门槛时
不写出最终文件（退出码 5），报告总会写出。

## 示例（RTX 5090，GPU 修复）

数据：`calibration_prompts.jsonl` 中的 483 个提示，教师为模型本身（MoE 为 80 个
代码类提示）。划分 80/10/10。NLL 在测试集上计算，即压缩过程从未见过的教师回答。
生成使用 16 个测试提示。输出保持源模型的编码格式（f16）。

| 模型 | `--ratio` | 参数量 | 测试 NLL：源模型 → 折叠 → 修复 | 门槛 |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.9 | 0.773B → 0.697B | 0.189 → 0.394 → 0.305 | 通过 |
| Qwen3.5-0.8B | 0.8 | 0.773B → 0.622B | 0.189 → 0.698 → 0.524 | 未通过：3/16 循环 |
| Qwen3.5-2B | 0.8 | 1.943B → 1.565B | 0.185 → 0.464 → 0.354 | 通过 |
| LFM2.5-8B-A1B（MoE） | 0.9 / 0.8 / 0.65 | 8.47B → 7.62 / 6.77 / 5.49B | 0.280 → 0.289 / 0.332 / 0.399（未修复） | — |

- 0.8B 有三分之一的参数是嵌入层，不参与压缩。因此 `--ratio 0.8` 会去掉 30% 的
  FFN 宽度，这已经过多。
- 在 5090 上运行一次 0.8B 约 6 分钟（不含数据生成）：照明 36 秒，折叠约 1 分钟，
  修复约 1.5 分钟，评估约 2 分钟。
- 作为对比，在 2B 上单纯的 q4tp 量化得到 1.03 GB、NLL 0.225。面向任务的压缩
  不能替代量化，而是叠加在量化之上。
- 大模型示例：在一块 RTX PRO 6000 上把 Qwen3.8-27B 压缩到 14B，模型、提示和
  教师数据集均已发布（见[延伸阅读](#4-延伸阅读)）。

## 3. 在其他命令中使用教师数据

`.tdata` 可以作为校准语料：`PATH.tdata[:train|val|test|all][=SHARE]`。

```bash
cortiq quantize-gptq model-f16.cmf --codec q4tp --calib-model model-q8_2f.cmf \
    --calib task.tdata:train=0.8 --calib wiki.txt=0.2 --tokens 65536 --output model-q4tp.cmf
```

窗口在记录内部构建，从不把两个提示拼接在一起。

## 4. 延伸阅读

- [面向任务压缩语言模型](research/TASK_COMPRESSION_ARTICLE.zh.md)：方法以及在
  Qwen3.5-0.8B 和 2B 上的测量。
- [在一块 RTX PRO 6000 上把 Qwen3.8-27B 压缩到 14B：逐步记录](research/QWEN38_27B_TASK_COMPRESSION.zh.md)。
- 得到的模型及其提示和教师数据集：
  [infosave/Qwen3.8-27B-Code-14B-Experimental-cmf](https://huggingface.co/infosave/Qwen3.8-27B-Code-14B-Experimental-cmf)。
