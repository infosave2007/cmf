# 面向任务压缩语言模型：照明、渗透折叠、修复、DTG-MA 掩码以及 NVG ↔ DTG-MA 的顺序

*语言：[English](TASK_COMPRESSION_ARTICLE.md) · [Русский](TASK_COMPRESSION_ARTICLE.ru.md) · **中文***

*Oleg Urevich Kirichenko。实现与测量：cortiq 引擎与 CMF 格式（`compress` 分支），2026 年 10 月 27–28 日。*

## 摘要

通用语言模型的大部分权重都花在具体任务用不到的地方。我们构建了一条命令
`cortiq compress SOURCE.cmf PROMPTS.jsonl OUTPUT.cmf --ratio R`，把通用模型变成
针对用户提示列表的专用模型。流水线如下：

1. 教师模型回答提示：可以是模型本身、更大的模型，或通过 OpenAI 兼容 API 访问的模型。
2. 从回答中可以看出任务使用了哪些 FFN 神经元（"照明"，illumination）。
3. 多余的神经元被移除，其贡献通过闭式解转移到保留的神经元中：渗透折叠
   （osmotic folding，专利 15，USPTO 申请 19/731,402）。
4. 模型在教师回答上继续训练（"修复"，healing；专利 15 和专利 16）。
5. 用留出（held-out）回答评估结果。

各层宽度也可以由训练得到的 DTG-MA 掩码选择（专利 2）。对 MoE 模型，会移除任务
不会路由到的专家。繁重的阶段运行在 GPU 训练器上：NVIDIA 通过 Vulkan 和张量核心，
Apple Silicon 通过 Metal 和 simdgroup 矩阵。

在 Qwen3.5-0.8B 和 2B 上得到以下结果：

- **渗透折叠是结构压缩的主要杠杆。** 把被移除神经元的贡献转移到保留的神经元
  中，与直接截断相比，FFN 输出误差减半。
- **神经元的选择对结果几乎没有影响。** 训练得到的 DTG-MA 掩码比照明得到更好的
  折叠，但修复之后优势消失：在这些模型上，修复抹平了选择上的差异。
- **量化模型上的掩码起到修复量化的作用。** 在用 NVG 编码到每权重 2 bit 的模型上，
  DTG-MA 掩码会关闭受损最严重的神经元。与教师的 KL 从 2.17 降到 1.23，而权重一步
  都没有更新。同等大小下，"先 NVG、后 DTG-MA"优于相反顺序：2 bit 时测试 NLL 为
  1.618 对 1.931，3 bit 时为 1.277 对 1.340。
- **小型稠密模型应当通过量化达到压缩比，剪枝只在量化之上叠加。**

## 1. 问题

给定一个 CMF 格式的模型和一个任务提示文件，生成指定大小的文件（同一编码格式下
为源模型的 `R` 倍），使其在该任务上的表现与源模型一致。要求如下：

- 一条命令，无需手动阶段；
- 每个阶段都显示进度和剩余时间估计；
- 中断的运行可以从停止处继续；
- 量化是用户单独的一步：输出保持源模型的编码格式；
- 不减少深度（层数），只压缩 FFN 宽度和 MoE 专家；
- CMF 格式自行记录每一层的宽度，因此被移除的神经元会从文件中物理消失，无需碎片
  整理或宽度对齐。

## 2. 本研究所依据的作者专利与工作

| 来源 | 本工作采用的内容 |
|---|---|
| **专利 2**，USPTO 申请 **19/452,464**（2026 年 1 月 19 日提交）：*System and Method for Dynamic Task-Guided Neural Network Compression with Catastrophic Forgetting Prevention*（DTG-MA、DFSC） | 在冻结权重的前提下对 MLP 神经元使用可训练掩码 σ(m)；朝向稀疏的 L1 压力；硬阈值 `1[σ(m) > τ]`；把掩码编译为物理截断的模型。本文把掩码嵌入 GPU 训练器（§3.5），并应用于 NVG 模型（§5.6） |
| **专利 15**，USPTO 申请 **19/731,402**（2026 年 7 月 6 日提交，为 19/452,440 和 19/452,464 的部分延续申请）：*Unified Execution Architecture for Serving a Plurality of Specialized Language Models from a Single Shared Backbone via Dynamically Overlaid Compressed Delta Representations Without Materializing Separate Models* | **渗透折叠**（§[0025]–[0026]）：被移除的 FFN 神经元通过协方差加权重建 `W_down' = W_down[:,kept] + W_down[:,pruned]·Σ_SS⁻¹Σ_PSᵀ` 汇入保留的神经元，而不是直接截断；**修复**（§[0022]）：对折叠后子网络的蒸馏修复；用 DTG-MA 掩码创建技能（§[0021]）。本文的折叠见 §3.3，岭参数按验证集选择 |
| **专利 16**，USPTO 申请 **19/738,763**（2026 年 7 月 13 日提交）：*Training-Free Conversion of Softmax Attention Layers … into Constant-Memory Streaming Attention …, and Teacher-Anchored Surroundings-Only Quality Restoration* | 质量恢复方案：归一化层和 FFN 张量在交叉熵与锚定教师的散度的混合损失上训练，其余冻结。本文的修复（§3.4）结构相同：训练 MLP，损失为对源模型的 `α·CE + (1−α)·KL` |
| **专利 12**（AWNP）：*System and Method for Activation-Weighted Structured Compression of Neural Networks with Architecture-Preserving Layer Selection* | 在任务校准数据上按激活加权的通道重要性；以三元组移除神经元（`gate`/`up` 的行、`down` 的列）；不带蒸馏的修复（§4.7），即 `--heal-loss ce` 模式。按作者决定，**未使用**层组选择（深度压缩） |
| **Frozen Core Decomposition（FCD）**，Zenodo，DOI 10.5281/zenodo.18006952 | 把修复视为在冻结核心之上训练尾部层。GPU 修复从 `train_from` 起训练各层，冻结前缀的输出只计算一次并保存在主机上 |
| **DTG-MA**，Zenodo，DOI 10.5281/zenodo.18088857 | A 阶段的做法：掩码初始化、压力调度、按留出数据上的硬掩码选择检查点 |
| **NVG（Null-Vector Gravity）**：每权重低于 1 bit 的编码器——符号拓扑 Ĉ、闭式度量 Ť、符号的离散精调；`blr` 容器 | 用于方法顺序实验（§5.6）中的编码以及 `nvg-train` 的度量训练 |
| **专利 1**，USPTO 申请 **19/452,440**（2026 年 1 月 19 日提交）：*System and Method for Unsupervised Multi-Task Routing via Signal Reconstruction Resonance* | NVG 编码器在选择候选符号时使用的"重建误差取最小"规则（通过 NVG 间接使用） |

## 3. 方法

### 3.1 教师数据

教师模型回答每个提示。默认是模型本身，也可以是使用相同分词器的另一个 CMF，或
任何 OpenAI 兼容 API 背后的模型（`openrouter:`、`openai:`、`api:` + `--api-base`）。
API 回答以文本形式返回，并用被压缩模型的分词器转为 token。推理内容（`reasoning` /
`reasoning_content` / `<think>`）放在 `</think>` 之前，与学生模型模板的续写方式
一致。因 token 上限被截断的回答作为不带结束 token 的截断记录保存。记录按提示划分
为训练/验证/测试集（80/10/10，按类别分层）。任何压缩阶段都看不到测试集。

### 3.2 照明

源模型在教师回答上运行。对每一层 `down_proj` 的输入累积 Gram 矩阵
`Σ = Σ_t a_t a_tᵀ`（训练集和验证集分开）。神经元 j 的重要性为

`I_j = ‖W_down[:, j]‖² · ⟨a_j²⟩`，

即该神经元在任务数据上对层输出的贡献。这就是专利 12 中按激活加权的通道重要性，
只是不做 SVD：直接评估 `down_proj` 的输出本身。

### 3.3 渗透折叠（专利 15）

某层保留神经元集合 S，移除集合 P。被移除神经元的贡献不是直接截断，而是通过
协方差加权重建汇入保留的神经元，即申请 19/731,402（§[0025]–[0026]）中的渗透折叠：

`W_down' = W_S + W_P · Πᵀ`，`Π = (Σ_SS + ε·I)⁻¹ Σ_SP`，

其中 Π 是用保留的神经元最佳预测每个被移除神经元激活的回归系数，ε 是相对岭参数。
在本实现中，岭参数以及"折叠还是直接截断"的选择都由验证集上的输出误差决定。
候选方案的误差通过被移除列的 Gram 矩阵 `G = W_Pᵀ W_P` 计算：

`‖W_P − A‖²_Σ = ⟨G, Σ_PP − 2·Πᵀ Σ_SP + Πᵀ Σ_SS Π⟩`。

这种形式需要 k²p + 2kp² 次运算，而显式构造 `A = W_P Πᵀ` 需要
h(k² + kp + p²) + hkp 次。

### 3.4 修复（专利 15、16 和 12）

折叠后的模型继续训练。这是申请 19/731,402（§[0022]）意义上的修复：在任务数据上
修复折叠带来的扰动。训练的是尾部层的 MLP 权重：权重的 f32 副本配合 Adam，f16
权重平面每一步更新。下面的层冻结（FCD），反向传播不会到达它们。损失把教师文本
上的 `α·CE` 与对未压缩源模型 top-64 分布的 `(1−α)·KL` 混合。这就是申请
19/738,763 的恢复方案：归一化层和 FFN 张量在交叉熵与锚定教师的散度的混合损失上
训练，其余冻结。当 α = 1 时，它就是不带蒸馏的修复（专利 12，§4.7）。最佳检查点
按验证集选择。

### 3.5 GPU 训练器中的 DTG-MA 掩码

每层的 SwiGLU 与 `down_proj` 之间有一个门控：`h ↦ h ⊙ s`，其中 `s = σ(m)`。门控
的前向和反向传播由两个融合内核完成。梯度
`∂L/∂s_j = Σ_t ∂L/∂(h s)_{t,j} · h_{t,j}` 在 GPU 上以按列归约计算。对 logits m
的 Adam（0.8B 上为 86,016 个数）在主机上运行。L1 压力 `λ·σ(m)` 在每次评估时增大，
直到达到目标稀疏度。每 25 步在验证集上检查硬掩码 `σ(m) > 0.5`，取第一个达到目标
的检查点。根据训练得到的 logits，用一个统一阈值确定每层宽度（128 的倍数），logits
的排序决定保留哪些神经元。之后照常进行折叠和修复（§3.3–3.4）。

梯度在真实的 0.8B 上做了检验。门控全开时，网络与无门控网络逐位一致。门控梯度
与有限差分相符：步长 0.08 时，最后一层误差在 0.3% 以内，第 0 层在 3.5% 以内。
第 0 层步长 0.02 时，由于前向中的 f16 舍入，偏差最高达 33%，与已验证过的第 0 层
归一化梯度表现相同。

### 3.6 MoE 模型

照明统计路由器在任务回答上选择每个专家的频率。文件中保留达到所需路由份额的
专家；阈值通过二分法针对 `R` 求得。路由器被裁剪到保留的专家，并像原来一样在
其中进行选择。

## 4. 基础设施

- **S0–S6 各阶段**是独立进程，缓存以输入为键。重复运行会从第一个未完成的阶段继续。
- **NVIDIA 上的 GPU 训练器**（Vulkan，16×16 协作矩阵），Qwen3.5-0.8B，RTX 5090：
  - 修复 99 秒，CPU 上为 6308 秒；
  - 照明 45 秒，CPU 上为 320 秒；
  - 测试 NLL 10 秒，CPU 上为 230 秒，与运行时结果相差不超过 2·10⁻⁴。
- **Apple Silicon 上的 GPU 训练器**（通过 wgpu 使用 Metal，8×8 simdgroup 矩阵）。
  GEMM 和注意力生成器推广到 8×8 片段。GDN 和符号训练内核运行在 32 宽的 SIMD 组上。
  0.8B 的一个修复步（2 个窗口 × 2048 token，所有层的 MLP）在 M4 上的耗时：

  | 启用的优化 | 秒/步 |
  |---|---|
  | 注意力和 GDN 的 f32 备用内核 | 47 |
  | + 基于 8×8 simdgroup 矩阵的注意力 | 27.6 |
  | + 基于 SIMD 组的 GDN 内核 | 17.4 |
  | + 64×64×32 的 GEMM 分块（3.0 TFLOPS，原为 1.7） | 12.2 |

  同样的 `compress` 在 Mac 上得到源模型 0.1891、折叠 0.3642；在 5090 上为 0.1891
  和 0.3643。
- **通过 API 的教师模型。** 经 OpenRouter 调用 Qwen3.8-27B，代码类提示：
  - 160 份不带推理的回答，32 个请求并行：2 分钟内 41.9 万 token，0.91 美元；
  - 开启推理且上限 8192 token 时，12 份回答中有 9 份在推理中途被截断；
  - 同一模型不同提供商的价格相差 2.5 倍，部分提供商以 fp4 提供该模型。选择以
    8/16 bit 权重提供服务的最便宜提供商，可使每 token 价格降低 30%。

## 5. 实验

模型为 Qwen3.5-0.8B 和 Qwen3.5-2B。任务为 483 个代码类提示，教师为模型本身。
过滤后 0.8B 保留 326 条记录，测试集 34 条（39,315 个回答 token）。指标为教师
测试回答上的 NLL。质量门槛为：循环、回答长度相对教师的比例，以及 16 个测试提示
中自行结束的回答比例。

### 5.1 折叠：贡献转移与宽度分配

| 比较项 | 结果 |
|---|---|
| 验证集上 FFN 平均输出误差，0.8B，保留 75% FFN：截断 → 渗透折叠 | 0.1053 → 0.0600 |
| 同上，2B | 0.1028 → 0.0625 |
| 0.8B 未修复（早期数据集，源模型 0.504）：直接截断 / 渗透折叠 | 0.753 / 0.629 |
| 宽度分配：统一 / 按误差曲线 / 按层能量加权 | 0.621 / 0.629 / 0.635 |
| 2B 折叠：保护首尾层（全宽） / 不保护 | 0.330 / 0.320 |
| 1130 个层上选出的岭参数 | 0.1 占 86%，0.03 占 10%，0.01 占 3%，0.003 从未入选 |

### 5.2 修复

0.8B，保留 75% FFN，折叠为 0.364。

| 修复损失 | 测试 NLL |
|---|---|
| 仅对源模型 top-64 的 KL | 0.303 |
| 仅 CE（无蒸馏） | 0.292（验证集最佳在第 150 步，之后过拟合） |
| **混合 0.5·CE + 0.5·KL** | **0.281** |
| 带解冻调度的混合（genesis） | 0.285 |

### 5.3 一条命令：`--ratio`

| 模型 | `--ratio` | 参数量 | 测试 NLL：源模型 → 折叠 → 修复 | 门槛 |
|---|---|---|---|---|
| 0.8B | 0.9 | 0.773B → 0.697B | 0.189 → 0.394 → 0.305 | 通过 |
| 0.8B | 0.8 | 0.773B → 0.622B | 0.189 → 0.698 → 0.524 | 未通过（循环） |
| 2B | 0.8 | 1.943B → 1.565B | 0.185 → 0.464 → 0.354 | 通过 |

0.8B 有三分之一的参数是嵌入层，不参与压缩。`--ratio 0.8` 会去掉 30% 的 FFN 宽度，
这已经是极限。

### 5.4 2B 上的量化与剪枝对比

| 文件 | 大小 | 测试 NLL |
|---|---|---|
| 源模型 | 1.97 GB（q8） | 0.185 |
| 仅 q4tp | 1.03 GB | 0.225 |
| 保留 75% FFN + 修复 + q4tp | 0.92 GB | 0.288 |

文件小 11%，代价是 NLL 增加 0.063。对小型稠密模型，剪枝代价很高：压缩比应当
来自量化，剪枝在其上叠加。

### 5.5 神经元选择：照明与 DTG-MA 掩码对比

在 GPU 上训练掩码每步 0.37 秒，主机上的 `skillbake` 实现为 16.7 秒（快 45 倍）。
0.8B 的完整掩码约 3 分钟训练完，原来需要 70 分钟。L1 压力线性增长时，剪枝停滞
在 15%：剩余门控上的任务梯度增长快于惩罚。每次评估压力乘以 1.5 时，325 步即可
达到 25% 的目标。

| 模型、目标 | 折叠：照明 / DTG-MA | 修复后：照明 / DTG-MA |
|---|---|---|
| 0.8B，保留 75% FFN | 0.364 / **0.339** | **0.281** / 0.282 |
| 0.8B，`--ratio 0.9` | 0.394 / **0.372** | **0.305** / 0.310 |
| 2B，保留 75% FFN | 0.320 / 0.320 | **0.255** / 0.261 |

掩码选择神经元优于照明：0.8B 上折叠效果更好。但修复会拉平结果。照明成本更低，
仍为默认方式。

### 5.6 方法顺序：NVG 与 DTG-MA

0.8B，保留 75% FFN，两种方案大小相同。

- **A：DTG-MA → NVG。** 先用掩码压缩 f16 模型，再用 NVG 编码，然后用
  `nvg-train` 继续训练度量。
- **B：NVG → DTG-MA。** 先编码整个模型，再在 NVG 模型上用同样的教师 KL 训练
  DTG-MA 掩码；被关闭的神经元直接从二值低秩张量中切除（`gate`/`up` 的行、`down`
  的列，无需重新编码）；然后像 A 一样继续训练度量。

为使大小一致，B 分支的 FFN 以除以 1.074 的比特率编码，切除之后张量的秩与 A 分支
相同。两个分支共享教师、数据（任务文本，2 个 epoch × 26.2 万 token）和 top-k 缓存。

| 比特率、大小 | 分支 | 测试 NLL | 循环 | 留出数据上的 KL |
|---|---|---|---|---|
| 2 bit/权重，0.398 GB | A，未继续训练 | 2.587 | 16/16 | 2.217 |
| | A，经 `nvg-train` | 1.931 | 16/16 | 1.577 |
| | B，切除后立即评估 | 1.681 | 16/16 | 1.225 |
| | **B，经 `nvg-train`** | **1.618** | 14/16 | **1.160** |
| 3 bit/权重，0.437 GB | A，经 `nvg-train` | 1.340 | 7/16 | 0.887 |
| | **B，经 `nvg-train`** | **1.277** | 13/16 | **0.774** |

源模型：测试 NLL 0.189。

**主要发现。** 在 NVG 模型上，DTG-MA 掩码不是破坏，而是修复。不更新任何权重，
仅关闭 27% 的神经元，就把与教师的 KL 从 2.17 降到 1.23。掩码找到编码失真最严重
的神经元并将其移除。"先 NVG"在同等大小下胜出，但优势随比特率升高而缩小：2 bit
时为 −0.31 nat，3 bit 时为 −0.06 nat。同时在 3 bit 时 B 分支更容易循环（13/16 对
7/16）。

### 5.7 MoE：移除专家

LFM2.5-8B-A1B，80 个代码类提示。测试集只有 2 条记录，因此数字仅供参考。

| `--ratio` | 参数量 | 测试 NLL |
|---|---|---|
| 源模型 | 8.468B | 0.280 |
| 0.9 | 7.620B | 0.289 |
| 0.8 | 6.772B | 0.332 |
| 0.65 | 5.494B | 0.399 |

`--ratio 0.8` 时，各层保留 32 个专家中的 20–30 个。在不修复的情况下，去掉 10% 的
参数代价为 +0.009 nat。

### 5.8 教师选择

0.8B，`--ratio 0.9`。通过 API 获得的 Qwen3.8-27B 回答（160 个提示）对比模型自身
的回答（483 个提示）。每个模型都在两位教师的测试集上评估。

| 压缩后的模型 | 在 27B 回答上的测试 NLL | 在自身回答上的测试 NLL |
|---|---|---|
| 教师：27B（API） | **0.492** | 0.442 |
| 教师：模型本身 | 0.564 | **0.305** |

每个模型在自己教师的分布上更好，而且 27B 方案的数据只有三分之一。NLL 无法说明
哪位教师更好，需要检验回答本身，例如执行代码。

## 6. 主要结论

1. **决定压缩质量的是渗透折叠（专利 15），而不是神经元的选择。** 以闭式解转移
   被移除神经元的贡献，可使 FFN 输出误差减半。改用直接截断对 NLL 的伤害超过任何
   其他参数。
2. **修复抹平了神经元选择上的差异。** 训练得到的 DTG-MA 掩码选择得比照明更好，
   但修复后结果相同（0.8B），或照明略好（2B）。
3. **剪枝作为量化修复。** 在用 NVG 编码到低于常规精度的模型上，DTG-MA 掩码移除
   编码受损最严重的神经元，在不训练权重的情况下把与教师的 KL 降低近一半。由此
   得出混合方案的顺序：先量化，再掩码和切除。
4. **掩码压力必须按几何级数增长。** 线性的 L1 调度会让剪枝停在约 15%。
5. **小型稠密模型的压缩应来自量化。** 宽度剪枝代价高，应叠加在量化之上。
6. **折叠的岭参数几乎总是 0.1。** 这说明最优岭参数高于旧的网格。新网格
   `0.03, 0.1, 0.3` 得到相同结果，并少做一次分解。

## 7. 局限

- 只有 Qwen3.5-0.8B 和 2B（外加一个 MoE），一个任务集（代码），一个随机种子。
  0.8B 的测试集为 34 条记录；小于约 0.01 nat 的差异在噪声范围内。
- 教师回答上的 NLL 不等于任务完成质量。代码需要通过执行来检验。
- 顺序实验中 2–3 bit/权重的分支在绝对意义上已经损坏：几乎处处循环。方法顺序
  已经测出，但在这个比特率下还没有可用的模型。
- 在 M4 上，0.8B 的修复受限于约 3 TFLOPS 的 GEMM，需要数小时。完整运行在
  RTX 5090 上完成。

## 8. 复现

```bash
# 任务 → 压缩后的模型（模型自己当教师）
cortiq compress Qwen3.5-0.8B.cmf prompts.jsonl q08-task.cmf --ratio 0.9

# 用训练得到的 DTG-MA 掩码选择神经元（专利 2）
cortiq compress Qwen3.5-0.8B.cmf prompts.jsonl q08-dtgma.cmf --ffn 0.75 --select dtgma

# 通过 API 的教师模型
OPENROUTER_API_KEY=… cortiq compress Qwen3.5-0.8B.cmf prompts.jsonl q08-27b.cmf --ratio 0.9 \
    --teacher openrouter:qwen/qwen3.8-27b --no-think --api-concurrency 32

# NVG → DTG-MA 顺序：编码、掩码与切除、度量训练
cortiq convert --model Qwen/Qwen3.5-0.8B --quant blr --bpw 2.0 --blr-paths 2 --calib task.txt --output nvg.cmf
cortiq nvg-train nvg.cmf --teacher q08-q8_2f.cmf --calib task.txt --tokens 262144 --epochs 2 \
    --mask-keep 0.75 --output nvg-dtgma.cmf

# 在同一批测试记录上比较多个模型
cortiq compress Qwen3.5-0.8B.cmf task.tdata run.cmf --ffn 0.75 --compare nvg-dtgma.cmf --compare other.cmf
```

在大模型上的后续工作——在一块 RTX PRO 6000 上把 Qwen3.8-27B 压缩到 14B——见
[QWEN38_27B_TASK_COMPRESSION.zh.md](QWEN38_27B_TASK_COMPRESSION.zh.md)。

## 参考文献

1. O. U. Kirichenko. USPTO 申请 19/452,464（专利 2），2026 年 1 月 19 日提交：
   *System and Method for Dynamic Task-Guided Neural Network Compression with
   Catastrophic Forgetting Prevention*。
2. O. U. Kirichenko. USPTO 申请 19/731,402（专利 15），2026 年 7 月 6 日提交，
   为 19/452,440 和 19/452,464 的部分延续申请：*Unified Execution Architecture
   for Serving a Plurality of Specialized Language Models from a Single Shared
   Backbone via Dynamically Overlaid Compressed Delta Representations Without
   Materializing Separate Models*。
3. O. U. Kirichenko. USPTO 申请 19/738,763（专利 16），2026 年 7 月 13 日提交：
   *Training-Free Conversion of Softmax Attention Layers of a Pre-Trained
   Language Model into Constant-Memory Streaming Attention … and
   Teacher-Anchored Surroundings-Only Quality Restoration*。
4. O. U. Kirichenko. 专利 12（AWNP）：*System and Method for Activation-Weighted
   Structured Compression of Neural Networks with Architecture-Preserving Layer
   Selection*。
5. O. U. Kirichenko. USPTO 申请 19/452,440（专利 1），2026 年 1 月 19 日提交：
   *System and Method for Unsupervised Multi-Task Routing via Signal
   Reconstruction Resonance*。
6. O. U. Kirichenko. *Frozen Core Decomposition (FCD): An Architectural Approach
   to Continual Learning Without Catastrophic Forgetting*. Zenodo, 2025.
   DOI 10.5281/zenodo.18006952.
7. O. U. Kirichenko. *DTG-MA: Hard Guardrails Against Catastrophic Forgetting
   Through Attention Masking and Task Graphs*. Zenodo, 2025.
   DOI 10.5281/zenodo.18088857.
8. O. U. Kirichenko. *Topology First: Sub-1-Bit LLM Compression by Separating
   Sign Topology from Metric (NVG)*，草稿，2026；*Null-Vector Gravity*，研究笔记。
9. Lee, Kim, You, Kim. *LittleBit: Ultra Low-Bit Quantization via Latent
   Factorization*. NeurIPS 2025 —— `blr` 容器的布局。
10. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative
    Pre-trained Transformers*. arXiv 2210.17323.
