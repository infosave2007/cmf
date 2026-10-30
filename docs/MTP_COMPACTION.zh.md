# MTP 草稿 FFN 的独立压缩

*语言：[English](MTP_COMPACTION.md) · [Русский](MTP_COMPACTION.ru.md) · **中文***

`cortiq mtp-compact` **只**缩小 MTP 草稿（draft）的稠密 FFN 层。主干
（backbone）不会被重新打包：计划中明确记录 `trunk.mode = preserve`。这样就可以
单独确定草稿层的大小，然后测量权重变小能否抵消接受率（acceptance）的下降。
该命令既不承诺质量，也不承诺加速：两者都必须通过下文所述的测试来确认。

> 这里的 MTP 是投机解码（speculative decode）使用的草稿，而不是主 MoE/主干。
> 物理压缩会改变草稿 FFN 的神经元，因此可能改变被接受的草稿 token 数量，
> 但不得改变主模型经过验证的 greedy 输出。

## 支持的情形

支持规范（canonical）**稠密** MTP 的两种存放方式：

1. `model.mtp.*` 位于主 CMF 中：结果是一个新的主 `.cmf`。
2. MTP 位于 sidecar 文件 `model.mtp.cmf` 中（MiMo 的典型情况）：命令会自动
   找到它，并为候选模型生成**完整的一对文件**。

`--output` 最好指定新的可运行**主** CMF 的路径。对于 MiMo，命令会在它旁边
自行创建对应的 `<output-stem>.mtp.cmf`：未分片的主干先从源文件建立硬链接
（不额外复制）；若文件系统不支持硬链接，则改为复制。分片的主干则以流式方式
合并（coalesce）为一个候选文件。硬链接是同一批字节的别名，命令不会修改它：
`mtp-compact` 既不写入源路径，也不写入别名。作业运行期间不要修改源文件；
硬链接方式成功之后，也不要"就地"编辑候选主文件——它与源文件是同一个
inode。如果需要可以独立修改的主文件，请在验证之后另行 copy/reflink。因此，
一条命令得到的是一对可用的文件，而不是单独的 sidecar：

```text
base/model.cmf              # 源主干，不会被修改
base/model.mtp.cmf          # 源草稿，不会被修改

candidate/model.cmf         # 新的可运行主文件（hard link/copy/coalesce）
candidate/model.mtp.cmf     # 新的紧凑草稿
```

为兼容早期脚本，外部 MiMo 草稿可以显式写成
`--output candidate/model.mtp.cmf`。在这种**旧式仅 sidecar 模式**（legacy
sidecar-only）下，只发布新的 sidecar 和计划；不会创建
`candidate/model.cmf`，这样的结果本身也不是可运行的模型，需要手动把它放到
兼容的主 CMF 旁边。新的或自动化的运行请始终给出不带 `.mtp.cmf` 后缀的主文件
路径。

不要替换源文件。命令拒绝覆盖已存在的主文件、sidecar 或计划，包括悬空的符号
链接。支持分片的**主文件**：流式写入器通过 mmap 读取合并后的分片，生成一个
新的、可独立运行的未分片候选文件（`candidate/model.cmf`）。对于 MiMo，这同样
适用于分片主文件加一个普通 `.mtp.cmf` sidecar 的情况：主文件被合并，草稿在
其旁边被压缩。分片的外部 `.mtp.cmf` 目前仍按 fail-closed 原则被拒绝——MiMo
加载器有意只查找一个名称由主文件派生的 sidecar。

## 校准与计划

请提供已定稿（finalized）且可校验的 `.tdata` 选择器。选择通道时只使用
train/calibration 划分；validation/test 留给后续评估。命令会自行校验主 CMF、
发现的 MTP sidecar 和 `.tdata`。在加载运行时之前，预检会解析**所有**
`records.jsonl`/`tokens.idx` 行（包括未选中的 val/test），并按 manifest 中的
size/SHA 认证负载。未选中的记录不会分配 `Vec<u32>`；没有 manifest 描述符的
旧版 v1 以固定的 64 KiB 缓冲区通过同样的读取栅栏。随后，选中的 train 记录
通过普通的流式遍历读取，用于校准。

预检之后，`.tdata` 必须在作业结束前保持不变。源 CMF 有开始/结束身份栅栏
（identity fence），而在长达数小时的校准之后再对数据集做一次完整 SHA 遍历是
有意不做的：那将是又一次完整的 I/O 开销。不要在命令运行期间替换
`records.jsonl`、`tokens.idx` 或 manifest。

这里的 O(1) 专指未选中记录的 **token 负载**。对原生 `input.format=jsonl-v1`，
严格的 prompt 账本（prompt-ledger）检查有意保留 `id → split` 映射和已见 ID
集合，即所需内存随 prompt 数量增长。prompt 达到数百万条时，不要认为这个账本
是免费的：请规划内存，或者准备单独的已定稿校准选择器。这是为检测所选划分
之外的重复或被替换的 prompt 而有意付出的代价；命令不会悄悄削弱它。

如果 manifest 包含原生 `tokenizer` 契约，命令会在第一次前向之前以
fail-closed 方式将 `vocab_size`、`vocab_sha256` 和 `config_sha256` 与主 CMF
核对。这样可以防止用另一个 ID 范围重叠的模型的 token ID 悄悄完成校准。没有
契约的旧式通用 `.tdata` 仍受支持，但会输出警告，并在计划中标记为
`tokenizer_contract: "absent-legacy"`；匹配的原生契约标记为 `"verified-v1"`。

目前校准在规范上**只在 CPU 上进行**：`probe_mtp_ffn` 有意在 CPU 上执行
teacher-forced 主干和 down 投影之前的草稿激活，并使用确定性的 `f64` 累加。
在此探测期间，运行时会强制关闭 `CMF_FFN_GATE_TOPK` 和 `CMF_FFN_ORACLE_TOPK`，
因此计划始终描述完整的稠密草稿，而不是某个偶然的服务端实验。`CMF_GPU=1`
不会加速这一阶段，也不应被理解为承诺 GPU 转换；Metal/Vulkan 在完成的候选
文件上另行检查。这让同一份计划不受各后端 FMA/SIMD 和归约顺序差异的影响。

物理写入面向大模型设计：保留的主干张量以 16 MiB 为单位从 mmap 传给
`CmfStreamWriter`，不会为整个模型分配 `Vec<u8>`。在私有的 `mtp-compact`
阶段，未修改负载已校验过的哈希在写入时被复用，因此不会仅为得到同一个哈希而
重读源文件；发布之前，候选文件仍会被完整地重新校验。MTP FFN 逐个投影
（projection）重写：同一时刻只存在一个投影的原始 f32 矩阵和压缩后的 f32
矩阵（对 `down_proj.t` 还有其临时转置）；已编码的 `down_proj` 在构建转置之前
被显式释放，而不是同时持有全部 gate/up/down。规范 MTP 校准也不保存完整的
主干 hidden 历史；MiMo 重放在一次探测内保留 128 行主干数据。主要的内存上限
现在是显式的：每条源记录被切分为不超过 `--max-record-tokens`（默认 4096）的
连续 fresh-causal 块，主干/草稿的 KV 在块边界处重置。因此，单条异常长的
`.tdata` 记录无法让 KV 增长到整条记录的长度；因果状态校准的内存受上限和模型
几何结构约束，而不受记录上下文长度约束。`.tdata` 读取器本身仍会把选中的记录
物化为 `Vec<u32>`，所以不要用一条病态的、长达数百万 token 的记录代替正常的
校准记录。这**不会**保留跨越边界的注意力：对质量要求高的长上下文模型，请
选择与工作上下文和可用内存相当的上限，而不要为了速度把它调小。磁盘必须能
同时容纳源文件和新的候选文件。不要同时对同一模型向同一个输出目录运行两次
重写。

对源 `q4tp`/`q2tp` 的 gate/up 还有一条额外的精确快速路径：保留的神经元是
hidden 轴不变的一整行，因此其权重平面（weight plane）、行阶梯（row ladder）
和缩放码（scale code）按新顺序逐字节复制。这样可以**精确**保留解码后的选中
行，不做第二次 PTQ，也不分配 f32 的 gate/up 面板。`down_proj` 和
`down_proj.t` 不能这样复制：压缩会选择并重排它们的列，从而改变编解码分组；
`down_proj` 被重新打包，转置则据此重新构建。不过对 `q4tp`/`q2tp`，写入器只
直接从行/组平面中解码 `down_proj` 实际选中的源列，不物化完整的源 f32 面板，
然后按常规方式编码紧凑矩阵。选中的列先按其源 32 宽 tile 分组，因此每个组/行
的 ladder/code 只读取一次，而结果中仍保持排序顺序。对于缺少这里及下文所述
直接路径的编解码格式，gate/up 和 down 仍走正确的 dequant → select → requant
路径。

对原始 `f32`/`bf16`/`f16`、`q8_row` 和 `q8_2f`，快速路径同样精确，但原因
不同：gate/up 复制选中的行，`down_proj` 则选择并重排原始值或 q8 码，连同它们
原有的行/列缩放。因此收窄不会多做一次 PTQ，也不会在源 CMF 之上引入额外误差；
`bf16` 保持自身 dtype，不会被悄悄降为 f16。如果存在 `down_proj.t`，只会短暂
解码已压缩的选中 `down_proj` 来构建转置；不会物化完整的源 `down_proj`。其余
编解码格式继续使用 fail-safe 回退路径。

实际的内存预算取决于**单个**最大的草稿投影，而不是主干。评分遍历现在直接在
mmap 字节上计算 `down_proj_column_l2_squared`，只保留一个 `Vec<f64>(I)`，
适用于 `f32`、`f16`、`bf16`、`q8_row`、`q8_2f`、`q4tp` 和 `q2tp`；f32 反量化
和 f64 求和的顺序与历史上的全面板排序器一致。因此对常用的生产编解码格式，
这一阶段不会创建临时的 `[H,I]` f32 缓冲区。少见的未知或复杂编解码格式仍走
fail-safe 的 `dequant → score` 回退路径，可能需要完整的源面板。对原始
`f32`/`bf16`/`f16`、`q8_row`、`q8_2f`、`q4tp` 和 `q2tp`，烘焙（bake）同样
不会分配完整的 f32 源 `down_proj`；在 `down_proj.t` 阶段需要同时存在压缩后的
`down_proj`、它的 f32 转置和编码后的转置；此时已写出的编码 `down_proj` 已被
释放。其他编解码格式在烘焙本身期间仍可能暂时同时持有原始面板和压缩后的
面板。

在每个草稿层的评分遍历之前，命令会打印层号、`down_proj` 的形状、编解码格式
和已用时间；这是为非常宽的矩阵有意设置的单独进度栅栏，并不表示卡住。计划的
层报告中保存 `down_proj_dtype`，这样作业结束后可以看到评估的是哪条存储路径。

这是有意的折中：不需要把完整的 `Vec<TensorSpec>` 或整个模型放进内存，但非常
宽的单个 MTP 层仍然需要足够的内存。如有疑问，先用一层或小模型试跑并观察
RSS，不要并行运行转换作业。

完整性校验是有意付出的安全代价：校准前后都会完整校验源文件；烘焙之后再次
校验源 MTP 草稿，发布之前完整校验候选文件。对 100B+ 模型，这意味着对磁盘做
多次顺序遍历；这不是"卡住"，也不承诺瞬间完成转换。每一次完整的 CMF 扫描
（包括外部 MiMo sidecar）都会打印开始/结束和墙钟时间；校准进度在第一个以及
此后每第八个有界块之后打印。命令结束前不要修改源文件：重复校验能发现常见的
就地替换，命令也不会发布混合的候选文件。

在真实的 teacher-forced MTP 路径上，它为每个通道收集

```text
sum_teacher_forced((activation(gate, up))^2) * down_proj_column_l2_squared
```

并保留得分最高的通道。这不是主干统计量，也不是根据主干 `intermediate_size`
做的猜测。`activation(gate, up)` 正是运行时中该草稿的 MLP 函数（例如
`silu(gate) * up`），并不总是 SiLU；其契约也会写入计划。`--keep-ratio` 作用
于每个草稿层，之后宽度向下取整到 32 通道一组；最少为一个 32 通道组。最终的
JSON 计划记录：选择器、读取的选中记录数以及单独统计的实际贡献记录数、实际的
块/token 数、上限与切分策略、tokenizer 契约状态、主文件/草稿各自的目录哈希
和合并后的 `trunk_hash`、通道顺序以及保留的能量比例——请把它和产物放在一起
保存。`trunk_hash` 覆盖所有合并分片的张量条目及其负载哈希。当
`mtp.mode=compact` 时，`cortiq tube-bake` 会将必需的
`source_draft_directory_hash` 以及（新计划中的）`source_draft_trunk_hash` 与
输入草稿核对；校准好的计划不会被意外应用到另一个形状相同的 sidecar 或被
替换的后续分片上。

较短的记录/块仍然参与它们所能到达的草稿层的统计，但准入前缀（admission
prefix）取自**最长**的合适有界块（不超过 256 个 token，且不短于
`draft_layers + 2`）。这样可以让小型运行时闸门覆盖更多位置，而无需保留整条
记录。如果没有这样的块，或者某一层在聚合后激活能量为零，命令会拒绝以任意的
平局规则（tie-break）选择通道，而不是输出一个形式上有效、却无法验证的紧凑
草稿。

## 针对自有 prompt 的简单命令

常见情况既不需要单独的教师模型，也不需要手工准备数据目录：

```bash
cortiq compact source.cmf prompts.jsonl task-model.cmf
```

`prompts.jsonl` 是普通的 JSONL：每行一个任务。简单的 `prompt` 就够了；对聊天
模型也可以传入完整的 `messages` 数组：

```jsonl
{"id":"sql","prompt":"写一个按 id 查找订单的安全 SQL 查询。"}
{"id":"review","messages":[{"role":"system","content":"回答要简洁。"},{"role":"user","content":"检查这段 Rust 代码是否有 off-by-one 错误。"}]}
```

这种形式把 `source.cmf` 用作教师模型，在候选文件旁边创建 `task-model.tdata`，
根据 `prompts.jsonl` 生成确定性的直接回答（direct-answer）记录，定稿并校验
数据集，然后校准并物理压缩 MTP 草稿，最后只发布经过验证的可运行
`task-model.cmf`。以相同的输出再次运行时，已定稿的 `task-model.tdata`
**不会被覆盖**，而是被复用；对另一组 prompt，请选择新的输出/缓存（或在运行前
有意删除旧缓存）。

这不是通用微调：主干与源文件逐字节相同，只有现有 MTP 草稿的稠密 FFN 发生
变化。投机验证器仍然是输出 token 的来源，因此 greedy 输出不应因接受了草稿而
改变。没有受支持的稠密 MTP 的模型会按 fail-closed 原则被拒绝，而不会得到
虚假的"适配"。为控制比例、划分和各项限制，简单形式接受 `--keep-ratio`、
`--max-tokens`、`--samples`、`--seed`、`--split`、`--limit`、`--max-records`、
`--max-record-tokens`、`--keep-truncated` 和 `--disable-mtp`。例如，把草稿
收窄 25% 的快速冒烟运行：

```bash
cortiq compact source.cmf prompts.jsonl task-model.cmf \
  --keep-ratio 0.75 --max-tokens 512 --limit 24 --split 100/0/0
```

若需要单独的教师模型、独立的 `.tdata` 路径、划分种子或审计计划，请使用下文
的扩展命令 `mtp-compact`。

## 一次运行：创建或复用教师数据

`mtp-compact` 可以在物理压缩之前自行准备缺失的校准数据集。这个编排器
**仅用于稠密 MTP 草稿 FFN**；它不是面向主干、MoE 或 DSpark 的通用
`compress`。

```bash
BIN=target/release/cortiq

# 若 calibration.tdata 尚不存在，命令会根据 teacher/prompts 创建它，
# 定稿后立即校准并写出候选文件。
"$BIN" mtp-compact base/model.cmf \
  --data calibration.tdata \
  --teacher /models/teacher.cmf \
  --prompts calibration_prompts.jsonl \
  --teacher-max-tokens 512 \
  --teacher-samples 1 \
  --teacher-seed 42 \
  --teacher-split 90/5/5 \
  --keep-ratio 0.90 \
  --max-record-tokens 4096 \
  --output candidate/model.cmf
```

`--data` 保持为显式路径，以便检查、保存和复用生成的语料。可以使用
`PATH.tdata[:train][=SHARE]`：压缩只使用 train 划分（必要时只用其中确定性的
一部分），把 val/test 留给单独评估。自动创建时，目录根始终是选择器/后缀之前
的 `PATH.tdata` 部分。目录不存在时，必须同时提供 `--teacher` 和 `--prompts`；
教师模型生成确定性的 greedy 直接回答记录（`--no-think`、`top-k=0`，不带
`--raw`），写入器先定稿一个不可变的 `.tdata`。教师模型可以与源文件相同，但
数据集的 token ID 和 tokenizer 契约仍会在第一次校准前向之前与源文件核对。
甚至在漫长的生成**之前**，元数据预检就会比较源文件与教师模型的词表和
tokenizer 配置的精确字节：不兼容的组合会被拒绝，不会浪费 pod 时间。对于马上
就要压缩的短小冒烟数据集，请设置 `--teacher-split 100/0/0`，让可用记录进入
train。

目录已存在时，命令**既不覆盖也不重新生成它**：只复用已定稿/可读的 `.tdata`，
打印这一决定，并照常执行完整的完整性/tokenizer 预检。此时传入的
`--teacher`、`--prompts` 和教师生成参数会被有意忽略。如果复用时仍然传入它们，
CLI 要求 `--teacher` 和 `--prompts` 一起提供。不完整或损坏的目录会按
fail-closed 原则失败；请选择新路径，而不要试图悄悄续写它。

生成时还可使用 `--teacher-limit`、`--teacher-keep-truncated`、
`--teacher-disable-mtp` 以及 `--teacher-split-seed`。`--max-records` 和
`--max-record-tokens` 属于压缩阶段，而不属于回答生成阶段。

## 手动模式：现成的教师数据

一条命令同时适用于内嵌 MTP **和**使用预先准备好的数据集的 MiMo sidecar：

```sh
BIN=target/release/cortiq

"$BIN" mtp-compact base/model.cmf \
  --data calibration.tdata:train \
  --keep-ratio 0.90 \
  --max-record-tokens 4096 \
  --output candidate/model.cmf \
  --plan-out candidate/model.mtp-plan.json
```

对内嵌 MTP，它写出 `candidate/model.cmf`；如果源文件是分片的，结果是一个
有效的未分片 CMF。对于 MiMo，它写出 `candidate/model.cmf` 和
`candidate/model.mtp.cmf`，无需手动 `ln`/`cp`；主文件分片时，它以流式方式
合并主文件。`--output candidate/model.mtp.cmf` 这种路径仅对外部 MiMo sidecar
保留为兼容的旧式写法：它发布一个紧凑草稿和计划，但不创建主 CMF。这不是可以
独立使用的模型；新的工作请使用 `--output candidate/model.cmf`，让命令输出一对
可验证、可运行的文件。

在发布最终名称之前，命令使用带随机数（nonce）的私有暂存目录，重新打开并
校验候选文件的哈希，构建 `Pipeline`，确认加载的确实是 MTP 草稿，并在保存的
校准序列上以新的 FFN 宽度执行 teacher-forced 草稿探测。为此，准入阶段保留
最长的合适校准前缀，最多 256 个 token（若草稿更深，则为 `draft_layers + 2`）；
其长度以 `admission_probe_tokens` 写入计划。它执行紧凑草稿的 teacher-forced
探测，**以及**一个单独的仅 CPU plain greedy 闸门：源文件和暂存的候选文件获得
相同的前 ≤16 个前缀 token，在关闭投机分支和 GPU 的情况下必须输出相同的接下来
2 个 token。烘焙之后，源主文件/草稿会按目录哈希**和**合并后的 `trunk_hash`
重新校验；源文件发生变化会导致拒绝。对于 MiMo，还会核对暂存主干的
`trunk_hash` 是否与构建计划时所用的一致。准入闸门不通过，最终候选文件就不会
发布。这个简短的 CPU 闸门不能证明 Metal/Vulkan 内核或投机验证器的正确性——
它们需要下文单独的精确 ID 测试。

在对源文件做完整哈希扫描之前，命令会检查显式 `--output`/`--plan-out` 的拓扑
（包括别名、父/子关系和 `.mtp.cmf` sidecar 名称），并确认这些最终路径的叶节点
是空闲的。因此，像 `--plan-out candidate.cmf/plan.json` 这样的错误不会在 100B
扫描上浪费数小时。真正防竞争（race-safe）的不覆盖（no-clobber）保护仍由最后的
原子发布完成；在 tokenizer 和完整 `.tdata` 预检成功之前，不会创建新的输出父
目录/暂存目录。因此，选择器错误或外来 tokenizer 不会在未来候选文件旁边留下
空的嵌套目录。打开源文件之后、读取 `.tdata` 之前，还会单独检查规范的稠密
MTP 几何结构、可物化的编解码格式和 `--keep-ratio`，这些错误同样不会启动
校准。

发布之前使用原子创建的、带随机数的私有暂存**目录**；内部写入器通过
`create_new` 打开文件，因此冲突不会覆盖他人的暂存内容。负载以及暂存/最终
位置的父目录条目在报告成功之前都会同步（`sync_data`/`fsync`）。每个最终名称
的发布同样不覆盖已有文件；对于 MiMo，sidecar 先于主文件发布，因此不会出现
可见的、缺少草稿的可运行主文件。这并不承诺在文件系统故障时整对文件/计划构成
单个事务：如果第二个最终链接失败，已发布的计划/sidecar 会被有意**保留**，以免
竞争条件删掉别人的文件。重试之前请手动检查并删除这种不完整的产物；命令会
拒绝覆盖它。因此请保留源文件和计划。闸门证明的是 CMF/MTP 的完整性和可运行
性，但不能代替留出集（hold-out）质量评估和目标后端上的基准测试。

`--max-records N` 只在必需的完整 `.tdata` 完整性预检之后才限制**读取的选中
记录**数量；它让计划依赖于截断的样本，并不是跳过大数据集校验的手段。在计划
中，`calibration_selected_records` 记录这次实际受限的遍历，
`calibration_records` 则只记录产生 ≥2 个 token 并贡献了统计量的记录。快速
调试请使用单独的小型已定稿选择器。
`--max-record-tokens N`（默认 4096）是生产环境的安全上限，而不是质量调节
旋钮：`N` 不得小于 `draft_layers + 2`，过小的 `N` 会因 fresh-causal 边界改变
长上下文的激活分布。对准备发布的候选文件，请使用预先固定的完整校准选择器，
以及预先选定并记录在案的上下文上限。

## 独立的草稿宽度

新的文件头保存 `arch.mtp.ffn_intermediate_sizes`——一个可选的 `Vec<usize>`，
记录每个 MTP 层本地的 FFN 宽度。压缩之后它可以小于主干的
`arch.intermediate_size`。旧文件没有该字段；加载器从经过校验的 `gate_proj`、
`up_proj`、`down_proj` 三个张量中得到宽度。若该字段存在，它必须与实际张量
形状一致。MiMo 同样根据每个草稿层自身的张量推导宽度，而不是根据主干的几何
结构。

`eh_proj`、`enorm`、`hnorm`、注意力以及层/最终归一化作用于 hidden 宽度，而
不是 FFN 轴：此操作会逐字节保留它们；只改变相互匹配的 `gate/up/down` 轴（以及
存在时的 `down_proj.t`）和 `arch.mtp.ffn_intermediate_sizes`。因此压缩器不会
试图沿错误的轴"缩小"投影或归一化。

结论：不要修改 JSON 中保存的目录/主干哈希或宽度，也不要应用来自其他模型的
计划。缺失或不匹配的哈希，以及文件头、层数或形状不一致，都是拒绝的理由，
而不是"试着运行一下"的理由。只有 `mimo_v2` 允许多个草稿层，因为运行时只为它
提供了多层执行器。

## 必需的正确性检查

先验证**普通的 CPU greedy**，然后单独在 Metal 后端上验证同样的流程。所有运行
都要固定同一个二进制、模板、`--raw`/prompt、`--greedy`、token 数和环境变量。
比较的是 **token id**，而不是渲染后的文本。`--trace-json` 把 JSONL 写到
stderr。

CPU 是压缩的必需基线检查：它不启用 MTP 配对（pair），因此不可能"成功"接受
有错误的草稿。请显式关闭 GPU，以免从 shell 继承到意外的后端：

```bash
set -euo pipefail

BIN="${BIN:-target/release/cortiq}"
PROMPT="${PROMPT:-Explain in one paragraph why deterministic decoding is useful.}"
mkdir -p verify

extract_trace_ids() {
  local trace_stderr="$1" trace_ids="$2"
  if ! grep '^{"t":' "$trace_stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$trace_ids"; then
    echo "cannot extract token IDs from $trace_stderr" >&2
    return 1
  fi
  if ! test -s "$trace_ids"; then
    echo "trace contains no token IDs: $trace_stderr" >&2
    return 1
  fi
}

trace_cpu_plain() {
  model="$1"; out="$2"
  CMF_GPU=0 CMF_MTP=0 CMF_MIMO_MTP=0 "$BIN" run "$model" \
    --raw --prompt "$PROMPT" --greedy -n 128 --trace --trace-json \
    >/dev/null 2>"$out.stderr"
  extract_trace_ids "$out.stderr" "$out.ids"
}

trace_cpu_plain base/model.cmf      verify/base-cpu-plain
trace_cpu_plain candidate/model.cmf verify/candidate-cpu-plain
cmp verify/base-cpu-plain.ids verify/candidate-cpu-plain.ids
```

对内嵌 MTP 只需检查一个 `.cmf`；对于 MiMo，还要对两个模型的主文件和
`.mtp.cmf` 执行 `"$BIN" verify`。准入命令已经做了结构检查，但这一追踪必须在
多个留出 prompt 上、以足够的长度运行。

### 主机融合配对不是发布默认值

`CMF_MTP_PAIR=1` 启用实验性的 CPU 融合配对（fused pair）。它优化两个位置的
运算，但不是验证器。在 q4tp Qwen3.5 上已经发现一个近似平局（near-tie）的
情形：融合配对给出的 greedy ID 与两次串行目标前向不同。因此标准的 CPU
`CMF_MTP=1` **不会**启用这条路径；不要把 `mtp_drafted` 为零的运行当作投机
加速的证据。

如果某个具体模型和 CPU 后端需要使用主机配对，请先用
`CMF_GPU=0 CMF_MTP=1 CMF_MTP_PAIR=1` 对源文件和候选文件分别运行单独的长
精确 ID 闸门，与它们各自的 plain 追踪对比。一旦出现不一致，就保持该变量
关闭。这个实验不能替代下文的 Metal 资格验证。

对 GDN 混合模型，主机配对还需要显式设置 `CMF_MTP=1`，因为其循环部分无法
并行，默认被排除在该实验之外；这不是推荐的速度预设。

追踪也是检查契约的一部分：每个输出的 token 必须恰好对应一行 JSON，包括一次
批处理图投机轮次中被接受的额外草稿 token。对于这些被接受的行，运行时不会仅为
遥测再做一次完整的 softmax，因此 `confidence` 如实地等于 `0.0`；这并不表示
概率为零，也不应用作质量指标。

## Metal/Vulkan 上的接受率与速度

首先确认后端可见：

```sh
"$BIN" gpu
```

### 原生 Metal：安全的默认值

对标准的规范 MTP，原生 Metal 的**批处理图验证器默认关闭**。这是有意的
fail-closed 默认值：在一个真实模型上，批处理验证器的某一近似平局行给出的
*输出* greedy ID 与普通串行 Metal 路径不同。因此，在未设置 `CMF_GRAPH_SPEC`
和 `CMF_MTP_PAIR` 的情况下，`CMF_GPU=1 CMF_MTP=1` 保持普通的串行目标解码；
这不是草稿加载错误，也不是重新转换模型的理由。在这样的运行中，规范 MTP 的
`mtp_drafted=0` 是预期的安全结果。

`CMF_GRAPH_SPEC=1` 只作为诊断内核的显式实验保留。它**不是生产速度预设**：
有限的 prompt 集合无法证明不存在下一个近似平局行。在批处理和串行内核被修复
并再次经过广泛压力测试之前，不要把这个变量写进模型说明。单独的
`CMF_MTP_PAIR=1` 同样是上一节所述的实验性主机融合配对；它既不会启用也不会
修复 Metal 图验证器。MiMo 投机由 `CMF_MIMO_MTP` 单独控制，因此这一限制专门
针对规范 MTP 图验证器，而不是所有 MTP 草稿。

运行安全的 Metal 冒烟测试/基准测试时，请显式移除实验性变量：

```sh
env -u CMF_GRAPH_SPEC -u CMF_MTP_PAIR \
  CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=0 \
  "$BIN" bench candidate/model.cmf --tokens 256 --ignore-eos --json \
  > verify/candidate-metal-serial.json
```

这是对紧凑候选文件普通 Metal 生成的正确测量，但不能证明 MTP 加速。请从
JSON 中保存 `decode_tok_s_steady`、后端、设备和二进制版本；不要把这一结果
称为投机基准。

### 原生 Metal：仅用于诊断的精确 ID 闸门

下面的闸门**仅用于调查** `CMF_GRAPH_SPEC=1`。CPU 与 Metal 的运算差异本身
不算错误，但在同一个 Metal 后端内，plain 与实验性图的 ID 必须一致。请使用
多个留出 prompt 和足够的长度。`CMF_METAL_VERIFY_CHECK=1` 正是原生 Metal 的
诊断工具：它打印批处理与串行的对比，但不会改变验证器已经做出的决定，也不是
回退机制。

```bash
set -euo pipefail

BIN="${BIN:-target/release/cortiq}"
PROMPT="${PROMPT:-Explain in one paragraph why deterministic decoding is useful.}"
mkdir -p verify

extract_trace_ids() {
  local trace_stderr="$1" trace_ids="$2"
  if ! grep '^{"t":' "$trace_stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$trace_ids"; then
    echo "cannot extract token IDs from $trace_stderr" >&2
    return 1
  fi
  if ! test -s "$trace_ids"; then
    echo "trace contains no token IDs: $trace_stderr" >&2
    return 1
  fi
}

trace_metal_graph_experiment() {
  mode="$1"; model="$2"; out="$3"
  if [ "$mode" = plain ]; then
    CMF_GPU=1 CMF_MTP=0 CMF_MIMO_MTP=0 CMF_METAL_VERIFY_CHECK=1 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  else
    CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=0 CMF_GRAPH_SPEC=1 \
      CMF_METAL_VERIFY_CHECK=1 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  fi
  extract_trace_ids "$out.stderr" "$out.ids"
}

trace_metal_graph_experiment plain base/model.cmf      verify/base-metal-plain
trace_metal_graph_experiment graph base/model.cmf      verify/base-metal-graph
trace_metal_graph_experiment plain candidate/model.cmf verify/candidate-metal-plain
trace_metal_graph_experiment graph candidate/model.cmf verify/candidate-metal-graph

cmp verify/base-metal-plain.ids verify/candidate-metal-plain.ids
cmp verify/base-metal-plain.ids verify/base-metal-graph.ids
cmp verify/candidate-metal-plain.ids verify/candidate-metal-graph.ids

# 任何 MISMATCH 都意味着批处理验证器在这次运行中
# 已不再精确。它不会自动回退。
if grep -H ' MISMATCH ' verify/*-metal-graph.stderr; then
  echo 'Metal graph verifier mismatch: leave CMF_GRAPH_SPEC unset.' >&2
  exit 1
fi
```

成功通过的简短闸门有助于查找内核缺陷，但不会取消上述安全默认值。如果
`mtp_drafted` 为零，说明图根本没有被准入，这样的追踪也不能检验投机路径。

### Vulkan/WGPU

Vulkan 保留现有的图策略，但若没有在**同一**设备上通过精确 ID 闸门，仍不能
宣称加速。请显式选择 Vulkan，在 pod 上固定 `XDG_RUNTIME_DIR`，并在同一后端内
比较 plain 与投机追踪。这是规范 MTP 的示例；MiMo 请另行用 `CMF_MIMO_MTP=1`
运行检查，因为它的轮次不受 `CMF_GRAPH_SPEC` 控制。

```bash
set -euo pipefail

BIN="${BIN:-target/release/cortiq}"
PROMPT="${PROMPT:-Explain in one paragraph why deterministic decoding is useful.}"
mkdir -p verify

extract_trace_ids() {
  local trace_stderr="$1" trace_ids="$2"
  if ! grep '^{"t":' "$trace_stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$trace_ids"; then
    echo "cannot extract token IDs from $trace_stderr" >&2
    return 1
  fi
  if ! test -s "$trace_ids"; then
    echo "trace contains no token IDs: $trace_stderr" >&2
    return 1
  fi
}

trace_vulkan() {
  mode="$1"; model="$2"; out="$3"
  if [ "$mode" = plain ]; then
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" WGPU_BACKEND=vulkan \
      CMF_GPU=1 CMF_MTP=0 CMF_MIMO_MTP=0 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  else
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" WGPU_BACKEND=vulkan \
      CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=0 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  fi
  extract_trace_ids "$out.stderr" "$out.ids"
}

trace_vulkan plain base/model.cmf      verify/base-vulkan-plain
trace_vulkan spec  base/model.cmf      verify/base-vulkan-spec
trace_vulkan plain candidate/model.cmf verify/candidate-vulkan-plain
trace_vulkan spec  candidate/model.cmf verify/candidate-vulkan-spec
cmp verify/base-vulkan-plain.ids verify/base-vulkan-spec.ids
cmp verify/candidate-vulkan-plain.ids verify/candidate-vulkan-spec.ids
```

在新的 Vulkan 设备上，先用 `CMF_MTP_GRAPH=0` 运行这个闸门，然后在通常的图
设置下重复。准入之后出现的迟到 `Declined` 会按 fail-closed 原则取消投机轮次，
而不保证透明地切换到主机路径；请查看 `mtp_drafted` 和日志。只有当
`mtp_drafted > 0` 时，才统计接受率，并在相同的二进制、上下文、设备和供电
条件下交替多次运行 base/candidate，比较 `decode_tok_s_steady`。更小的草稿
可能降低接受率并使候选文件更慢——这也是有效的结果。

q4tp 使用 `gate_proj` 的本地宽度，而不是主干的 `intermediate_size`；32 通道
对齐使其仍适用于图路径。q8_2f 需要满足自身的对齐/`graph_x` 条件。规范 MTP 的
q2tp 有意不被原生 Metal 草稿图接受，仍走普通路径。具有独立草稿宽度的 MiMo
sidecar 可以正确加载，但它的小批量草稿目前还没有专门的融合原生 Metal MTP
图。因此 `mtp-compact` 能减少其草稿存储和 CPU 工作量，但在没有单独的精确 ID
和基准测试证据之前，不承诺 Metal 加速。

## 限制与拒绝

该命令有意采用 fail-closed 策略。它面向完整的规范稠密 MTP FFN 三元组，形状为
`gate=[I,H]`、`up=[I,H]`、`down=[H,I]`，其中 `I` 是 32 的倍数。以下情况在没有
单独实现/验证之前不受支持：

* 路由式/MoE 或分段式 MTP FFN、不完整的张量三元组以及不匹配的形状；
* 宽度无法容纳一个 32 通道组的 MTP、无效的保留比例，或为空的 teacher-forced
  校准；
* 无法表示为这种稠密三元组的特殊 MTP 格式，以及不是 MTP FFN 的外部
  DSpark/草稿 sidecar；
* 需要从分片中物理合并 MiMo 主文件时的冻结基因组（frozen genome）：完整重写会
  破坏仅追加（append-only）布局，因此这样的输入会被拒绝；
* 缺少 `selection.source_draft_directory_hash`、`source_draft_trunk_hash` 不
  匹配（新计划中）或哈希属于其他草稿的计划；修改计划并不会取消对源张量/配置
  的检查。

在所有检查完成之前，请保留源主文件、sidecar 和计划。只有在通过文件完整性
检查、精确 greedy 对比，以及按上述相应精确 ID 策略进行的目标后端基准测试
之后，才能接受紧凑草稿。
