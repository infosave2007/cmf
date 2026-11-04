# Embryo 聊天模型的组装方式

一个封存的 Embryo 基因组、与之并列的知识记录，以及选择记录的共振路由器。所有步骤都在 `scripts/embryo/` 中。

## 训练（GPU）

命令见[英文版](EMBRYO_CHAT.md#training-gpu)：`build_brief_mix.py` 把每个回答截断为最前面的几句，`sft-prepare --seq 256`，然后 `sft --replay-fifths 2`（3/5 聊天记录加 2/5 原始文本窗口，基础模型继续学习）。在 24 GB 的 Mac 上一次只运行一个训练进程。

聊天训练会削弱长距离记忆。修复不触及网络其余部分：`birth --train-only layers.3.attn,layers.7.attn` 只在带检索任务的原始文本窗口上训练锚点层，其他张量和路由描述符逐字节不变。

## 组装

`scripts/embryo/assemble_chat.sh <ckpt> <dir>` 导出带聊天模板和文件内采样默认值的封存（`DTYPE`，发布版为 q8_2f）基因组（`--chat`、`--sampling`），构建记录，在 GPU 上计算 φ 并拟合路由器。`scripts/embryo/eval_chat.sh <dir>` 运行各项门槛检查。

## 路由器

`router.phi.pool = key_span`。引擎在消息中找到表键，并对 φ 层之后的隐藏状态做池化：φ = unit(unit(键词元的均值) + α·unit(消息其余部分的均值))。每条记录是一个独立子空间（均值加前 k 个主方向）；请求进入重构误差最小的记录，前提是它比主干子空间好出一个裕度。只有表中含该键的记录才是候选，没有键的消息直接交给主干，不做 φ 前向。

既没有键也不是追问的消息，按整条消息的 φ 在两个子空间（`router.keyless`）之间判断：询问具体事实的问题与普通对话。询问事实的问题转到 `dontknow` 记录并如实回答，因为网络本身不掌握事实；闲聊、任务和建议交给网络。`--keyless` 记录用第一条的卡片回答。

前一轮提到键、当前消息没有键时（追问），按 φ = unit(unit(前一轮的键词元) + `followup_alpha`·unit(当前消息)) 和 `followup_margin` 判断，使用自己的独立子空间（`router.followup`）：关于该记录的追问与前面带同一键的无关消息。第一轮的子空间没有见过这种形式，普通追问的通过率只有 1–2 %。仅适用于带字段的卡片。

用于拟合的通用集合必须包含日常语句中出现表键的提示，并按键限量，否则一个高频词会占满主干子空间。

## 小模型的文件默认值

`sampling.loop_stop = 1`：回答尾部开始重复时停止（结束原因 `loop`），返回文本中该段只出现一次。`sampling.history_turns = 1`：网络只看到最后一轮用户消息，因为小模型会复制自己之前的回答。记录始终保留对话记忆。

## 运行

`cortiq run` 和 `cortiq serve` 默认让 Embryo 文件运行在常驻 GPU 图上（`CMF_GPU=wgpu`、`CMF_EMBRYO_RESIDENT=1`，在 Mac 上是基于 Metal 的 wgpu），除非你指定了后端。`CMF_GPU=0` 走 CPU 路径，贪心输出文本相同。Embryo 文件在 CPU 上使用浮点激活值，因为它的路由器是按精确的 φ 拟合的，而其他 q8 模型所用的 8 位激活值会使 φ 偏移 4e-3（`CMF_SDOT=1` 可开启：预填充快 5 倍，文本不同）。两条路径与精确运算的差距都不超过 3e-5。

C ABI（`crates/cortiq-ffi`，移动应用基于它构建）用同样的引擎调用路由每轮对话，`cortiq_last_route` 给出回答所用的记录。无论 `cortiq_set_gpu` 如何设置，φ 都在 CPU 上计算：在 Xiaomi 12 Lite 上，逐操作 GPU 路径的首次仲裁耗时 235 秒，而 CPU 每个问题约 100 毫秒。

矩阵在设备上保持文件中的形式，即 f16 或 q8_2f（`gpu_wgpu/embryo_wide.wgsl` 配合 `embryo_wide_f16.wgsl` 或 `embryo_wide_q8.wgsl`）。一个 token 约需 80 次调度，两次调度之间的边界比大多数内核本身更耗时（M4 上 8 µs，RTX 2000 Ada 上 5 µs），而且 GPU 只有在大量短线程下才能满载。因此一个线程最多处理一行中的 128 个值，部分和由下一个归一化阶段相加；q8_2f 矩阵的列缩放在生成输入向量的阶段一次性乘入，权重码按整数解包；锚点按绝对位置旋转查询和键，所以环形缓冲中存放的是旋转后的键，打分就是普通点积；值按块累加，只取权重高于最大权重 e^-16 的键；delta 规则及其归一化在每个头共用一个工作组。输出头只读取对数概率高于 −9.21 的簇（`CMF_EMBRYO_HEAD_GATE`；`-1e30` 读取所有行，被跳过的簇的行返回 −1e30）。`CMF_EMBRYO_WIDE=0` 使用 f32 路径。
