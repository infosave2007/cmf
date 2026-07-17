# 技能：一个文件里的专家集群

*语言：[English](SKILLS.md) · [Русский](SKILLS.ru.md)*

一个 `.cmf` 文件可以携带一个共享主干（backbone）加任意数量的**技能**——
从真实微调检查点移植来的替换张量集合。运行时读取技能张量来*替代*主干张量
（从不相加，也从不组装第二个模型），因此切换专家是免费的：同一个 mmap、
同样的内存、相同地址上的不同权重。提示词可以自动路由到合适的技能——文件
自己知道哪位专家最匹配，无需训练门控，也无需外部路由器。

## 术语——需要知道的三个词

- **主干（基座模型）**——用 `cortiq convert` 转换成 CMF 的普通 LLM。
  它是模型的共享权重，*不含*技能；技能存放在同一文件中，激活时替换其
  部分张量。
- **技能**——一组替换张量（"这些层的 FFN 用专家权重，其余来自主干"）
  加上约 4 KB 的路由描述符，让文件能识别"这个提示词属于我"。
- **供体（donor）**——*同一主干的微调检查点*（SFT 或已合并的 LoRA），
  技能可以从中现成提取。供体不是另一个模型（"编程模型"、"哲学模型"）：
  从 continued-pretrain 亲缘模型移植是行不通的，下文有实测。

## 技能从哪里来：两条路径

**路径 1——只有数据集，没有供体：`cortiq skill bake`。** 只要有主干和
领域文本（语料、基准任务），引擎就能自己找出相关神经元。这正是
imatrix 式"点亮"直觉的完整版：让语料流过模型，观察哪些 FFN 神经元承载
任务——但掩码不是按"或"累加，而是*训练*出来的（按 DTG-MA 做 L1
正则——USPTO 申请 19/452,464，[PATENTS.md](../PATENTS.md)），因为实测表明未经训练的 top-K 点亮会摧毁质量（PPL 59 对 11.9，
见下文）。训练出的掩码丢弃噪声神经元——模型在该领域先*变好*——然后
最后几层对精确教师做抛光（FCD）。全程不需要第二个模型。

**路径 2——现成供体：`cortiq skill add`。** 如果有人已经微调过你的
主干（text-to-SQL 微调、俄语 SFT），则完全无需训练："点亮"已经由供体
作者完成——他们的微调就是训练结果。剩下的是提取变化部分：每个供体张量
与主干比较，微调没碰过的丢弃（`--min-delta`），变化了的作为技能移植进
文件。笔记本上几分钟，不需要 GPU。

两条路径的终点相同：文件获得一个带 id、名称和路由描述符的技能，
`skill list` 能看到它，`--quality` 把实测结论写进注册表。

### 常见问题

**主干是包含掩码/技能的权重吗？** 不是。主干是共享模型权重；技能和
掩码是同一文件中的独立目录项，技能激活时运行时读取它的张量而非主干的。

**不训练怎么从供体里提取相关权重？** 训练已经发生了——就在供体里。
我们不在供体中寻找"技能神经元"，而是取*差异*：参考解码器测量每个张量
相对主干的相对变化，只保留微调改动过的张量。阈值即 `--min-delta`。

**没有供体怎么办？** 走路径 1：数据集 → `skill bake` → 训练掩码 +
抛光。这正是"跑 50 道基准题得到一个 swe-skill"的场景：把文本交给
`--files`，其余由引擎完成（下文有真实运行日志）。

**技能能在人与人之间分享吗？** 能。技能是 `.cmf` 内的目录项；可以
整个文件连同集群一起交付，而在同一主干的文件间搬移单个技能是目录操作
（无需重算权重）。skill-hub 正是这个设计指向的方向。

以下所有内容都可以用三条命令在 Hugging Face 的真实检查点上复现；提示词与
评测文件在 [`docs/skills-demo/`](skills-demo/)。测量环境：`cortiq` 0.3.4+，
主干为 Qwen2.5-0.5B-Instruct（q8）。

## 演示：一个文件，四种人格

```sh
# 1. 主干
cortiq convert --model Qwen/Qwen2.5-0.5B-Instruct --quant q8 --output swarm.cmf

# 2. 三个技能，移植自同一基座的真实微调
cortiq skill add swarm.cmf --from vindows/qwen2.5-0.5b-text-to-sql-merged \
  --id sql --name "SQL assistant" --prompts docs/skills-demo/prompts-sql.txt

cortiq skill add swarm.cmf --from Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct \
  --id ru --name "Русский ассистент" --prompts docs/skills-demo/prompts-ru.txt \
  --quality docs/skills-demo/eval-ru.txt

cortiq skill add swarm.cmf --from ewre324/ewre324-Thinker-Qwen2.5-0.5B-Instruct-Reasoning \
  --id thinker --name "Step-by-step verifier" --prompts docs/skills-demo/prompts-think.txt
```

`skill list` 显示文件现在携带的内容：

```
3 skill(s):
  sql        SQL assistant            72 tensor(s), 314.3 MB, layers [0..23], routable
  ru         Русский ассистент        72 tensor(s), 314.3 MB, layers [0..23], routable
      quality: {"backbone":11.875,"overlaid":11.027,"metric":"ppl","file":"eval-ru.txt"}
  thinker    Step-by-step verifier    72 tensor(s), 314.3 MB, layers [0..23], routable
```

一个 1.45 GB 的文件，替代四个各 479 MB 的独立模型（1.9 GB）；一个
mmap，人格切换在加载时零开销——技能张量就是普通的目录条目，运行时
指向它们而不是主干条目。

## 可测量的提升

**路由：6/6，全部是路由器从未见过的提示词。** 文件把新鲜提示词路由到
正确的专家，误差余量 3–5 倍（`E = ‖r−BBᵀr‖²/‖φ‖²`，越低越接近）：

```
$ cortiq route swarm.cmf -p "Show me the SQL to find orders with no matching invoice."
  sql      E = 0.0016   ← 胜者
  thinker  E = 0.0073
  ru       E = 0.0112

$ cortiq route swarm.cmf -p "Расскажи, как приготовить сырники из творога."
  ru       E = 0.0050   ← 胜者
  thinker  E = 0.0168
  sql      E = 0.0204

$ cortiq route swarm.cmf -p "Verify step by step whether 91 is divisible by 7 and by 13."
  thinker  E = 0.0047   ← 胜者
  sql      E = 0.0165
  ru       E = 0.0190
```

**`ru`：俄语散文困惑度 −7.1%**，在烘焙时记录进文件自己的注册表
（留出俄语文本上主干 11.88 → 叠加后 11.03）。生成质量的差异肉眼可见：
主干输出循环矛盾的要点列表，技能写出连贯有结构的散文。

**`thinker`：验证而不是断言。** 问 *"Is 17077 a prime number?"*——主干
自信地编造出错误的因数分解（`17077 = 3 × 5699`，这是错的：17077 是
素数）；技能则开始系统地检查整除性。同一个文件，`--skill thinker`。

**`sql`：text-to-SQL 结构。** 对模式问题，技能用窗口函数和 CTE 作答，
而主干只写平铺的 join。

关于指标，说句实话：困惑度适合衡量*改变领域分布*的技能（比如 `ru` 之于
俄语文本）。对指令风格的 SFT 供体（`sql`、`thinker`），收益在*生成行为*
而非文本似然——请用 A/B 生成或任务准确率来衡量它们，不要惊讶于手写语料
上的 PPL 持平甚至略差。`--quality` 把你实际测到的东西记进注册表；它绝不
会默默给技能背书。

## 使用技能

```sh
cortiq run swarm.cmf -p "..."                      # 自动路由：文件自己挑选专家
cortiq run swarm.cmf -p "..." --skill thinker      # 显式固定某个技能
cortiq run swarm.cmf -p "..." --skill none         # 强制主干
cortiq route swarm.cmf -p "..."                    # 为所有技能打分
cortiq explain swarm.cmf --prompt "..."            # 路由 + 首词元分布
cortiq run swarm.cmf -p "..." --blend auto         # 前 2 名的软叠加
cortiq run swarm.cmf -p "..." --route-dynamic --trace   # 逐词元切换
```

`--blend auto` 以 softmax(−E/T) 权重混合前 2 个技能——一条克罗地亚语的
算术验证提示词恰好落在 `ru` 与 `thinker` 之间，约 50/50。
`--route-dynamic` 随上下文演化*在流中*切换活跃技能；配合 `--trace` 可以
看到全程（一条俄语+SQL 的混合提示词恰在生成 `… WHERE o.id IS NULL` 时
从 `thinker → sql` 切换，随后回到主干做俄语解释）。切换阈值有滞回保护；
若你的技能彼此靠得更近，用 `CMF_ROUTE_EON` / `CMF_ROUTE_EOFF`
（激活 / 放弃的 E）调节。

带数字的真实逐词元轨迹（`--trace`，混合提示词"Найди в базе всех
клиентов без заказов: напиши SQL-запрос и объясни его по-русски"，
`CMF_ROUTE_EON=0.9 CMF_ROUTE_EOFF=1.0`）：

```
  #  词元         技能        E
  0  Вот          —           —       开头几个词元：路由器累积 φ-EMA
  7  ,            —           0.857   E < EON(0.9) → 切入
  8  который      thinker     0.857   thinker 激活
 23  sql          thinker     0.662   E 谷底——SQL 代码块
 31  c            thinker     0.834   俄语解释：E 上升……
 47  ет           thinker     0.865   ……但 E < EOFF(1.0)——滞回保持
```

把这些数字当作裕量来读：切入（0.857 对阈值 0.9）与退出（0.865 对
1.0）之间还有约 0.13 的空间——不会抖动。默认阈值（EON 0.62 /
EOFF 0.74）偏保守：这个集群的 E 谷底是 0.662，未调参的文件会诚实地
留在主干上。当你的技能 φ 统计离实际提示词更远时，调高这两个阈值。

## 创建技能：真正重要的事

`cortiq skill add` 把**供体**检查点的张量移植进主干容器：

```sh
cortiq skill add <backbone.cmf> \
  --from <hf仓库或本地目录>       # 供体，同一架构
  --id <id> [--name "..."]        # 注册表身份
  --layers all|A-B|i,j,k          # 哪些层（默认全部）
  --tensors ffn|attn|all          # 哪些族（默认 ffn）
  --prompts file.txt              # 8+ 条示例提示词 → 路由子空间
  --quality held-out.txt          # PPL 门槛，记录进注册表
  --min-delta 0.02                # 丢弃微调没碰过的张量
  --skill-quant vbit --mean-bits 6  # 叠加层用更便宜的编码
  --output out.cmf                # 默认：原地重写
```

供体张量用主干自身的逐张量编码重新量化（q8 主干得到 q8 技能）；移植是
逐位忠实的：用主干*自己的*源检查点重新烘焙一个技能，PPL 变化恰好 +0.0%。

**选主干自身的微调作为供体。** 这是决定成败的唯一规则。FFN 张量与其
周围的注意力权重是协同适应的：

- **同一基座的 SFT / 合并 LoRA——可行。** 三个演示供体都是
  Qwen2.5-0.5B-Instruct 的微调；它们的 FFN 移植连贯且行为各异。
- **继续预训练的近亲——作为部分移植不可行。** 我们替你量过了：从
  Qwen2.5-**Coder**-0.5B-Instruct（形状相同、同一家族，但经过深度代码
  继续预训练的兄弟模型）移植 FFN（甚至 FFN+attention），得到 PPL 238，
  而主干是 2.6。它的权重与技能不携带的归一化层和词嵌入协同适应了。
  供体偏离到这种程度，你需要的是独立的模型文件，而不是技能。

**路由提示词**（`--prompts`）：8 条*听起来像该技能用户*的短提示词。
由它们拟合 φ(x) 统计量（2/3 深度处平均池化隐状态的均值 + 秩 2 PCA
基），存入文件头——每个技能约 4 KB，此后文件就能自己路由。替换非 FFN
张量的技能被排除在*动态*路由之外（静态 `--skill` 仍可用）——运行时会
就此告警。

## 资源：需要什么，在什么硬件上

**路径 2（`skill add`）很轻。** 完全不需要 GPU：

| 项目 | 成本 |
|---|---|
| 供体下载 | 一个 HF 检查点（0.5B 约 1 GB bf16），缓存于 `~/.cache/cortiq/hf` |
| 时间 | 笔记本上几分钟：量化移植张量 + 重写文件 |
| 内存 | ~主干大小 + 一个供体张量（供体经 mmap 流式读取） |
| 文件增长 | 仅移植的张量：q8 0.5B 上全层 FFN 技能 +314 MB（主干字节不变） |
| 技能的运行时开销 | 不激活时为零；激活的技能 = 主干同速（同形状、同内核） |
| 路由开销 | 提示词到 φ 层的一次 prefill |

**路径 1（`skill bake`）是 CPU 训练。** 烘焙持有模型的 f32 副本
（每参数 4 字节）加上仅限抛光层的优化器状态，因此上限由普通内存决定：

| 主干 | 烘焙内存（估算：4 B/参数 + 优化器） | 时间（M4，240+120 步） |
|---|---:|---:|
| 0.5B | ~3 GB | 8.8 分钟（实测） |
| 1–2B | ~6–10 GB | ~20–40 分钟 |
| 4B | ~18 GB | ~1.5 小时 |
| 7–8B | ~35 GB | ~3 小时 |
| 27B | ~115 GB——128 GB+ 的 Mac Studio，或在 CUDA 上用 torch 配方 | 一夜 |

语料：100–200 KB 领域文本即可（上文真实运行用了 82 个 256-token 块）。
要做"基准技能"，把题目和参考答案收集成文本文件传给 `--files`。

## 一条命令，不用 Python：`cortiq skill bake`

整个 DTG-MA 配方原生运行——掩码训练、FCD 精修和 defrag 烘焙，一条命令在
CPU 上完成（训练 GEMM 走与 prefill 相同的 Accelerate 路径；注意力被冻结，
反向传播只沿 FFN 链行进）：

```sh
cortiq skill bake backbone.cmf \
  --files docs/CMF_V2_SPEC.ru.md README.ru.md docs/COMPARISON.ru.md \
  --output rutech-specialist.cmf
```

一次真实的逐步运行——Qwen2.5-0.5B-Instruct（q8），语料就是本仓库的俄语
文档，Apple M4，**端到端 8.8 分钟**：

```
bake: 70 calib + 12 held chunks of 256 tokens | FCD last 4 layer(s)
baseline (full): 24.157
  [A] step  30: L1=0.015 pruned= 0% hard-PPL=23.648 (bottom 23.648@0%)
  [A] step  60: L1=0.020 pruned= 2% hard-PPL=21.110 (bottom 21.110@2%)   <- 去噪谷底
  [A] step  90: L1=0.025 pruned= 6% hard-PPL=22.778 (bottom 21.110@2%)
  [A] step 120: L1=0.030 pruned=10% hard-PPL=25.610 (bottom 21.110@2%)
  [A] step 180: L1=0.040 pruned=16% hard-PPL=49.659 (bottom 21.110@2%)   <- 过了谷底质量崩塌
[A] 314s: masked-PPL 21.110                                              <- 恢复谷底检查点
  [B] step  30: held-PPL 18.304
  [B] step  60: held-PPL 17.840
  [B] step  90: held-PPL 17.474
  [B] step 120: held-PPL 17.423                                          <- FCD 继续向下
=== bake: baseline 24.157 | mask 21.110 | mask+FCD 17.423 | pruned 2% -> SPECIALIST <= baseline
runtime gate (held-out, real engine): backbone 24.173 -> specialist 19.039 (-21.2%)
```

三个值得读两遍的结论：

- 训练副本与真实引擎一致（baseline 24.157 对 24.173）——烘焙优化的就是
  运行时供给的。17.4（f32 副本）与 19.0（写出的文件）之间的差距是训练后
  FFN 的 q8 再量化——已测量，不隐藏。
- **泛化**：在语料中从未出现的俄语技术文档（`PERFORMANCE_ROADMAP.ru.md`）
  上，专家 22.56 对主干 25.62——**未见文本上 −12.0%**。
- 此处去噪谷底落在 2%（主干 PPL 24——这个领域主干并不算弱），所以体积
  收益小（479 → 472 MB）；故事在质量。在弱领域（主干 PPL 70）同一配方
  剪枝 11% 并削掉四分之一困惑度——见下一节。

## 比原始模型更小——而且更好：DTG-MA 烘焙

技能最强的形态不是伴随主干，而是在一个领域里*取代*它。DTG-MA 配方
（USPTO 申请 19/452,464，法律背景见 [PATENTS.md](../PATENTS.md)）
在你的任务语料上训练 FFN 神经元的 L1 正则掩码，捕捉*去噪谷底*
（先剪掉噪声神经元会让模型先变好，然后才开始变差），用 FCD 对末尾几层
做精修，最后 `--defrag` 烘焙出一个独立文件——被剪掉的神经元物理上不存在，
既不存储也不计算（申请 19/452,464 的 claims 9/10）：

```sh
# 原生，一条命令（见上一节）：
cortiq skill bake backbone.cmf --files corpus1.txt corpus2.txt --output specialist.cmf

# 或原始 torch 配方（相同两阶段；在 CUDA 机器上有用）：
python3 converter/make_skill_l1fcd.py --model <hf_snapshot_dir>   --id ru --files corpus1.txt corpus2.txt --out skill-ru
cortiq convert --model <hf_snapshot_dir> --defrag skill-ru   --quant q8_2f --output ru-specialist.cmf
```

用本仓库的运行时在 Qwen3.5-0.8B 上端到端实测，评测集是配方从未见过的
留出俄语技术文档：

| | 体积 | PPL（俄语技术，留出） | 解码 |
|---|---:|---:|---:|
| 原始检查点（bf16） | 1.6 GB | — | — |
| CMF q8_2f 基线 | 733 MB | 13.97 | 86.0 tok/s |
| **ru 专家（掩码 11% + FCD，defrag）** | **705 MB** | **11.92（−14.7%）** | **89.7 tok/s** |

比原始 LLM 小 2.3 倍、在领域上更好、而且更快——一次烘焙全部拿到。配方
自己在其领域内留出集上的报告走得更远：掩码谷底 70.5 → 54.2 PPL
（剪枝 11%），FCD 后 −38.7%，在独立的未见过技术语料上 −19.4%。

两条来自实测的诚实规则：

- **烘焙在主干薄弱处最亮眼。** 俄语技术文本（主干 PPL 70）给出完整
  效果；同一配方用在主干本就擅长的领域（代码，PPL 8.7）反而*更差*。
  先测主干在该领域的 PPL。
- **去噪谷底真实且狭窄。** 训练过程中掩码的留出 PPL：0% 时 67.9 →
  **11% 时 54.2（谷底）** → 18% 时 77.1。配方会自动停在谷底；不要为了
  体积硬推过去。

`cortiq skill add --sparse <keep>` 提供掩码的*免训练*快速近似（按任务
激活质量做逐层 top-K）——适合探路，但诚实实测：8 条提示词的校准下
keep 0.5 时崩溃（PPL 59 对 11.9）。训练出的掩码才是真方法；这个参数
只是侦察兵。

## 缩小技能的磁盘占用

技能的成本应当等于微调实际改变的部分——没被碰过的神经元不值得占用字节。
两个参数控制这一点，都经过实测，都由 `--quality` 把关：

**`--min-delta <x>`——不存储微调没碰过的东西。** 每个候选张量都会经由
参考解码器与主干比较；相对变化低于阈值的张量被丢弃——运行时在那里读主干
条目，而供体里放的本来就是同样的东西。门控会打印完整的增量分布，先看清
再下刀：

```
# ru 供体，实测：
delta gate ≥ 0.0000001: kept 72 / dropped 0 unchanged tensor(s) (−0.0 MB);
rel-delta min 0.0980 / median 0.2196 / max 0.3724
```

请用主干的*量化噪声底*来校准阈值：用主干自己的源检查点做一次自体移植
（`--min-delta 0.0000001`），读打印出的中位数——那就是纯量化误差
（我们的 q8 主干实测中位数 0.92%；自体移植配 `--min-delta 0.02` 会正确
丢弃全部 72 个张量并拒绝烘焙）。这个量级的增量是噪声而非信号。三个演示
供体处处都在它之上（sql 0.9–2.6%、thinker 1.8–3.5%、ru 9.8–37%），所以
门控正确地全部保留——它在微调局部化时才有收益（只合并进几个投影的
LoRA、只动过深层的微调），而且从不撒谎：先测量，后裁剪。

**`--skill-quant <enc> [--mean-bits N]`——给叠加层用更便宜的编码。**
逐张量 dtype 是 CMF 的原生能力（规范 §3），技能可以比主干住得更省。
在 q8 主干上的 `ru` 技能实测（留出俄语散文，主干 PPL 11.88）：

| 编码 | 体积 | 技能 PPL | 结论 |
|---|---:|---:|---|
| q8（与主干相同） | 314 MB | 11.03（−7.1%） | 完整质量 |
| `--skill-quant vbit --mean-bits 6` | 257 MB（−18%） | 11.19（−5.8%） | 保住大部分收益 |
| `--skill-quant vbit`（4.25 位） | 184 MB（−41%） | 12.17（+2.5%） | **比主干还差** |
| `--skill-quant q4` | 176 MB（−44%） | 12.20（+2.7%） | **比主干还差** |

教训与量化一贯的教训相同，只是下沉了一层：*微调的信号*必须扛得住新增
的噪声。0.5B 的 SFT 增量（此处相对 10–37%）能撑过约 6 位，在 4 位时
淹没。缩小之后务必用 `--quality` 重测——无论结果如何，数字都会写进注册
表，文件自带裁决。

也可以手动限制层数（`--layers 12-23` 体积减半）；规则相同——验证你
关心的行为是否保留。

## 技能的"护照"——交接时要附带什么

注册表已经携带了诚实交换所需的几乎一切；如果你要发布技能（或为将来的
skill-hub 做准备），最小护照如下：

| 字段 | 来源 | 用途 |
|---|---|---|
| `id`、`name` | `skill add --id/--name` | 注册表身份 |
| 来源 | 供体仓库+修订号，或烘焙语料文件列表 | 可复现性 |
| 供体许可证 | HF 检查点的卡片 | 再分发权重差量的权利 |
| `quality` | `--quality held-out.txt`——写入注册表 | 实测结论：主干 → 叠加后，在什么上测 |
| `--min-delta` 阈值 | 烘焙命令 | 有多少张量被当作量化噪声丢弃 |
| 编码 | `--skill-quant`/`--mean-bits` | 读者可判断信号是否扛过压缩 |
| 路由提示词 | `--prompts file.txt` | φ 描述符：文件如何认出"自己的"提示词 |

除许可证和来源外，其余都随文件本身携带（`skill list` 可见）；这两项
请写在旁边的说明里。在同一主干的文件间搬移技能是目录操作——无需重算
权重。

## 底层原理

- 技能张量是名为 `skill.{id}.{tensor}` 的普通目录条目——像其它张量一样
  mmap，首次触碰时换页加载。存储按 |主干| + Σ|增量| 增长
  （[规范 §9](CMF_V2_SPEC.zh.md)）。
- 运行时通过来源间接层解析每个张量：主干条目或
  `skill.{active}.{name}`——是替换而非相加，从不组装完整的技能模型。
- 路由是仿射子空间上的 recon-argmin：没有训练过的门控、没有分类器——
  选择是*文件本身的属性*。`route`、`explain`、`--blend`、
  `--route-dynamic` 读取的都是同一份约 4 KB 的描述符。
- 注册表的 `quality` 字段是诚实契约（claim 16，[PATENTS.md](../PATENTS.md)）：测了什么、在什么上测、
  结果如何。`skill list` 会显示它。
