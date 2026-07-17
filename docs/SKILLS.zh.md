# 技能：一个文件里的专家集群

*语言：[English](SKILLS.md) · [Русский](SKILLS.ru.md)*

一个 `.cmf` 文件可以携带一个共享主干（backbone）加任意数量的**技能**——
从真实微调检查点移植来的替换张量集合。运行时读取技能张量来*替代*主干张量
（从不相加，也从不组装第二个模型），因此切换专家是免费的：同一个 mmap、
同样的内存、相同地址上的不同权重。提示词可以自动路由到合适的技能——文件
自己知道哪位专家最匹配，无需训练门控，也无需外部路由器。

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
cortiq run swarm.cmf -p "..."                      # 主干
cortiq run swarm.cmf -p "..." --skill thinker      # 显式叠加
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

## 资源

| 项目 | 成本 |
|---|---|
| 供体下载 | 一个 HF 检查点（0.5B 约 1 GB bf16），缓存于 `~/.cache/cortiq/hf` |
| 烘焙时间 | 笔记本上几分钟：量化移植张量 + 重写文件 |
| 文件增长 | 仅移植的张量：q8 0.5B 上全层 FFN 技能 +314 MB（主干字节不变） |
| 烘焙内存 | ~主干大小 + 一个供体张量（供体经 mmap 流式读取） |
| 技能的运行时开销 | 不激活时为零；激活的技能 = 主干同速（同形状、同内核） |
| 路由开销 | 提示词到 φ 层的一次 prefill |

想要更小的体积，就限制层数（`--layers 12-23` 体积减半），并用
`--quality` 验证你关心的行为是否保留——诚实的权衡只差一个参数。

## 底层原理

- 技能张量是名为 `skill.{id}.{tensor}` 的普通目录条目——像其它张量一样
  mmap，首次触碰时换页加载。存储按 |主干| + Σ|增量| 增长
  （[规范 §9](CMF_V2_SPEC.zh.md)）。
- 运行时通过来源间接层解析每个张量：主干条目或
  `skill.{active}.{name}`——是替换而非相加，从不组装完整的技能模型。
- 路由是仿射子空间上的 recon-argmin：没有训练过的门控、没有分类器——
  选择是*文件本身的属性*。`route`、`explain`、`--blend`、
  `--route-dynamic` 读取的都是同一份约 4 KB 的描述符。
- 注册表的 `quality` 字段是诚实契约（claim 16）：测了什么、在什么上测、
  结果如何。`skill list` 会显示它。
