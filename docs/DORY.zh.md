# Nemotron-Labs Dory 与 cortiq

*语言:[English](DORY.md) · [Русский](DORY.ru.md) · 中文*

[Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) 是 NVIDIA 的循环 Transformer:
16 个输入子层、40 个循环子层、16 个输出子层,共 40 亿参数。循环块在同一份权重上运行多次,因此同一个权重文件
无需重新训练即可跑 1、2 或 4 次循环。模型在 `<think>` 中推理,并能调用工具。

## 运行

```bash
cargo install cortiq-cli
hf download infosave/Nemotron-Labs-Dory1.0-4B-cmf dory-turbo.cmf --local-dir .
cortiq run dory-turbo.cmf --prompt "写一个反转列表的 Rust 函数。" --no-think
```

| 文件 | 权重 | 大小 | wikitext-2 ppl | 用途 |
|---|---|---:|---:|---|
| `dory-turbo.cmf` | 4 位 GPTQ | 2.09 GB | 14.22 | 与 `q4tp-v2` 相同的权重,加上路由器和远场;默认 |
| `dory-q4tp-v2.cmf` | 4 位 GPTQ | 2.09 GB | 13.96 | 每个 token 两次循环 |
| `dory-q8_2f.cmf` | 8 位 | 4.01 GB | 13.44 | 最接近原始模型 |

困惑度:wikitext-2 test 的 12 个 512-token 窗口,逐 token 计算。bf16 原始模型为 13.42。

## 循环模式

| 参数 | 含义 |
|---|---|
| `--loops N` | 固定循环次数,1 到 16;在 Turbo 文件上同时关闭路由器、远场等功能 |
| `--loops auto` | 每个 token 一次或两次:使用文件的路由器,文件没有路由器时使用最高概率 |
| `--kv-cache per_loop\|last_loop` | 每次循环一个缓存,或全部共用一个 |
| `--o1 m,w,sink` | 全注意力层上的远场,在 8192 行处封存 |
| `--full-window N` | 改为把全注意力层截断到窗口 |

`CMF_DORY_O1=m,w,sink,seal` 同时设置封存位置,`CMF_DORY_O1=off` 关闭 Turbo 文件的远场。`CMF_DORY_GATE=conf:0.7`
或 `recon:FILE:bias` 选择门控。`CMF_DORY_PROMPT_LOOPS=N` 设置提示词的循环次数,`CMF_DORY_EXACT=FILE:rows` 设置精确
行的选择,`CMF_DORY_EXACT_LOG=FILE` 写出每行的得分。

## Turbo 文件做了什么

Turbo 文件保存相同的权重,文件头开启五项功能。

![一次或两次循环](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/router_zh.png)

最后一次循环缓存。每个循环注意力层只保留一个缓存,每次循环覆盖本 token 的行。循环次数可以逐 token 变化,不留空洞。

共振路由器。第一次循环后,由两个相互隔离、秩为 16 的 PCA 子空间重建状态。它们分别拟合在第二次循环改变了答案的
token 和没有改变的 token 上。误差较小的一方决定 token 是否离开循环。没有梯度训练,决策也不需要经过输出块和输出头。
约 85% 的 token 只走一次循环。

远场。全注意力层精确保留最前几行和最后 1536 行,其余全部压缩到每个头 32 个地标(`nystrom`()。滑动层本来就只有 1024
行窗口。设备内存不再增长,每个 token 的代价也不再增长。

![长提示词在显卡上保留什么](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/far_field_zh.png)

精确行。只靠远场会丢失密码:精确模型能在 100 000 字符的文本中找到它,远场不能。因此远场会丢失的行保持精确。对每个
全注意力层的每个 KV 头,每行计算两个数:键离地标张成空间有多远,以及值的范数。它们各自变成提示词内的排名,再由 72
个排名的线性拟合给出该行得分。在提示词最后 512 行之前,得分最高的 512 行及其相邻行离开远场,远场不含它们重新构建。
这个顺序很重要:问题会把在文本中找到的内容带到回答中,所以它自己必须读到精确的事实。密码的行在数万行中排在最前面。

![密码如何保持精确](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/exact_rows_zh.png)

单次循环读取提示词。在最后一次循环缓存下,提示词的行只留下缓存行,没有人读取它们的 logits。除最后一块外都只走一次
循环,代价是后续内容的困惑度上升 1%。

## 数字

![普通版与 Turbo](https://huggingface.co/infosave/Nemotron-Labs-Dory1.0-4B-cmf/resolve/main/assets/compare_zh.png)

解码速度,单位 tok/s。每组第一行是 12 个真实提问,每个 200 token,通过 `cortiq run` 运行;其余是相应长度提示词之后的
`cortiq bench --core`。

| | `dory-q4tp-v2` | `dory-turbo` |
|---|---:|---:|
| Vulkan,NVIDIA L4((24 GB) | | |
| 真实提问 | 49.6 | 60.7 |
| 上下文 1 000 | 41.1 | 60.6 |
| 上下文 4 000 | 35.1 | 51.8 |
| 上下文 16 000 | 21.5 | 52.2 |
| 上下文 32 000 | 14.1 | 54.0 |
| 上下文 64 000 | 25 分钟未读完提示词 | 53.0 |
| 首 token,16 000 / 32 000 / 64 000 | 78 秒 / 360 秒 / 超过 25 分钟 | 53 秒 / 112 秒 / 228 秒 |
| 设备 KV 内存,4 000 / 32 000 | 1.0 GB / 4.3 GB | 0.64 GB / 0.50 GB |
| Metal,MacBook Air M4((24 GB) | | |
| 真实提问 | 25.7 | 32.9 |
| 上下文 1 000 | 22.1 | 34.6 |
| 上下文 4 000 | 20.3 | 31.0 |
| 上下文 16 000 | 14.8 | 29.3 |
| 首 token,16 000 | 211 秒 | 134 秒 |
| CPU | | |
| M4,上下文 256 | 18.0 | 27.7 |
| M4,上下文 2 000 | 12.1 | 18.3 |
| EPYC 9254,24 核,上下文 256 | 7.6 | 11.6 |
| EPYC 9254,24 核,上下文 4 000 | 5.3 | 8.2 |

约 85% 的 token 只走一次循环,维基百科文本和代码回答都是如此。`dory-q8_2f.cmf` 的真实提问速度:Vulkan
32.0 tok/s,Metal 15.4,M4 CPU 13.8。长窗口,两个 16384 token 窗口的困惑度:精确 8.10,Turbo 的近场加
远场 8.85,截断到 2048 行 11.03。

## 构建 Turbo 文件

路由器拟合在模型自己的回答上。脚本在 `scripts/dory/router`,参数写在每个脚本的开头:

| 脚本 | 作用 |
|---|---|
| `gen_samples.py` | 模型对混合提问集的回答 |
| `loop_probe.py` | 教师强制下 1、2、4 次循环的 logits 和循环后的状态 |
| `router_fit.py` | 在一半回答上拟合两个拓扑,在另一半上算 AUC |
| `make_bin.py` | 把拓扑写成平面文件,并给出几种单次循环比例对应的阈值 |
| `illum_feat.py` | 带植入事实的提示词,每行的残差和值范数 |
| `illum_budget.py` | 拟合、在未见提示词上的覆盖率、写入文件的权重 |

```bash
cortiq dory-turbo dory-q4tp-v2.cmf dory-turbo.cmf --topologies phi1.bin --bias -0.000258 \
    --o1 32,1536,4,8192 --exact-weights keep.bin --exact-rows 512
```

阈值越高,走两次循环的 token 越多。窗口、sink 和精确行总数不超过 2052 时,封存后的提示词块使用批处理内核;超过时逐行处理。

不覆盖的情况。提示词结束后到达的行,只能留在窗口内或进入远场。与已封存对话不一致的一轮会重新读取整个提示词并重新选择。

## 转换

```bash
hf download nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 --local-dir bf16
cortiq convert --model bf16 --quant f16 --output dory-f16.cmf --also-output q8_2f=dory-q8_2f.cmf
cortiq quantize-gptq dory-f16.cmf --codec q4tp --calib-model dory-q8_2f.cmf \
    --calib wiki_train.txt=0.6 --calib data_chat.txt=0.25 --calib data_code.txt=0.15 \
    --tokens 32768 --window 2048 --lambda 0.01 --output dory-q4tp-v2.cmf
```

`scripts/dory/` 中有参考实现、一致性和测量脚本,`JOURNAL_DORY_2026-11-04.md` 记录了尝试过的内容。

## 一致性

原始提示词上的贪心文本与 f32 参考完全相同。与 vLLM 在 NVIDIA 权重上对比:64 个 token 中 64 个相同,对数概率最大差 3e-4,
包括 1500 token 的提示词。q4tp 在 Vulkan 上与 CPU 相比:8.9e-4;Metal 与 CPU 相比:7.7e-3。注意力内核在 Vulkan 和 Metal 上
把困惑度保持到三位数字。

## 许可证

权重:[OpenMDW-1.1](https://openmdw.ai/license/1-1/)。路由器和远场受美国专利申请 No. 19/452,440 和 No. 19/738,763 保护;
见 [PATENTS.md](../PATENTS.md)。
