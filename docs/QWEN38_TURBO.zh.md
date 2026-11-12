# cortiq 上的 Qwen3.8-27B Turbo

*语言:[English](QWEN38_TURBO.md) · [Русский](QWEN38_TURBO.ru.md) · **中文***

[Turbo 文件](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf)包含 Qwen3.8-27B 的 4 位权重,文件头让长提示下的显存保持不变。短于 8192 个 token 的提示使用精确注意力,与[普通文件](https://huggingface.co/infosave/Qwen3.8-27B-cmf)相同。

## 运行

```bash
cargo install cortiq-cli
hf download infosave/Qwen3.8-27B-Turbo-cmf qwen38-27b-turbo.cmf --local-dir .
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think
cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think --o1-spill /mnt/ssd/cortiq
CMF_GPU=cuda cortiq run qwen38-27b-turbo.cmf --prompt-file report.txt --greedy --no-think   # 用 --features cuda 构建
```

| 参数 | 含义 |
|---|---|
| `--o1-spill DIR` | 把长提示的行放在 `DIR` 的文件中,也可用 `CMF_O1_SPILL` |
| `--prompt-file FILE` | 比命令行能容纳的更长的提示 |
| `CMF_O1_PREFIX_MAX=N` | 在 `N` 行封存,不按显卡选择 |
| `CMF_O1=off` | 普通模型,任何长度都用精确注意力 |
| `CMF_O1_SPEC=0` | 远场上不用草稿 |
| `CMF_O1_STAGES=0` | 远场注意力使用每个头一条长线程 |
| `CMF_WGPU_BLOCK_MB=N` | Vulkan 分配器在小缓冲区之间共享的块,默认 16 MiB |

## 长提示的处理过程

![小显卡上的长提示](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/ceiling_zh.png)

1. 前面的行在批处理图上精确读取:8192 行;显卡预算放不下时为 4096 行(`o1_seal_for_card`)。
2. 后面的块读取由已读各行构建的远场。它们的查询、键和值通过主机可见的缓冲区离开显卡,进入内存或文件(`O1Archive`)。
3. 在最后 512 行之前,从所有保存的行中选出精确行,并用这些行逐层重建远场,所以内存中不会同时存放两层的行。
4. 最后的行和每个解码 token 读取汇点、精确行、1800 行窗口和每个头 32 个地标。状态大小固定。

第 3 步的顺序很重要:问题把在文本中找到的内容带到答案中,所以它必须亲自读到精确的事实。如果在问题的行之后才选择,虽然能找到事实所在的行,但只有八分之一的回答正确。

**精确行。** 每个完整层的每个 KV 头,每行两个数:键与地标张成空间的距离,以及值的范数。两者都转为提示内的排名,再由 128 个排名的线性拟合给出分数,不使用词表。在拟合未见过的提示上,插入事实所在的行平均位于前 0.22%,最差也在前 0.96%。

**短线程。** NVIDIA 显卡的一个工作组停留在一个 SM 上,所以每个头一个 256 线程的工作组会让显卡闲置。`o1_stages` 把运算拆开:每个键一条线程计算近场分数、地标分数、最大分数、每个头的远场部分、权重、分段计算值的和,再把各段相加。封存后的 token 从 31.5 提升到 56 tok/s,验证五行从 57 毫秒降到 41 毫秒。

**草稿。** 验证块计算注意力时不移动远场,它的键和值留在显卡上等待。然后 `o1_spec_commit` 只把状态前移到被接受的行。草稿头在提示的最后几行上预热。贪婪解码时,有无草稿的文本相同。

**显存。** 33 到 64 MiB 的权重以前各占一个 64 MiB 的块,所以 12.6 GB 权重占用了 16.6 GB 显存。16 MiB 的块为 Vulkan 上的所有模型收回了这部分空间,速度不变。

## 数字

![普通文件与 turbo](https://huggingface.co/infosave/Qwen3.8-27B-Turbo-cmf/resolve/main/assets/compare_zh.png)

RTX 5090,Vulkan,`cortiq bench --core`,在该长度的提示之后生成 60 个 token。

| 提示,token 数 | 首个 token,秒(普通) | 首个 token,秒(**turbo**) | 解码,tok/s(普通) | 解码,tok/s(**turbo**) | 显存峰值,GB(普通) | 显存峰值,GB(**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 2.2 | **2.2** | 105 | **105** | 14.5 | **14.4** |
| 4 000 | 8.8 | **8.8** | 104 | **104** | 14.9 | **14.8** |
| 16 000 | 42 | **47** | 87 | **92** | 16.5 | **15.5** |
| 32 000 | 107 | **92** | 71 | **91** | 18.5 | **15.7** |
| 64 000 | 316 | **177** | 51 | **92** | 22.8 | **15.7** |

真实回答:在 100 000 字符的维基百科之后生成 400 个 token 的摘要,首个 token 从 76.7 秒到 72.1 秒,解码从 50.9 到 60.4 tok/s。在同一张 5090 上设置 16 GB 显卡的预算(`CMF_GPU_VRAM_MB=15000`):15.0 GB,64 000 时首个 token 176 秒,解码 92 tok/s。

真正的 16 GB 显卡 RTX 2000 Ada 在短提示下占用 14.2 GB。`cortiq run --greedy --no-think`,回答 220 个 token:

| 回答 | 没有草稿,tok/s | 有草稿,tok/s |
|---|---:|---:|
| 一篇文章 | 13.3 | **20.5** |
| 俄语的解释 | 13.4 | **21.0** |
| 带测试的 Rust 函数 | 13.4 | **24.4** |
| 引用提示中的句子 | 13.1 | **25.1** |
| 修改提示中的代码 | 13.1 | **28.4** |

在对话中,草稿链的深度按该显卡自身的轮次耗时调整:同一进程的五轮对话里,第二到第五轮为 29.8 tok/s。

读取提示时的主机内存,不含内存映射的权重:

| 提示,token 数 | 内存 | 使用 `--o1-spill`:内存 | 使用 `--o1-spill`:磁盘 |
|---:|---:|---:|---:|
| 16 000 | 4.4 GB | 2.2 GB | 4.7 GB |
| 32 000 | 7.0 GB | 2.2 GB | 8.7 GB |
| 64 000 | 13.2 GB | 3.0 GB | 18.7 GB |

在写入 4.5 GB/s、读取 5.2 GB/s 的磁盘上,速度与使用内存相同。

MacBook Air M4,24 GB,Metal,16 000 个 token 时:除内存映射的权重外 6.3 GB,首个 token 545 秒,解码 4 tok/s。其余见[在 Mac 上](#在-mac-上)。

在 100 000 字符中插入事实:12/12,其中 8 次使用 15 GB 预算和 SSD;在 230 000 字符(约 53 000 token)中同样路径 6/6。没有精确行的远场:0/4。封存点之后各行的困惑度,两个 16 384 token 的维基百科窗口:精确注意力 4.76,远场 4.97。

## 在 Mac 上

测量机器是 MacBook Air M4,24 GB:没有风扇,长时间运行会发热降速;内存总线是基础版 M4 的,Apple 标称 120 GB/s。它读取每个 token 所需的 13.6 GB 权重,速度为 92 到 95 GB/s,这就是这条总线的速度:没有草稿时 6.4 tok/s。M4 Pro 的总线标称 273 GB/s,M4 Max 最高 546 GB/s;这些机器没有测量。其余来自草稿和 Neural Engine：在 q4tp 文件上草稿默认开启（`CMF_GRAPH_SPEC=0` 关闭；cortiq 0.8.63 及更早版本需要 `CMF_GRAPH_SPEC=1`），Neural Engine 是一个默认关闭的开关。

**草稿**。草稿头提出 token,对权重的一次遍历检查它们,于是一次遍历给出不止一个 token。MacBook Air M4,贪心解码,220 个 token,有草稿和没有草稿的文本相同:

| 回答 | 没有草稿,tok/s | 有草稿,tok/s |
|---|---:|---:|
| 一篇文章 | 6.4 | **10.1** |
| 俄语的解释 | 6.4 | **10.7** |
| 带测试的 Rust 函数 | 6.4 | **12.1** |
| 引用提示中的句子 | 6.4 | **15.0** |
| 修改提示中的代码 | 6.4 | **20.5** |

在 3 600 个 token 的提示之后:6.0 → 8.3 tok/s。这来自三件事。两行或三行的检查走批量 matvec,它读取权重与单行一样快:95.7 和 82.4 GB/s,而 GEMM 是 60 到 67,其 logits 与普通路径在小数点后四位内相同。链的深度跟随每个草稿成立的频率:多数文本上是两个草稿,而固定的七个在文章上没有任何收益。当回答重复提示或自身的一段时,那一段后面的 token 把链延长到七个草稿(`CMF_SPEC_LOOKUP=0` 关闭)。封存点之后草稿同样划算：OrcaCyber-Nano-1.0（Qwen3.8-27B 的衍生模型，形状相同）用 Neural Engine 读取 16 000 个 token 后，有草稿解码 8.2 tok/s，无草稿 5.8。前提是提示读完后立即释放 Neural Engine 的表面：一直保留时，它们会把权重映射挤出 24 GB 的 Mac，解码降到 6.4。

**Neural Engine**,`CMF_ANE=1`。它与 GPU 一起读取提示的各块:每层的 FFN、GDN 混合器的 `z` 投影和注意力层的 `q`。3 600 个 token 时首个 token 73.9 → 28.3 秒,文本相同;10 000 个 token 加 `--o1-spill` 时 217.6 → 113.5 秒,进程内存峰值 3.4 GB。解码留在 GPU 上。

表格由 `scripts/dory/q27_spec_decode.sh` 运行;日志在 `JOURNAL_Q27_MAC_DECODE_2026-11-06.md` 和 `JOURNAL_Q27_ANE_2026-11-05.md`。

## 构建 turbo 文件

```bash
python3 scripts/dory/router/q27_feat.py cortiq qwen38-27b-q4tp.cmf tokenizer.json wiki_test.txt keep27.bin
cortiq far-turbo qwen38-27b-q4tp.cmf qwen38-27b-turbo.cmf --exact-weights keep27.bin
```

`scripts/dory/router/keep27.bin` 是 Qwen3.8-27B 的拟合结果。`--o1 m,w,sink`、`--seal` 和 `--exact-rows` 设置远场;窗口、汇点和精确行必须不超过 2052 个键。其他带完整注意力层的模型可以用同样的命令和自己的拟合;目前只测量过 Qwen3.8-27B。

`scripts/dory/q27_final.sh` 和 `q27_quality.sh` 生成上面的表格,`q27_figures.py` 绘制图片,`JOURNAL_Q27_TURBO_2026-11-05.md` 记录尝试过的内容。

## 未包含的内容

- 远场运行在 Vulkan、DX12、Metal(`gpu_metal/o1c.rs`)和 CUDA([docs/CUDA.zh.md](CUDA.zh.md#16-gb-显卡上的-turbo):在 16 GB 显卡上整个运行保持在 15.0 GB,64 000 词元的提示也是如此)上。在 CPU 上,该文件按普通文件运行。测量在 NVIDIA 显卡和 M4 上进行。
- 封存点之后的文本是近似的:远处的内容会淡化,保持精确的行除外。拟合只见过插入的事实,其他类型的重要文本没有测试。
- 提示结束后到达的行留在窗口中或进入远场。与已封存对话不一致的回合会重新读取整个提示。
- 在 Vulkan 上,草稿需要不带惩罚的贪婪解码。
- 从封存点以下开始的生成保持精确,其内存随每个 token 增长。

## 许可证

权重:[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)。远场受美国专利申请号 19/738,763 保护;见 [PATENTS.md](../PATENTS.md)。
