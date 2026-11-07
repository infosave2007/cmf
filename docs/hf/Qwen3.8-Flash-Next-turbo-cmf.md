---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
library_name: cortiq
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q2tp
  - moe
  - long-context
  - speculative-decoding
  - vulkan
language:
  - en
  - ru
  - zh
---

# Qwen3.8-Flash-Next Turbo — CMF

The 2-bit experts of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) under a header
that holds the attention's device memory in place on long texts, and a second file that keeps the
busiest 5 % of every expert's neurons at eight bits. Prompts under 8192 tokens run exactly as the
[standard file](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf) does. One
[CMF](https://github.com/infosave2007/cmf) file each for `cortiq`, a Rust inference engine with no Python
and no ML framework; the 77 GB file runs on one GPU with the experts streaming from system RAM.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli --locked
hf download infosave/Qwen3.8-Flash-Next-turbo-cmf qwen38-flash-next-q2tp-turbo.cmf qwen38-flash-next-q2tp-turbo.mtp.cmf flashnext.profile --local-dir .
CMF_QWEN_PROFILE=flashnext.profile cortiq run qwen38-flash-next-q2tp-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Use cortiq 0.8.62 or later ([prebuilt binaries](https://github.com/infosave2007/cmf/releases)). `--no-think`
asks for a direct answer; without it the model reasons first, so raise `--max-tokens`. The `.mtp.cmf` file
next to the model turns on draft tokens. Each `cortiq run` loads the model again; for chat keep it loaded with
`cortiq serve` (an OpenAI-compatible server, as on the standard page).

## Files

| file | contents | size | code ppl | use |
|---|---|---:|---:|---|
| `qwen38-flash-next-q2tp-turbo.cmf` | the q2tp-v2 experts (2-bit gate/up, 4-bit down, GPTQ-calibrated) under the far-field header | 76.88 GB | 4.438 | **default** |
| `qwen38-flash-next-q2tp-hot-turbo.cmf` | the same, the busiest 5 % of every expert's neurons at 8 bits | {{HOT_SIZE}} GB | **{{HOT_PPL}}** | better quality, {{HOT_SPEED_NOTE}} |
| `qwen38-flash-next-q2tp-turbo.mtp.cmf` | multi-token-prediction head | 1.08 GB | | optional: draft tokens |
| `qwen38-flash-next-q2tp-hot-turbo.mtp.cmf` | the same head under the hot file's name | 1.08 GB | | optional |
| `flashnext.profile` | routing profile (not weights) | 98 KB | | recommended: warms the expert cache |

Code ppl: perplexity on 1,536 tokens of Rust source, lower is better; the standard page's q2tp-v2 file
measures 4.438 and its q4tp file (97 GB) 4.111. Attention, embeddings and the output head are 8-bit in
every file. An `.mtp.cmf` file is used only when it sits next to its model file under the name shown.

## What the files do

**A far field for the attention layers.** Flash-Next has 36 Gated DeltaNet layers, whose state is fixed
already, and 12 attention layers, whose keys and values grow with the text. In the turbo file the first
8192 rows of a sequence are read exactly on the card; at that point the attention layers' keys, values
and a trace of their queries come to the host, and every KV head gets a state of 4 sinks, 32 landmarks
and a window of the last 1800 rows. From there every later row, of the prompt or of the answer, attends
the window and the landmarks, the exact caches leave the card, and the attention's footprint and cost
stop growing with the text. The experts, the routing and the GDN layers do not see the change.
Patent application No. 19/738,763. `CMF_QWEN_O1=off` runs the file exact; `CMF_QWEN_O1=m,w,sink,seal`
runs any Flash-Next file with a far field of that shape.

**Hot neurons at eight bits.** The 2-bit experts lose most against the 4-bit ones in a few neurons of each
expert: over {{PROBE_TOKENS}} tokens of code and text the top 5 % of an expert's 640 neurons carry
{{PROBE_SHARE}} % of its Σ(w·act)². The hot file stores those 32 neurons per expert as one 8-bit tube
(gate, up and down rows from the bf16 checkpoint), zeroes their rows in the expert's 2-bit triple, and the
card adds every winner's tube to the token's MoE output in two small kernels; the file grows by
{{HOT_DELTA}} GB and the expert cache does not change.

## Performance

RTX 5090 (32 GB), Vulkan, cortiq 0.8.62, `cortiq bench --core --ctx N --tokens 60` with
`flashnext.profile` and a 29 GB device budget. Standard = the q2tp-v2 file of the standard page.

| prompt, tokens | first token, s (standard) | first token, s (**turbo**) | decode, tok/s (standard) | decode, tok/s (**turbo**) | peak VRAM, MiB (standard) | peak VRAM, MiB (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
{{BENCH_ROWS}}

A real prompt, the 6,045-token specification of a Three.js page, greedy, `--no-think`, 160 tokens of
answer with the MTP head: standard first token 113 s, decode 26.5 tok/s; sealed at 4096 rows
(`CMF_QWEN_O1=32,1800,4,4096`) 105 s and 29.3 tok/s, 76 % of the drafts accepted, the answer a working
page in both runs.

## Quality

| text | exact | sealed at 4096 rows |
|---|---:|---:|
| WikiText-2 test, 8192 tokens, ppl | 2.241 | {{WIKI_SEALED}} |

The rows past the seal read the window and the landmarks only; the first 4096 rows score the same in
both runs.

{{HOT_QUALITY}}

## Hardware and memory

Disk: the file size. RAM: the file size for full speed; with less the experts page in from disk. GPU:
Vulkan on NVIDIA (`cortiq gpu` lists adapters, `CMF_GPU_ADAPTER` picks one); `CMF_GPU_VRAM_MB` caps the
budget the expert cache takes, 29000 on the 32 GB card measured here. On headless Linux, if `cortiq gpu`
lists nothing, install `libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools`
([script](https://github.com/infosave2007/cmf/blob/master/tools/runpod_vulkan_setup.sh)) and export
`XDG_RUNTIME_DIR=/tmp` for every run.

## How the files were made

```bash
# the far-field header over the standard page's q2tp-v2 file
cortiq far-turbo qwen38-flash-next-q2tp-v2.cmf qwen38-flash-next-q2tp-turbo.cmf --o1 32,1800,4 --seal 8192
# the hot neurons: a statistic over code and text, then the tubes from the bf16 checkpoint
cortiq hot-probe --experts qwen38-flash-next-q2tp-v2.cmf probe.txt experts.stat --tokens {{PROBE_TOKENS}} --window 2048
cortiq hot-bake --experts qwen38-flash-next-q2tp-turbo.cmf Qwen/Qwen3.8-Flash-Next experts.stat qwen38-flash-next-q2tp-hot-turbo.cmf --frac 0.05
```

## Checksums

```
{{SHA_BLOCK}}
```

`cortiq verify FILE` checks every per-tensor hash.

## License

Qwen Community License 1.0 of the base model; the files are a quantization of its weights.

---

## Документация на русском

2-битные эксперты [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) под заголовком,
который держит память внимания на карте на месте при длинных текстах, и второй файл, где самые занятые 5 %
нейронов каждого эксперта хранятся в восьми битах. Запросы короче 8192 токенов идут точно так же, как в
[стандартном файле](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf). По одному файлу
[CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без Python и ML-фреймворков;
файл на 77 ГБ работает на одной карте, эксперты подтягиваются из оперативной памяти.

### Быстрый старт

```bash
cargo install cortiq-cli --locked
hf download infosave/Qwen3.8-Flash-Next-turbo-cmf qwen38-flash-next-q2tp-turbo.cmf qwen38-flash-next-q2tp-turbo.mtp.cmf flashnext.profile --local-dir .
CMF_QWEN_PROFILE=flashnext.profile cortiq run qwen38-flash-next-q2tp-turbo.cmf --prompt-file report.txt --greedy --no-think
```

Нужен cortiq 0.8.62 или новее ([готовые сборки](https://github.com/infosave2007/cmf/releases)). `--no-think`
просит прямой ответ; без него модель сначала рассуждает — поднимите `--max-tokens`. Файл `.mtp.cmf` рядом с моделью
включает черновые токены. Каждый `cortiq run` загружает модель заново; для чата держите её загруженной в
`cortiq serve` (OpenAI-совместимый сервер, как на стандартной странице).

### Файлы

| файл | содержимое | размер | ppl на коде | назначение |
|---|---|---:|---:|---|
| `qwen38-flash-next-q2tp-turbo.cmf` | эксперты q2tp-v2 (gate/up 2 бита, down 4 бита, калибровка GPTQ) под заголовком далёкого поля | 76,88 ГБ | 4,438 | **по умолчанию** |
| `qwen38-flash-next-q2tp-hot-turbo.cmf` | то же, самые занятые 5 % нейронов каждого эксперта в 8 битах | {{HOT_SIZE}} ГБ | **{{HOT_PPL}}** | выше качество, {{HOT_SPEED_NOTE_RU}} |
| `qwen38-flash-next-q2tp-turbo.mtp.cmf` | голова multi-token prediction | 1,08 ГБ | | необязательно: черновые токены |
| `qwen38-flash-next-q2tp-hot-turbo.mtp.cmf` | та же голова под именем hot-файла | 1,08 ГБ | | необязательно |
| `flashnext.profile` | профиль маршрутизации (не веса) | 98 КБ | | рекомендуется: прогревает кэш экспертов |

Ppl на коде — перплексия на 1 536 токенах исходников Rust, меньше — лучше; файл q2tp-v2 со стандартной
страницы даёт 4,438, файл q4tp (97 ГБ) — 4,111. Внимание, эмбеддинги и выходная голова в каждом файле 8-битные.
Файл `.mtp.cmf` берётся только когда лежит рядом с моделью под указанным именем.

### Что делают файлы

**Далёкое поле для слоёв внимания.** У Flash-Next 36 слоёв Gated DeltaNet, состояние которых фиксировано и так, и
12 слоёв внимания, чьи ключи и значения растут с текстом. В turbo-файле первые 8192 строки последовательности
читаются на карте точно; в этой точке ключи, значения и след запросов слоёв внимания уходят на хост, и каждая
KV-голова получает состояние из 4 стоков, 32 ориентиров и окна из последних 1800 строк. Дальше каждая строка —
запроса или ответа — смотрит на окно и ориентиры, точные кэши покидают карту, и след внимания и цена токена
перестают расти с текстом. Эксперты, маршрутизация и слои GDN изменения не видят. Патентная заявка
№ 19/738,763. `CMF_QWEN_O1=off` запускает файл точно; `CMF_QWEN_O1=m,w,sink,seal` запускает любой файл Flash-Next
с далёким полем такой формы.

**Горячие нейроны в восьми битах.** 2-битные эксперты больше всего проигрывают 4-битным в немногих нейронах
каждого эксперта: на {{PROBE_TOKENS}} токенах кода и текста верхние 5 % из 640 нейронов эксперта несут
{{PROBE_SHARE}} % его Σ(w·act)². Hot-файл хранит эти 32 нейрона эксперта как одну 8-битную трубку (строки gate,
up и down из bf16-чекпойнта), обнуляет их строки в 2-битной тройке эксперта, а карта двумя маленькими ядрами
добавляет трубку каждого победителя к выходу MoE токена; файл вырастает на {{HOT_DELTA}} ГБ, кэш экспертов не
меняется.

### Производительность

RTX 5090 (32 ГБ), Vulkan, cortiq 0.8.62, `cortiq bench --core --ctx N --tokens 60` с `flashnext.profile` и
бюджетом карты 29 ГБ. Standard — файл q2tp-v2 со стандартной страницы.

| запрос, токенов | первый токен, с (standard) | первый токен, с (**turbo**) | декод, tok/s (standard) | декод, tok/s (**turbo**) | пик VRAM, МиБ (standard) | пик VRAM, МиБ (**turbo**) |
|---:|---:|---:|---:|---:|---:|---:|
{{BENCH_ROWS}}

Реальный запрос — спецификация страницы Three.js на 6 045 токенов, greedy, `--no-think`, 160 токенов ответа с
головой MTP: standard — первый токен 113 с, декод 26,5 tok/s; с печатью на 4096 строках
(`CMF_QWEN_O1=32,1800,4,4096`) — 105 с и 29,3 tok/s, принято 76 % черновиков, в обоих прогонах ответ — рабочая
страница.

### Качество

| текст | точно | печать на 4096 строках |
|---|---:|---:|
| WikiText-2 test, 8192 токенов, ppl | 2,241 | {{WIKI_SEALED}} |

Строки после печати читают только окно и ориентиры; первые 4096 строк в обоих прогонах оцениваются одинаково.

{{HOT_QUALITY_RU}}

### Оборудование и память

Диск: размер файла. ОЗУ: размер файла для полной скорости; при меньшем объёме эксперты читаются с диска. GPU:
Vulkan на NVIDIA (`cortiq gpu` перечисляет адаптеры, `CMF_GPU_ADAPTER` выбирает); `CMF_GPU_VRAM_MB` ограничивает
бюджет кэша экспертов — 29000 на измеренной здесь карте 32 ГБ. На безголовом Linux, если `cortiq gpu` ничего не
показывает, установите `libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools`
([скрипт](https://github.com/infosave2007/cmf/blob/master/tools/runpod_vulkan_setup.sh)) и экспортируйте
`XDG_RUNTIME_DIR=/tmp` для каждого запуска.

### Как сделаны файлы

```bash
cortiq far-turbo qwen38-flash-next-q2tp-v2.cmf qwen38-flash-next-q2tp-turbo.cmf --o1 32,1800,4 --seal 8192
cortiq hot-probe --experts qwen38-flash-next-q2tp-v2.cmf probe.txt experts.stat --tokens {{PROBE_TOKENS}} --window 2048
cortiq hot-bake --experts qwen38-flash-next-q2tp-turbo.cmf Qwen/Qwen3.8-Flash-Next experts.stat qwen38-flash-next-q2tp-hot-turbo.cmf --frac 0.05
```

### Контрольные суммы

```
{{SHA_BLOCK}}
```

`cortiq verify FILE` проверяет хеш каждого тензора.

### Лицензия

Qwen Community License 1.0 базовой модели; файлы — квантизация её весов.

---

## 中文文档

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的 2 位专家权重，带一个在长文本上把注意力的显存占用固定不变的文件头；
另一个文件把每个专家最活跃的 5 % 神经元保留为 8 位。短于 8192 token 的提示与
[标准文件](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf) 完全一致。每个都是一个
[CMF](https://github.com/infosave2007/cmf) 文件，供 `cortiq` 使用——一个不依赖 Python 和 ML 框架的 Rust 推理引擎；
77 GB 的文件在一张 GPU 上运行，专家从系统内存流式读取。

### 快速开始

```bash
cargo install cortiq-cli --locked
hf download infosave/Qwen3.8-Flash-Next-turbo-cmf qwen38-flash-next-q2tp-turbo.cmf qwen38-flash-next-q2tp-turbo.mtp.cmf flashnext.profile --local-dir .
CMF_QWEN_PROFILE=flashnext.profile cortiq run qwen38-flash-next-q2tp-turbo.cmf --prompt-file report.txt --greedy --no-think
```

需要 cortiq 0.8.62 或更新版本（[预编译二进制](https://github.com/infosave2007/cmf/releases)）。`--no-think` 要求直接回答；
不加它模型会先推理，请提高 `--max-tokens`。放在模型旁边的 `.mtp.cmf` 文件开启草稿 token。每次 `cortiq run` 都会重新加载模型；
聊天请用 `cortiq serve` 常驻（OpenAI 兼容服务器，用法见标准页面）。

### 文件

| 文件 | 内容 | 大小 | 代码 ppl | 用途 |
|---|---|---:|---:|---|
| `qwen38-flash-next-q2tp-turbo.cmf` | q2tp-v2 专家（gate/up 2 位、down 4 位、GPTQ 校准）加远场文件头 | 76.88 GB | 4.438 | **默认** |
| `qwen38-flash-next-q2tp-hot-turbo.cmf` | 同上，每个专家最活跃的 5 % 神经元为 8 位 | {{HOT_SIZE}} GB | **{{HOT_PPL}}** | 质量更好，{{HOT_SPEED_NOTE_ZH}} |
| `qwen38-flash-next-q2tp-turbo.mtp.cmf` | 多 token 预测头 | 1.08 GB | | 可选：草稿 token |
| `qwen38-flash-next-q2tp-hot-turbo.mtp.cmf` | 同一个头，使用 hot 文件的名字 | 1.08 GB | | 可选 |
| `flashnext.profile` | 路由配置（不是权重） | 98 KB | | 推荐：预热专家缓存 |

代码 ppl：1,536 个 Rust 源码 token 上的困惑度，越低越好；标准页面的 q2tp-v2 文件为 4.438，q4tp 文件（97 GB）为 4.111。
每个文件中注意力、嵌入和输出头均为 8 位。`.mtp.cmf` 文件只有按所示名字放在模型文件旁边时才会被使用。

### 文件的作用

**注意力层的远场。** Flash-Next 有 36 个 Gated DeltaNet 层（状态本来就固定）和 12 个注意力层（键值随文本增长）。
在 turbo 文件中，序列的前 8192 行在显卡上精确读取；到这一点，注意力层的键、值和查询轨迹传到主机，每个 KV 头得到
4 个 sink、32 个 landmark 和最近 1800 行的窗口组成的状态。之后每一行——无论提示还是回答——只看窗口和 landmark，
精确缓存离开显卡，注意力的占用和每个 token 的开销不再随文本增长。专家、路由和 GDN 层不受影响。专利申请号 19/738,763。
`CMF_QWEN_O1=off` 让文件精确运行；`CMF_QWEN_O1=m,w,sink,seal` 让任何 Flash-Next 文件以该形状的远场运行。

**8 位热神经元。** 2 位专家相对 4 位专家的损失主要集中在每个专家的少数神经元上：在 {{PROBE_TOKENS}} 个代码和文本 token 上，
每个专家 640 个神经元中最活跃的 5 % 承载了 {{PROBE_SHARE}} % 的 Σ(w·act)²。hot 文件把每个专家的这 32 个神经元存成一根
8 位管（gate、up、down 行取自 bf16 检查点），在专家的 2 位三元组中把它们的行置零，显卡用两个小内核把每个被选中专家的管加到
该 token 的 MoE 输出上；文件增大 {{HOT_DELTA}} GB，专家缓存不变。

### 性能

RTX 5090（32 GB），Vulkan，cortiq 0.8.62，`cortiq bench --core --ctx N --tokens 60`，使用 `flashnext.profile`，显卡预算 29 GB。
Standard = 标准页面的 q2tp-v2 文件。

| 提示 token 数 | 首 token，s（standard） | 首 token，s（**turbo**） | 解码 tok/s（standard） | 解码 tok/s（**turbo**） | 峰值 VRAM，MiB（standard） | 峰值 VRAM，MiB（**turbo**） |
|---:|---:|---:|---:|---:|---:|---:|
{{BENCH_ROWS}}

真实提示：6,045 token 的 Three.js 页面规格，greedy，`--no-think`，带 MTP 头生成 160 个 token：standard 首 token 113 s、
解码 26.5 tok/s；在 4096 行封存（`CMF_QWEN_O1=32,1800,4,4096`）105 s、29.3 tok/s，76 % 的草稿被接受，两次运行的回答都是可用的页面。

### 质量

| 文本 | 精确 | 在 4096 行封存 |
|---|---:|---:|
| WikiText-2 test，8192 token，ppl | 2.241 | {{WIKI_SEALED}} |

封存之后的行只读窗口和 landmark；前 4096 行在两次运行中得分相同。

{{HOT_QUALITY_ZH}}

### 硬件与内存

磁盘：文件大小。内存：达到全速需要文件大小的内存；更少时专家从磁盘换入。GPU：NVIDIA 上的 Vulkan（`cortiq gpu` 列出适配器，
`CMF_GPU_ADAPTER` 选择）；`CMF_GPU_VRAM_MB` 限制专家缓存的预算，本文在 32 GB 显卡上为 29000。无头 Linux 上若 `cortiq gpu`
什么都不列出，请安装 `libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools`
（[脚本](https://github.com/infosave2007/cmf/blob/master/tools/runpod_vulkan_setup.sh)）并在每次运行前导出 `XDG_RUNTIME_DIR=/tmp`。

### 文件的制作方法

```bash
cortiq far-turbo qwen38-flash-next-q2tp-v2.cmf qwen38-flash-next-q2tp-turbo.cmf --o1 32,1800,4 --seal 8192
cortiq hot-probe --experts qwen38-flash-next-q2tp-v2.cmf probe.txt experts.stat --tokens {{PROBE_TOKENS}} --window 2048
cortiq hot-bake --experts qwen38-flash-next-q2tp-turbo.cmf Qwen/Qwen3.8-Flash-Next experts.stat qwen38-flash-next-q2tp-hot-turbo.cmf --frac 0.05
```

### 校验和

```
{{SHA_BLOCK}}
```

`cortiq verify FILE` 校验每个张量的哈希。

### 许可

基础模型的 Qwen Community License 1.0；这些文件是其权重的量化版本。
