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
  - q8
  - moe
  - speculative-decoding
  - vulkan
language:
  - en
  - ru
  - zh
---

# Qwen3.8-Flash-Next hot — CMF

The 2-bit experts of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with the shared
expert and the 28 busiest experts of every layer at eight bits: code perplexity 4.167 against 4.438 of the
77 GB 2-bit file and 4.111 of the 97 GB 4-bit file on the [standard page](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf),
in 83.8 GB. One [CMF](https://github.com/infosave2007/cmf) file for `cortiq`, a Rust inference engine with no
Python and no ML framework; it runs on one GPU with the experts streaming from system RAM.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli --locked
hf download infosave/Qwen3.8-Flash-Next-hot-cmf qwen38-flash-next-q2tp-hot.cmf qwen38-flash-next-q2tp-hot.mtp.cmf flashnext.profile --local-dir .
CMF_QWEN_PROFILE=flashnext.profile cortiq run qwen38-flash-next-q2tp-hot.cmf --prompt-file report.txt --greedy --no-think
```

Use cortiq 0.8.62 or later ([prebuilt binaries](https://github.com/infosave2007/cmf/releases)): an older cortiq
does not read the 8-bit tubes and runs the file with their rows at zero. `--no-think` asks for a direct answer;
without it the model reasons first, so raise `--max-tokens`. The `.mtp.cmf` file next to the model turns on draft
tokens. Each `cortiq run` loads the model again; for chat keep it loaded with [`cortiq serve`](#openai-compatible-server).

## OpenAI-compatible server

```bash
CMF_QWEN_PROFILE=flashnext.profile cortiq serve qwen38-flash-next-q2tp-hot.cmf --host 127.0.0.1 --port 8080

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen38-flash-next-q2tp-hot", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Write a Rust function that reverses a string, with a test."}]}'
```

The model loads once and stays in memory; the `.mtp.cmf` file next to it drafts tokens under `temperature: 0`.
Endpoints: `/v1/chat/completions` (streaming, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`, a web dashboard on `/`. Without `--host` the server listens on all interfaces (`0.0.0.0`).
`CMF_GPU_VRAM_MB` is the card's budget for the expert cache and the 8-bit banks (29000 on the 32 GB card measured
here); when the banks do not fit the budget the file runs on the host path. One stream was measured; concurrent
requests share the card.

## Files

| file | contents | size | code ppl |
|---|---|---:|---:|
| `qwen38-flash-next-q2tp-hot.cmf` | the q2tp-v2 experts (2-bit gate/up, 4-bit down, GPTQ-calibrated); the shared expert and the 28 busiest experts of every layer as 8-bit tubes | 83.81 GB | **4.167** |
| `qwen38-flash-next-q2tp-hot.mtp.cmf` | multi-token-prediction head, optional: draft tokens | 1.08 GB | |
| `flashnext.profile` | routing profile (not weights), recommended: warms the expert cache | 98 KB | |

Code ppl: perplexity on 1,536 tokens of Rust source, lower is better; the standard page's q2tp-v2 file measures
4.438 and its q4tp file (97 GB) 4.111. Attention, embeddings and the output head are 8-bit. The `.mtp.cmf` file
is used only when it sits next to the model file under the name shown.

## What the file does

![How the hot file is built: probe, bake, run](https://huggingface.co/infosave/Qwen3.8-Flash-Next-hot-cmf/resolve/main/assets/hot_build_en.png)

The 2-bit experts are GPTQ-calibrated, so lifting single neurons buys little: the busiest 5 % of every expert's
neurons at 8 bits moved the code ppl 4.438 → 4.425 for +6.7 GB. What pays is the grain the model routes at. The
shared expert runs on every token: whole at 8 bits it takes the ppl to 4.241 for +0.3 GB. Over 8192 tokens of
code and text the 28 busiest of a layer's 512 experts carry 43 % of the layer's Σ(w·act)²: whole at 8 bits they
take it on to 4.167 for +6.3 GB, 83 % of the way to the 97 GB q4tp file (the 28 experts alone 4.327; 28 random
experts 4.388). The file stores each of these experts as one 8-bit tube (gate and up rows from the bf16
checkpoint, the down columns from the 2-bit file's own calibrated down), zeroes their rows in the 2-bit triples,
and the card adds a winner's tube to the token's MoE output in two small kernels. The tubes stay resident
(6.5 GB), so the expert cache gets that much less of the device budget.

## Performance

RTX 5090 (32 GB), Vulkan, cortiq 0.8.62, `cortiq bench --core --ctx N --tokens 60` with `flashnext.profile`
and a 29 GB device budget; the 2-bit file is the standard page's.

| prompt, tokens | first token, s (2-bit) | first token, s (**hot**) | decode, tok/s (2-bit) | decode, tok/s (**hot**) | peak VRAM, MiB (2-bit / hot) |
|---:|---:|---:|---:|---:|---:|
| 1 000 | 3.4 | 9.0 | 59.6 | 27.8 | 30 070 / 30 420 |
| 4 000 | 19.8 | 43.4 | 54.6 | 29.9 | 29 902 / 30 829 |

On a 32 GB card the file runs at about half the 2-bit file's speed: the resident tubes come out of the expert
cache (23.4 → 17.1 GB here), so more experts stream from RAM per token, and every layer runs two more kernels.
More VRAM gives the cache back.

## Quality

![Code perplexity against file size](https://huggingface.co/infosave/Qwen3.8-Flash-Next-hot-cmf/resolve/main/assets/hot_quality_en.png)

| file | code ppl, 1536 tokens | WikiText-2 test, 12 × 512 tokens, ppl |
|---|---:|---:|
| `qwen38-flash-next-q2tp-hot.cmf` | **4.167** | **5.782** |
| standard page, q2tp-v2, 77 GB | 4.438 | 6.062 |
| standard page, q4tp, 97 GB | 4.111 | |

## The far field, optional

`CMF_QWEN_O1=32,1800,4,8192` runs the file with an O(1) far field for the 12 attention layers: the first 8192
rows are read exactly, then every KV head keeps 4 sinks, 32 landmarks and a window of the last 1800 rows, and the
attention's footprint stops growing with the text (patent application No. 19/738,763). On Flash-Next, whose
attention is 12 layers with 2 KV heads of 256, it is not faster: exact against sealed at 8192, decode 59.6 /
54.6 / 37.7 against 62.3 / 44.6 / 25.8 tok/s at 1 000 / 4 000 / 16 000 rows, the first token at 16 000 94
against 127 s; WikiText-2 over 8192 tokens 2.241 exact, 2.323 sealed at 4096. Use it when the attention state
must stay bounded.

## Hardware and memory

Disk: the file size. RAM: the file size for full speed; with less the experts page in from disk. GPU:
Vulkan on NVIDIA (`cortiq gpu` lists adapters, `CMF_GPU_ADAPTER` picks one); `CMF_GPU_VRAM_MB` caps the
budget the expert cache takes, 29000 on the 32 GB card measured here. On headless Linux, if `cortiq gpu`
lists nothing, install `libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools`
([script](https://github.com/infosave2007/cmf/blob/master/tools/runpod_vulkan_setup.sh)) and export
`XDG_RUNTIME_DIR=/tmp` for every run.

## How the file was made

```bash
# a statistic over code and text, then the tubes from the bf16 checkpoint over the standard page's q2tp-v2 file
cortiq hot-probe --experts qwen38-flash-next-q2tp-v2.cmf probe.txt experts.stat --tokens 8192 --window 2048
cortiq hot-bake --experts qwen38-flash-next-q2tp-v2.cmf Qwen/Qwen3.8-Flash-Next experts.stat qwen38-flash-next-q2tp-hot.cmf --whole-experts 28 --shared-q8
```

## Checksums

```
f571f2aeebf0de192386f60e037e62e6e9a94c5691a112c0716317d795ec94b6  qwen38-flash-next-q2tp-hot.cmf
a9cc9760d57db47d7365efcb21bdc13bdfc910947f3a547832d1945f0b56350a  qwen38-flash-next-q2tp-hot.mtp.cmf
```

`cortiq verify FILE` checks every per-tensor hash.

## License

Qwen Community License 1.0 of the base model; the file is a quantization of its weights.

---

## Документация на русском

2-битные эксперты [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), у которых общий эксперт и
28 самых занятых экспертов каждого слоя хранятся в восьми битах: перплексия на коде 4,167 против 4,438 у 2-битного
файла на 77 ГБ и 4,111 у 4-битного на 97 ГБ со [стандартной страницы](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf),
в 83,8 ГБ. Один файл [CMF](https://github.com/infosave2007/cmf) для `cortiq` — движка на Rust без Python и
ML-фреймворков; работает на одной карте, эксперты подтягиваются из оперативной памяти.

### Быстрый старт

```bash
cargo install cortiq-cli --locked
hf download infosave/Qwen3.8-Flash-Next-hot-cmf qwen38-flash-next-q2tp-hot.cmf qwen38-flash-next-q2tp-hot.mtp.cmf flashnext.profile --local-dir .
CMF_QWEN_PROFILE=flashnext.profile cortiq run qwen38-flash-next-q2tp-hot.cmf --prompt-file report.txt --greedy --no-think
```

Нужен cortiq 0.8.62 или новее ([готовые сборки](https://github.com/infosave2007/cmf/releases)): более старый cortiq
не читает 8-битные трубки и запускает файл с их обнулёнными строками. `--no-think` просит прямой ответ; без него
модель сначала рассуждает — поднимите `--max-tokens`. Файл `.mtp.cmf` рядом с моделью включает черновые токены.
Каждый `cortiq run` загружает модель заново; для чата держите её загруженной в [`cortiq serve`](#openai-совместимый-сервер).

### OpenAI-совместимый сервер

```bash
CMF_QWEN_PROFILE=flashnext.profile cortiq serve qwen38-flash-next-q2tp-hot.cmf --host 127.0.0.1 --port 8080

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen38-flash-next-q2tp-hot", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Напиши функцию на Rust, которая переворачивает строку, с тестом."}]}'
```

Модель загружается один раз и остаётся в памяти; файл `.mtp.cmf` рядом с ней даёт черновые токены при
`temperature: 0`. Эндпоинты: `/v1/chat/completions` (потоковая выдача, `tools`, `enable_thinking`),
`/v1/completions`, `/v1/models`, `/healthz`, веб-панель на `/`. Без `--host` сервер слушает все интерфейсы
(`0.0.0.0`). `CMF_GPU_VRAM_MB` — бюджет карты под кэш экспертов и 8-битные банки (29000 на измеренной здесь карте
32 ГБ); если банки в бюджет не помещаются, файл идёт по хост-пути. Измерен один поток; параллельные запросы делят
карту.

### Файлы

| файл | содержимое | размер | ppl на коде |
|---|---|---:|---:|
| `qwen38-flash-next-q2tp-hot.cmf` | эксперты q2tp-v2 (gate/up 2 бита, down 4 бита, калибровка GPTQ); общий эксперт и 28 самых занятых экспертов каждого слоя — 8-битными трубками | 83,81 ГБ | **4,167** |
| `qwen38-flash-next-q2tp-hot.mtp.cmf` | голова multi-token prediction, необязательно: черновые токены | 1,08 ГБ | |
| `flashnext.profile` | профиль маршрутизации (не веса), рекомендуется: прогревает кэш экспертов | 98 КБ | |

Ppl на коде — перплексия на 1 536 токенах исходников Rust, меньше — лучше; файл q2tp-v2 со стандартной страницы
даёт 4,438, файл q4tp (97 ГБ) — 4,111. Внимание, эмбеддинги и выходная голова 8-битные. Файл `.mtp.cmf` берётся
только когда лежит рядом с моделью под указанным именем.

### Что делает файл

![Как собран hot-файл: зонд, сборка, запуск](https://huggingface.co/infosave/Qwen3.8-Flash-Next-hot-cmf/resolve/main/assets/hot_build_ru.png)

2-битные эксперты калиброваны GPTQ, поэтому поднимать отдельные нейроны почти бесполезно: самые занятые 5 %
нейронов каждого эксперта в 8 битах сдвинули ppl на коде с 4,438 до 4,425 за +6,7 ГБ. Окупается зерно, которым
модель маршрутизирует. Общий эксперт работает на каждом токене: целиком в 8 битах он даёт 4,241 за +0,3 ГБ. На
8192 токенах кода и текста 28 самых занятых из 512 экспертов слоя несут 43 % его Σ(w·act)²: целиком в 8 битах они
доводят до 4,167 за +6,3 ГБ — 83 % пути к файлу q4tp на 97 ГБ (28 экспертов без общего — 4,327; 28 случайных —
4,388). Файл хранит каждый такой эксперт одной 8-битной трубкой (строки gate и up из bf16-чекпойнта, столбцы
down — из калиброванного down самого 2-битного файла), обнуляет их строки в 2-битных тройках, а карта двумя
маленькими ядрами добавляет трубку победителя к выходу MoE токена. Трубки резидентны (6,5 ГБ), и кэш экспертов
получает на столько меньше бюджета карты.

### Производительность

RTX 5090 (32 ГБ), Vulkan, cortiq 0.8.62, `cortiq bench --core --ctx N --tokens 60` с `flashnext.profile` и
бюджетом карты 29 ГБ; 2-битный файл — со стандартной страницы.

| запрос, токенов | первый токен, с (2 бита) | первый токен, с (**hot**) | декод, tok/s (2 бита) | декод, tok/s (**hot**) | пик VRAM, МиБ (2 бита / hot) |
|---:|---:|---:|---:|---:|---:|
| 1 000 | 3,4 | 9,0 | 59,6 | 27,8 | 30 070 / 30 420 |
| 4 000 | 19,8 | 43,4 | 54,6 | 29,9 | 29 902 / 30 829 |

На карте 32 ГБ файл идёт примерно вдвое медленнее 2-битного: резидентные трубки отнимают память у кэша экспертов
(здесь 23,4 → 17,1 ГБ), так что больше экспертов читается из ОЗУ на каждый токен, и каждый слой запускает два
лишних ядра. Больше VRAM возвращает кэш.

### Качество

![Перплексия на коде против размера файла](https://huggingface.co/infosave/Qwen3.8-Flash-Next-hot-cmf/resolve/main/assets/hot_quality_ru.png)

| файл | ppl на коде, 1536 токенов | WikiText-2 test, 12 × 512 токенов, ppl |
|---|---:|---:|
| `qwen38-flash-next-q2tp-hot.cmf` | **4,167** | **5,782** |
| стандартная страница, q2tp-v2, 77 ГБ | 4,438 | 6,062 |
| стандартная страница, q4tp, 97 ГБ | 4,111 | |

### Далёкое поле, по желанию

`CMF_QWEN_O1=32,1800,4,8192` запускает файл с далёким полем O(1) для 12 слоёв внимания: первые 8192 строки читаются
точно, дальше каждая KV-голова держит 4 стока, 32 ориентира и окно из последних 1800 строк, и след внимания
перестаёт расти с текстом (патентная заявка № 19/738,763). На Flash-Next, где внимание — 12 слоёв с 2 KV-головами
по 256, это не быстрее: точно против печати на 8192 — декод 59,6 / 54,6 / 37,7 против 62,3 / 44,6 / 25,8 tok/s на
1 000 / 4 000 / 16 000 строках, первый токен на 16 000 — 94 против 127 с; WikiText-2 на 8192 токенах 2,241 точно,
2,323 с печатью на 4096. Нужно, когда состояние внимания должно оставаться ограниченным.

### Оборудование и память

Диск: размер файла. ОЗУ: размер файла для полной скорости; при меньшем объёме эксперты читаются с диска. GPU:
Vulkan на NVIDIA (`cortiq gpu` перечисляет адаптеры, `CMF_GPU_ADAPTER` выбирает); `CMF_GPU_VRAM_MB` ограничивает
бюджет кэша экспертов — 29000 на измеренной здесь карте 32 ГБ. На безголовом Linux, если `cortiq gpu` ничего не
показывает, установите `libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools`
([скрипт](https://github.com/infosave2007/cmf/blob/master/tools/runpod_vulkan_setup.sh)) и экспортируйте
`XDG_RUNTIME_DIR=/tmp` для каждого запуска.

### Как сделан файл

```bash
cortiq hot-probe --experts qwen38-flash-next-q2tp-v2.cmf probe.txt experts.stat --tokens 8192 --window 2048
cortiq hot-bake --experts qwen38-flash-next-q2tp-v2.cmf Qwen/Qwen3.8-Flash-Next experts.stat qwen38-flash-next-q2tp-hot.cmf --whole-experts 28 --shared-q8
```

### Контрольные суммы

```
f571f2aeebf0de192386f60e037e62e6e9a94c5691a112c0716317d795ec94b6  qwen38-flash-next-q2tp-hot.cmf
a9cc9760d57db47d7365efcb21bdc13bdfc910947f3a547832d1945f0b56350a  qwen38-flash-next-q2tp-hot.mtp.cmf
```

`cortiq verify FILE` проверяет хеш каждого тензора.

### Лицензия

Qwen Community License 1.0 базовой модели; файл — квантизация её весов.

---

## 中文文档

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的 2 位专家权重，其中共享专家和每层最活跃的 28 个专家保留为
8 位：代码困惑度 4.167，而[标准页面](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf)上 77 GB 的 2 位文件为 4.438、97 GB 的
4 位文件为 4.111；本文件 83.8 GB。一个 [CMF](https://github.com/infosave2007/cmf) 文件，供 `cortiq` 使用——一个不依赖 Python 和
ML 框架的 Rust 推理引擎；在一张 GPU 上运行，专家从系统内存流式读取。

### 快速开始

```bash
cargo install cortiq-cli --locked
hf download infosave/Qwen3.8-Flash-Next-hot-cmf qwen38-flash-next-q2tp-hot.cmf qwen38-flash-next-q2tp-hot.mtp.cmf flashnext.profile --local-dir .
CMF_QWEN_PROFILE=flashnext.profile cortiq run qwen38-flash-next-q2tp-hot.cmf --prompt-file report.txt --greedy --no-think
```

需要 cortiq 0.8.62 或更新版本（[预编译二进制](https://github.com/infosave2007/cmf/releases)）：更旧的 cortiq 不读取 8 位管，
会以置零的行运行该文件。`--no-think` 要求直接回答；不加它模型会先推理，请提高 `--max-tokens`。放在模型旁边的 `.mtp.cmf`
文件开启草稿 token。每次 `cortiq run` 都会重新加载模型；聊天请用 [`cortiq serve`](#openai-兼容服务器) 常驻。

### OpenAI 兼容服务器

```bash
CMF_QWEN_PROFILE=flashnext.profile cortiq serve qwen38-flash-next-q2tp-hot.cmf --host 127.0.0.1 --port 8080

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen38-flash-next-q2tp-hot", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "写一个反转字符串的 Rust 函数，并附带测试。"}]}'
```

模型只加载一次并常驻内存；放在旁边的 `.mtp.cmf` 文件在 `temperature: 0` 下提供草稿 token。端点：`/v1/chat/completions`
（流式、`tools`、`enable_thinking`）、`/v1/completions`、`/v1/models`、`/healthz`，以及 `/` 上的网页面板。不加 `--host`
时服务器监听所有网卡（`0.0.0.0`）。`CMF_GPU_VRAM_MB` 是显卡用于专家缓存和 8 位库的预算（本文在 32 GB 显卡上为 29000）；
当库放不进预算时，文件走主机路径。只测过单路请求；并发请求共用同一张显卡。

### 文件

| 文件 | 内容 | 大小 | 代码 ppl |
|---|---|---:|---:|
| `qwen38-flash-next-q2tp-hot.cmf` | q2tp-v2 专家（gate/up 2 位、down 4 位、GPTQ 校准）；共享专家和每层最活跃的 28 个专家为 8 位管 | 83.81 GB | **4.167** |
| `qwen38-flash-next-q2tp-hot.mtp.cmf` | 多 token 预测头，可选：草稿 token | 1.08 GB | |
| `flashnext.profile` | 路由配置（不是权重），推荐：预热专家缓存 | 98 KB | |

代码 ppl：1,536 个 Rust 源码 token 上的困惑度，越低越好；标准页面的 q2tp-v2 文件为 4.438，q4tp 文件（97 GB）为 4.111。
注意力、嵌入和输出头均为 8 位。`.mtp.cmf` 文件只有按所示名字放在模型文件旁边时才会被使用。

### 文件的作用

![hot 文件是如何构建的：探测、烘焙、运行](https://huggingface.co/infosave/Qwen3.8-Flash-Next-hot-cmf/resolve/main/assets/hot_build_zh.png)

2 位专家经过 GPTQ 校准，单独提升神经元收益很小：每个专家最活跃的 5 % 神经元改为 8 位，代码 ppl 只从 4.438 到 4.425，却多了
6.7 GB。有效的是模型路由所用的粒度。共享专家每个 token 都会运行：整体改为 8 位后 ppl 到 4.241，只多 0.3 GB。在 8192 个代码和
文本 token 上，每层 512 个专家中最活跃的 28 个承载该层 43 % 的 Σ(w·act)²：整体改为 8 位后 ppl 到 4.167，多 6.3 GB——走完了到
97 GB q4tp 文件 83 % 的路（仅这 28 个专家 4.327；随机 28 个专家 4.388）。文件把每个这样的专家存成一根 8 位管（gate、up 行取自
bf16 检查点，down 列取自 2 位文件自身校准过的 down），在 2 位三元组中把它们的行置零，显卡用两个小内核把被选中专家的管加到该
token 的 MoE 输出上。这些管常驻显存（6.5 GB），专家缓存的预算相应减少。

### 性能

RTX 5090（32 GB），Vulkan，cortiq 0.8.62，`cortiq bench --core --ctx N --tokens 60`，使用 `flashnext.profile`，显卡预算 29 GB；
2 位文件来自标准页面。

| 提示 token 数 | 首 token，s（2 位） | 首 token，s（**hot**） | 解码 tok/s（2 位） | 解码 tok/s（**hot**） | 峰值 VRAM，MiB（2 位 / hot） |
|---:|---:|---:|---:|---:|---:|
| 1 000 | 3.4 | 9.0 | 59.6 | 27.8 | 30 070 / 30 420 |
| 4 000 | 19.8 | 43.4 | 54.6 | 29.9 | 29 902 / 30 829 |

在 32 GB 显卡上，该文件的速度约为 2 位文件的一半：常驻的管占用了专家缓存的显存（此处 23.4 → 17.1 GB），每个 token 要从内存
流式读取更多专家，而且每层多运行两个内核。更大的显存会把缓存还回来。

### 质量

![代码困惑度与文件大小](https://huggingface.co/infosave/Qwen3.8-Flash-Next-hot-cmf/resolve/main/assets/hot_quality_zh.png)

| 文件 | 代码 ppl，1536 token | WikiText-2 test，12 × 512 token，ppl |
|---|---:|---:|
| `qwen38-flash-next-q2tp-hot.cmf` | **4.167** | **5.782** |
| 标准页面，q2tp-v2，77 GB | 4.438 | 6.062 |
| 标准页面，q4tp，97 GB | 4.111 | |

### 远场（可选）

`CMF_QWEN_O1=32,1800,4,8192` 让文件带着 12 个注意力层的 O(1) 远场运行：前 8192 行精确读取，之后每个 KV 头保留 4 个 sink、
32 个 landmark 和最近 1800 行的窗口，注意力的占用不再随文本增长（专利申请号 19/738,763）。在 Flash-Next 上（注意力为 12 层、
每层 2 个 256 维 KV 头）它并不更快：精确对比在 8192 行封存，1 000 / 4 000 / 16 000 行的解码为 59.6 / 54.6 / 37.7 对
62.3 / 44.6 / 25.8 tok/s，16 000 行的首 token 为 94 s 对 127 s；8192 个 WikiText-2 token 的 ppl 精确 2.241，在 4096 行封存 2.323。
需要注意力状态保持有界时使用。

### 硬件与内存

磁盘：文件大小。内存：达到全速需要文件大小的内存；更少时专家从磁盘换入。GPU：NVIDIA 上的 Vulkan（`cortiq gpu` 列出适配器，
`CMF_GPU_ADAPTER` 选择）；`CMF_GPU_VRAM_MB` 限制专家缓存的预算，本文在 32 GB 显卡上为 29000。无头 Linux 上若 `cortiq gpu`
什么都不列出，请安装 `libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools`
（[脚本](https://github.com/infosave2007/cmf/blob/master/tools/runpod_vulkan_setup.sh)）并在每次运行前导出 `XDG_RUNTIME_DIR=/tmp`。

### 文件的制作方法

```bash
cortiq hot-probe --experts qwen38-flash-next-q2tp-v2.cmf probe.txt experts.stat --tokens 8192 --window 2048
cortiq hot-bake --experts qwen38-flash-next-q2tp-v2.cmf Qwen/Qwen3.8-Flash-Next experts.stat qwen38-flash-next-q2tp-hot.cmf --whole-experts 28 --shared-q8
```

### 校验和

```
f571f2aeebf0de192386f60e037e62e6e9a94c5691a112c0716317d795ec94b6  qwen38-flash-next-q2tp-hot.cmf
a9cc9760d57db47d7365efcb21bdc13bdfc910947f3a547832d1945f0b56350a  qwen38-flash-next-q2tp-hot.mtp.cmf
```

`cortiq verify FILE` 校验每个张量的哈希。

### 许可

基础模型的 Qwen Community License 1.0；该文件是其权重的量化版本。
