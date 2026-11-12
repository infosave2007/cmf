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
  - q4tp
  - moe
  - code
  - expert-pruning
  - speculative-decoding
  - vulkan
  - cuda
language:
  - en
  - ru
  - zh
---

# Qwen3.8-Flash-Next Fast Code — CMF

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) cut for code: of the 512 routed experts of
every layer it keeps the 256 that code and a long front-end task route to, at 4 bits, with the shared expert, the
draft (MTP) head and the n-gram table in one 66.8 GB [CMF](https://github.com/infosave2007/cmf) file. Given a
2 151-token spec of a 3D aquarium, it writes **a working three.js page on the first try** (4 of 4 runs); the 2-bit
Strata Coder cut of the same model needed a repair turn. The experts fit on one 96 GB card: a prompt is read at
**~840 tokens/s**, four times the full 512-expert file on the same card, and code is written at **136–166 tokens/s**,
in 41 GB of VRAM instead of 60. Over CUDA it runs on **cards from 12 GB**: the card keeps the experts that fit and the
host's cores compute the rest, and the same working aquarium page is written at **105 / 118 / 133 tokens/s** within
12 / 16 / 24 GB of card memory. It runs with `cortiq`, a Rust inference engine with no Python and no ML framework.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

![The aquarium page Fast Code wrote](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_aquarium.png)

*Fast Code's greedy answer to the aquarium spec, rendered as written: one HTML file, a glass tank, three clownfish,
plants and stones.*

## Quick start

```bash
cargo install cortiq-cli --locked
hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile --local-dir .
CMF_QWEN_PROFILE=fast-code.profile cortiq run qwen38-flash-next-fast-code.cmf --prompt-file task.txt --greedy --no-think
```

Use cortiq 0.8.65 or later ([prebuilt binaries](https://github.com/infosave2007/cmf/releases)). `--no-think` asks
for a direct answer; without it the model reasons first, so raise `--max-tokens`. Each `cortiq run` loads the model
again; for chat and editors keep it loaded with [`cortiq serve`](#openai-compatible-server).

An NVIDIA card from 12 GB through CUDA (cortiq 0.8.74 or later, built with the `cuda` feature):

```bash
cargo install cortiq-cli --locked --features cuda
CMF_GPU=cuda CMF_QWEN_PROFILE=fast-code.profile CMF_QWEN_ROUTE_TAIL=0.05 \
  CMF_SERVE_SLOTS=1 CMF_QWEN_CUDA_REG=1 \
  cortiq serve qwen38-flash-next-fast-code.cmf --port 8080 --host 127.0.0.1
```

## Files

| file | contents | size |
|---|---|---:|
| `qwen38-flash-next-fast-code.cmf` | 256 of 512 routed experts a layer (4-bit q4tp) and the shared expert; attention, embeddings and the output head 8-bit; the n-gram table 4-bit; the draft (MTP) head | 66.76 GB |
| `fast-code.profile` | routing profile (not weights): the experts' order on the card | 49 KB |
| `qwen38-flash-next-fast-code.cmf.sha256` | checksum | |

The kept experts are those routed to most over two texts, each normalized and weighted equally: 450 KB of
programming tasks and answers (Magicoder OSS-Instruct across its languages and Evol-Instruct) and the aquarium spec
with six answers to it. They hold 83 % of code's routing mass and 81 % of the task's; a cut by code alone held 87 %
and 65 %, and its aquarium page did not run. The aquarium spec was among the selection texts. The cut is made for
code; for general chat use the [full file](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf).

## Speed and accuracy

One RTX PRO 6000 (96 GB, Vulkan), one stream, the same HTTP client and prompts for every row, greedy, thinking off,
draft tokens on. Strata Coder is ISTA-DASLab's 2-bit cut of the same model (256 experts chosen for code, agents and
vision) on its own engine Strata 0.1.58 (llama.cpp-based, CUDA); the full file is the 512-expert
[q4tp file](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf) on cortiq.

| | Fast Code | Strata Coder (Strata) | full file (cortiq) |
|---|---:|---:|---:|
| aquarium spec, greedy: a working page | **first try** | after a repair turn | first try |
| HumanEval pass@1 (164 tasks) | 93.3 % | 94.5 % | 94.5 % |
| code perplexity, code written after 20.10.2026 (8192 tokens) | **3.337** | 3.436¹ | 3.366 |
| code perplexity, Rust (1536 tokens) | 4.900 | 4.690¹ | 4.111 |
| prompt 1K / 4K / 16K / 32K tokens, tok/s | 798 / 827 / 844 / 842 | 1916 / 5492 / 7229 / 7659 | 171 / 208 / 206 / 210 |
| generation after a 4K / 16K / 32K prompt, tok/s | 161 / 166 / 136 | 220 / 223 / 222 | 118 / 138 / 116 |
| file on disk | 66.8 GB | 58.4 GB + 5.2 GB MTP | 98.6 GB |
| peak VRAM (32K context) | 40.9 GB | 31.1 GB | 59.6 GB |

¹ Strata Coder's weights imported into CMF and run by cortiq: the same text and tokenizer for both.
HumanEval: greedy, no thinking, the answer's code run against the tests (±3 tasks is run-to-run noise). Fast Code's
speeds are the mean of two runs. The n-gram table (28 GB) stays in system RAM in every row.

![Speed on one RTX PRO 6000](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_speed_en.png)

![Accuracy on code](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_quality_en.png)

## Cards from 12 GB (CUDA)

From cortiq 0.8.74 the CUDA backend serves the file on a card that cannot hold all its experts. The card keeps the
experts that fit, the busiest of the routing profile first; an expert a token routes to outside of them is computed
by the host's cores (an AVX-512 VNNI kernel on the 4-bit weights) while the card computes the held ones, and the
experts a text keeps asking for move to the card between tokens. Within 12 GB the card holds 1 357 of the 12 288
experts.

| card memory | aquarium page written to the end, tok/s |
|---|---:|
| 12 GB | **104.9** |
| 16 GB | **117.6** |
| 24 GB | **132.8** |

One RTX 5090 held to the memory shown (`CMF_GPU_VRAM_MB`), host 2× EPYC 9554 with 31 threads and 62 GB of RAM for
the container, one request at a time, greedy, thinking off, draft tokens on, the command of the quick start. The
answer is the complete working page in every row. Within 12 GB, 38 of the first 40 HumanEval tasks pass. The host
needs AVX-512 VNNI (Zen 4, Sapphire Rapids or newer) and eight threads or more; on another host every missing expert
is copied to the card instead, which is slower. `CMF_QWEN_CPU_EXPERT_THREADS` sets the number of host threads.
`CMF_QWEN_CUDA_REG=1` keeps the experts outside the card pinned in RAM, so the host never waits on the disk for them.

## OpenAI-compatible server

```bash
CMF_QWEN_PROFILE=fast-code.profile cortiq serve qwen38-flash-next-fast-code.cmf --host 127.0.0.1 --port 8080

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "fast-code", "enable_thinking": false, "temperature": 0,
       "messages": [{"role": "user", "content": "Write a Rust function that parses a semantic version, with tests."}]}'
```

Endpoints: `/v1/chat/completions` (streaming, `tools`, `enable_thinking`), `/v1/completions`, `/v1/models`,
`/healthz`, a web dashboard on `/`. Draft tokens are used under `temperature: 0`. Without `--host` the server listens
on all interfaces.

## Hardware

The experts live on the card: about 41 GB of VRAM at a 32K context, plus 28 GB of system RAM for the n-gram table.
Measured on an RTX PRO 6000 (96 GB). A card from 12 GB runs it through CUDA with the host's cores computing the
experts that do not fit ([above](#cards-from-12-gb-cuda)); the file then stays in system RAM (62 GB measured). `cortiq gpu` lists the adapters; `CMF_GPU_ADAPTER` picks one. On headless Linux set
`XDG_RUNTIME_DIR=/tmp`.

## Long context

Attention is exact on the card up to 262 144 tokens (the sparse attention keeps 2 048 positions a token).
Generation holds with length: 166 tok/s after 16K tokens, 136 after 32K.

## Verify the download

```bash
sha256sum -c qwen38-flash-next-fast-code.cmf.sha256
cortiq verify qwen38-flash-next-fast-code.cmf
```

Weights derive from Qwen's release and remain under its license. Review generated code before running it.

---

## Документация на русском

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), урезанная под код: из 512 экспертов каждого
слоя оставлены 256, к которым обращаются код и большая фронтенд-задача, в 4 битах, вместе с общим экспертом, головой
черновиков (MTP) и таблицей n-грамм, в одном файле [CMF](https://github.com/infosave2007/cmf) на 66.8 ГБ. По ТЗ
3D-аквариума на 2 151 токен она пишет **рабочую страницу на three.js с первой попытки** (4 из 4 запусков); 2-битной
сборке Strata Coder той же модели понадобился ход с исправлением. Эксперты помещаются на одну карту на 96 ГБ: промпт
читается со скоростью **~840 токенов/с** — вчетверо быстрее полного файла на той же карте, — код пишется со скоростью
**136–166 токенов/с**, в 41 ГБ видеопамяти вместо 60. Через CUDA она работает на **картах от 12 ГБ**: карта держит экспертов, которые
поместились, остальных считают ядра процессора, и та же рабочая страница аквариума пишется со скоростью
**105 / 118 / 133 токена/с** в 12 / 16 / 24 ГБ видеопамяти. Запускается движком `cortiq` на Rust, без Python и
ML-фреймворков.

![Страница аквариума от Fast Code](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_aquarium.png)

*Жадный ответ Fast Code на ТЗ аквариума, как он написан: один HTML-файл, стеклянный аквариум, три рыбы-клоуна,
растения и камни.*

### Быстрый старт

```bash
cargo install cortiq-cli --locked
hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile --local-dir .
CMF_QWEN_PROFILE=fast-code.profile cortiq run qwen38-flash-next-fast-code.cmf --prompt-file task.txt --greedy --no-think
```

Нужен cortiq 0.8.65 или новее. `--no-think` — ответ без рассуждений; без него поднимите `--max-tokens`. Для чата и
редакторов держите модель загруженной через `cortiq serve` (OpenAI-совместимый сервер, см. выше).

Карта NVIDIA от 12 ГБ через CUDA (cortiq 0.8.74 или новее, сборка с признаком `cuda`):

```bash
cargo install cortiq-cli --locked --features cuda
CMF_GPU=cuda CMF_QWEN_PROFILE=fast-code.profile CMF_QWEN_ROUTE_TAIL=0.05 \
  CMF_SERVE_SLOTS=1 CMF_QWEN_CUDA_REG=1 \
  cortiq serve qwen38-flash-next-fast-code.cmf --port 8080 --host 127.0.0.1
```

### Файлы

| файл | содержимое | размер |
|---|---|---:|
| `qwen38-flash-next-fast-code.cmf` | 256 из 512 экспертов слоя (4 бита, q4tp) и общий эксперт; внимание, эмбеддинги и выходная голова — 8 бит; таблица n-грамм — 4 бита; голова черновиков (MTP) | 66.76 ГБ |
| `fast-code.profile` | профиль маршрутизации (не веса): порядок экспертов на карте | 49 КБ |

Оставлены эксперты, к которым чаще всего обращаются два текста, каждый нормирован, веса равные: 450 КБ задач и решений
(Magicoder OSS-Instruct на всех его языках и Evol-Instruct) и ТЗ аквариума с шестью ответами на него. Они держат 83 %
массы маршрутизации кода и 81 % задачи; срез только по коду держал 87 % и 65 %, и его страница аквариума не работала.
ТЗ аквариума было среди текстов отбора. Срез сделан под код; для общего чата нужен
[полный файл](https://huggingface.co/infosave/Qwen3.8-Flash-Next-cmf).

### Скорость и точность

Одна RTX PRO 6000 (96 ГБ, Vulkan), один поток, один HTTP-клиент и одни промпты, жадный выбор, без рассуждений, черновики
включены. Strata Coder — 2-битная сборка ISTA-DASLab той же модели на собственном движке Strata 0.1.58 (на основе
llama.cpp, CUDA).

| | Fast Code | Strata Coder (Strata) | полный файл (cortiq) |
|---|---:|---:|---:|
| ТЗ аквариума, жадно: рабочая страница | **с первой попытки** | после хода с исправлением | с первой попытки |
| HumanEval pass@1 (164 задачи) | 93.3 % | 94.5 % | 94.5 % |
| перплексия на коде после 20.10.2026 (8192 токена) | **3.337** | 3.436¹ | 3.366 |
| перплексия на Rust (1536 токенов) | 4.900 | 4.690¹ | 4.111 |
| промпт 1K / 4K / 16K / 32K токенов, ток/с | 798 / 827 / 844 / 842 | 1916 / 5492 / 7229 / 7659 | 171 / 208 / 206 / 210 |
| генерация после промпта 4K / 16K / 32K, ток/с | 161 / 166 / 136 | 220 / 223 / 222 | 118 / 138 / 116 |
| пик VRAM (контекст 32K) | 40.9 ГБ | 31.1 ГБ | 59.6 ГБ |

¹ Веса Strata Coder, импортированные в CMF и запущенные cortiq: один текст и один токенизатор. HumanEval: ±3 задачи —
разброс между прогонами. Скорости Fast Code — среднее двух прогонов.

![Скорость на одной RTX PRO 6000](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_speed_ru.png)

![Точность на коде](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_quality_ru.png)

### Карты от 12 ГБ (CUDA)

Начиная с cortiq 0.8.74 CUDA-бэкенд обслуживает файл на карте, куда все эксперты не помещаются. Карта держит тех,
что поместились, начиная с самых востребованных по профилю маршрутизации; эксперта вне карты считают ядра процессора
(ядро AVX-512 VNNI на 4-битных весах), пока карта считает своих, а эксперты, к которым текст обращается постоянно,
переезжают на карту между токенами. В 12 ГБ карта держит 1 357 экспертов из 12 288.

| память карты | страница аквариума, написанная до конца, ток/с |
|---|---:|
| 12 ГБ | **104.9** |
| 16 ГБ | **117.6** |
| 24 ГБ | **132.8** |

Одна RTX 5090, ограниченная указанной памятью (`CMF_GPU_VRAM_MB`), хост 2× EPYC 9554 с 31 потоком и 62 ГБ ОЗУ на
контейнер, один запрос за раз, жадный выбор, без рассуждений, черновики включены, команда из быстрого старта. Ответ
в каждой строке — полная рабочая страница. В 12 ГБ проходят 38 из первых 40 задач HumanEval. Хосту нужны AVX-512 VNNI
(Zen 4, Sapphire Rapids и новее) и не меньше восьми потоков; на другом хосте каждый недостающий эксперт копируется на
карту, что медленнее. `CMF_QWEN_CPU_EXPERT_THREADS` задаёт число потоков хоста.
`CMF_QWEN_CUDA_REG=1` закрепляет в памяти экспертов вне карты, чтобы хост не ждал их с диска.

### Железо и длинный контекст

Эксперты лежат на карте: около 41 ГБ VRAM при контексте 32K и 28 ГБ оперативной памяти под таблицу n-грамм. Карта
от 12 ГБ запускает модель через CUDA, а не поместившихся экспертов считают ядра процессора (см. выше); файл при этом
лежит в оперативной памяти (измерено с 62 ГБ). Внимание точное до 262 144 токенов; скорость
генерации держится: 166 ток/с после 16K токенов, 136 — после 32K.

---

## 中文文档

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的代码专用裁剪版：每层 512 个路由专家中保留代码与一个大型前端任务最常用的
256 个（4 位），连同共享专家、草稿（MTP）头和 n-gram 表，打包为一个 66.8 GB 的 [CMF](https://github.com/infosave2007/cmf) 文件。
面对 2 151 token 的 3D 水族箱需求说明，它**一次写出可运行的 three.js 页面**（4 次中 4 次）；同一模型的 2 位 Strata Coder 裁剪版需要一轮修复。
专家可完整放入一张 96 GB 显卡：读取提示约 **840 token/s**，是同卡完整文件的 4 倍；生成代码 **136–166 token/s**，显存 41 GB 而非 60 GB。
通过 CUDA 可在 **12 GB 起的显卡**上运行：显卡保留放得下的专家，其余由主机 CPU 核心计算，在 12 / 16 / 24 GB 显存内以
**105 / 118 / 133 token/s** 写出同一可运行的水族箱页面。使用 `cortiq` 运行：Rust 推理引擎，无需 Python 和机器学习框架。

![Fast Code 写出的水族箱页面](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_aquarium.png)

*Fast Code 对水族箱需求的贪心回答，按原样渲染：一个 HTML 文件，玻璃水族箱、三条小丑鱼、水草和石头。*

### 快速开始

```bash
cargo install cortiq-cli --locked
hf download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile --local-dir .
CMF_QWEN_PROFILE=fast-code.profile cortiq run qwen38-flash-next-fast-code.cmf --prompt-file task.txt --greedy --no-think
```

需要 cortiq 0.8.65 或更高版本。`--no-think` 直接回答；否则模型先推理，请调大 `--max-tokens`。聊天与编辑器请用
`cortiq serve`（兼容 OpenAI 的服务器）保持模型常驻。

12 GB 起的 NVIDIA 显卡通过 CUDA 运行（cortiq 0.8.74 或更高版本，启用 `cuda` 特性构建）：

```bash
cargo install cortiq-cli --locked --features cuda
CMF_GPU=cuda CMF_QWEN_PROFILE=fast-code.profile CMF_QWEN_ROUTE_TAIL=0.05 \
  CMF_SERVE_SLOTS=1 CMF_QWEN_CUDA_REG=1 \
  cortiq serve qwen38-flash-next-fast-code.cmf --port 8080 --host 127.0.0.1
```

### 速度与准确率

一张 RTX PRO 6000（96 GB，Vulkan），单路请求，所有行使用同一 HTTP 客户端与提示，贪心解码，关闭思考，开启草稿。

| | Fast Code | Strata Coder（Strata） | 完整文件（cortiq） |
|---|---:|---:|---:|
| 水族箱需求，贪心：可运行页面 | 一次成功 | 一轮修复后 | 一次成功 |
| HumanEval pass@1（164 题） | 93.3% | 94.5% | 94.5% |
| 2026-10-20 之后代码的困惑度（8192 token） | 3.337 | 3.436¹ | 3.366 |
| Rust 代码困惑度（1536 token） | 4.900 | 4.690¹ | 4.111 |
| 提示 1K / 4K / 16K / 32K token，token/s | 798 / 827 / 844 / 842 | 1916 / 5492 / 7229 / 7659 | 171 / 208 / 206 / 210 |
| 4K / 16K / 32K 提示后的生成，token/s | 161 / 166 / 136 | 220 / 223 / 222 | 118 / 138 / 116 |
| 显存峰值（32K 上下文） | 40.9 GB | 31.1 GB | 59.6 GB |

¹ Strata Coder 权重导入 CMF 后由 cortiq 运行：相同文本与分词器。HumanEval 的 ±3 题属于运行间波动。Fast Code 的速度为两次运行的平均值。

专家选择：两段文本各自归一化、等权合并——450 KB 编程任务与答案（Magicoder OSS-Instruct、Evol-Instruct）以及水族箱需求及其六个回答；覆盖代码路由质量的 83%、任务的 81%。水族箱需求属于选择文本。

![单张 RTX PRO 6000 上的速度](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_speed_zh.png)

![代码准确率](https://huggingface.co/infosave/Qwen3.8-Flash-Next-Fast-Code-cmf/resolve/main/assets/fc_quality_zh.png)

### 12 GB 起的显卡（CUDA）

从 cortiq 0.8.74 起，CUDA 后端可在放不下全部专家的显卡上运行该文件。显卡保留放得下的专家，按路由画像从最常用的开始；
路由到显卡之外的专家由主机 CPU 核心计算（针对 4 位权重的 AVX-512 VNNI 内核），显卡同时计算已保留的专家；文本反复使用的专家会在
token 之间迁入显卡。在 12 GB 内，显卡保留 12 288 个专家中的 1 357 个。

| 显存 | 完整写出水族箱页面，token/s |
|---|---:|
| 12 GB | **104.9** |
| 16 GB | **117.6** |
| 24 GB | **132.8** |

一张 RTX 5090，限制为表中显存（`CMF_GPU_VRAM_MB`），主机为 2× EPYC 9554，容器可用 31 个线程和 62 GB 内存，单路请求，贪心解码，
关闭思考，开启草稿，命令见快速开始。每一行的回答都是完整可运行的页面。在 12 GB 内，HumanEval 前 40 题通过 38 题。主机需要
AVX-512 VNNI（Zen 4、Sapphire Rapids 或更新）和至少 8 个线程；否则每个缺失的专家都要复制到显卡，速度更慢。
`CMF_QWEN_CPU_EXPERT_THREADS` 设置主机线程数。`CMF_QWEN_CUDA_REG=1` 把显卡之外的专家锁定在内存中，主机不必等待磁盘。

### 硬件与长上下文

专家常驻显卡：32K 上下文约 41 GB 显存，另需 28 GB 内存存放 n-gram 表。12 GB 起的显卡通过 CUDA 运行，放不下的专家由主机 CPU 核心计算（见上）；此时文件驻留在内存中（实测 62 GB）。
注意力在显卡上精确支持到 262 144 token；生成速度随长度保持：16K 后 166 token/s，32K 后 136 token/s。
