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
in 41 GB of VRAM instead of 60. It runs with `cortiq`, a Rust inference engine with no Python and no ML framework.

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
Measured on an RTX PRO 6000 (96 GB); on a card with less memory the experts stream from system RAM (not measured
here). `cortiq gpu` lists the adapters; `CMF_GPU_ADAPTER` picks one. On headless Linux set
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
**136–166 токенов/с**, в 41 ГБ видеопамяти вместо 60. Запускается движком `cortiq` на Rust, без Python и ML-фреймворков.

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

### Железо и длинный контекст

Эксперты лежат на карте: около 41 ГБ VRAM при контексте 32K и 28 ГБ оперативной памяти под таблицу n-грамм. На карте
с меньшей памятью эксперты подкачиваются из ОЗУ (здесь не измерялось). Внимание точное до 262 144 токенов; скорость
генерации держится: 166 ток/с после 16K токенов, 136 — после 32K.

---

## 中文文档

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的代码专用裁剪版：每层 512 个路由专家中保留代码与一个大型前端任务最常用的
256 个（4 位），连同共享专家、草稿（MTP）头和 n-gram 表，打包为一个 66.8 GB 的 [CMF](https://github.com/infosave2007/cmf) 文件。
面对 2 151 token 的 3D 水族箱需求说明，它**一次写出可运行的 three.js 页面**（4 次中 4 次）；同一模型的 2 位 Strata Coder 裁剪版需要一轮修复。
专家可完整放入一张 96 GB 显卡：读取提示约 **840 token/s**，是同卡完整文件的 4 倍；生成代码 **136–166 token/s**，显存 41 GB 而非 60 GB。
使用 `cortiq` 运行：Rust 推理引擎，无需 Python 和机器学习框架。

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

### 硬件与长上下文

专家常驻显卡：32K 上下文约 41 GB 显存，另需 28 GB 内存存放 n-gram 表。显存较小的显卡会从内存流式加载专家（此处未测）。
注意力在显卡上精确支持到 262 144 token；生成速度随长度保持：16K 后 166 token/s，32K 后 136 token/s。
