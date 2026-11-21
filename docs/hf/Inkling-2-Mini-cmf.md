---
license: apache-2.0
library_name: cortiq
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - q4tp
  - moe
  - looped-transformer
  - reasoning
  - tool-calling
  - cuda
  - vulkan
  - metal
language:
  - en
  - ru
  - zh
---

# Inkling-2-Mini — CMF

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Thinking Machines' 26.4B mixture-of-experts
model (3.4B parameters active a token; its 12 middle layers run three times, 5.4B effective), packaged as a single
[CMF](https://github.com/infosave2007/cmf) file. It runs with `cortiq`, a Rust inference engine with no Python and no
ML framework: NVIDIA GPUs through CUDA or Vulkan, AMD and Intel GPUs through Vulkan/DX12, Apple silicon through
Metal, and the CPU. The file holds the language model; the release's vision tower and multi-token drafter are not in
it. Measured:

- **14.69 GB, every weight on one 24 GB card.** NVIDIA L4, CUDA: 51.7 tok/s; a 2 048-token prompt is read in
  3.82 s and a 32 768-token one in 111 s, after which it writes 39.1 tok/s in 15.1 GB of the card.
- **The same card over Vulkan:** 18.7 tok/s in 13.7 GB; **MacBook Air M4 (24 GB):** 21.5 tok/s through Metal, 10.3 on
  the CPU alone.
- **Accuracy:** WikiText-2 perplexity 9.057 against 8.944 for the release's own weights in f32 (Transformers), 7.383
  against 7.310 over windows of 4 096 tokens. CUDA, Vulkan and Metal agree within 0.1 %, and five chat questions get
  the same answers word for word on the three.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Inkling-2-Mini-cmf Inkling-2-Mini-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run Inkling-2-Mini-q4tp.cmf --prompt "Why is the sky blue? Answer in three sentences."
```

Use cortiq 0.9.11 or later; prebuilt binaries for Linux, macOS and Windows are on the
[releases page](https://github.com/infosave2007/cmf/releases). Leave out `--features cuda` and `CMF_GPU=cuda` for Vulkan,
Metal or the CPU.

## Files

| file | quantization | size | WikiText-2 ppl | Rust ppl | Python ppl | WikiText-2, 4 096-token windows |
|---|---|---:|---:|---:|---:|---:|
| `Inkling-2-Mini-q4tp.cmf` | 4-bit: routed experts, attention and the other dense matrices; head 8-bit | 14.69 GB | 9.057 | 3.148 | 2.861 | 7.383 |
| reference: Transformers f32 | the release's NVFP4 / MXFP8 experts, decoded | 19.53 GB | 8.944 | 3.124 | 2.851 | 7.310 |

Perplexity over 12 windows of 512 tokens on WikiText-2 test, 1.8 MB of Rust and 477 KB of Python, and over 2 windows
of 4 096 tokens of WikiText-2, the same windows for every row, on CUDA. The release ships its routed experts in NVFP4
(layers 0–19) and MXFP8 (20–23); they are decoded and re-encoded as q4tp. The attention, its output gate and
relative-position projections are GPTQ-rounded on 3 072 tokens of prose, Rust and Python that the evaluation texts do
not hold; the head is 8-bit, the embedding f16, the routers, convolutions, block-mixing queries and message gates
f32. The reference runs the release's layers (transformers PR #51050) in f32 with the experts decoded exactly.

The same file on the other backends:

| backend | WikiText-2 ppl | Python ppl | WikiText-2, 4 096-token windows |
|---|---:|---:|---:|
| CUDA (NVIDIA L4) | 9.057 | 2.861 | 7.383 |
| Vulkan (NVIDIA L4) | 9.057 | — | 7.377 |
| Metal (Apple M4) | 9.055 | 2.866 | 7.380 |

## Performance

One stream, `cortiq bench --core`, cortiq 0.9.11: the time to the first token of a prompt of that length
and the decode speed after it (64 tokens). Card memory is the peak `nvidia-smi` shows during the run. Read each
backend's section on its own.

### CUDA · NVIDIA L4

Stand: NVIDIA L4 (24 GB), Google Colab, the CUDA backend (`CMF_GPU=cuda`). Without a prompt: 51.7 tok/s. A
prompt is read 512 rows a call: the weights through cuBLAS, the routed experts grouped so that each is read once a
call.

![Inkling-2-Mini on CUDA](https://huggingface.co/infosave/Inkling-2-Mini-cmf/resolve/main/assets/speed_cuda_en.png)

| prompt, tokens | first token, s | decode, tok/s | peak card memory, GB |
|---:|---:|---:|---:|
| 512 | 0.93 | 49.6 | 14.0 |
| 2 048 | 3.82 | 48.5 | 14.1 |
| 8 192 | 18.2 | 45.9 | 14.3 |
| 16 384 | 42.7 | 43.5 | 14.5 |
| 32 768 | 111.0 | 39.1 | 15.1 |

### Vulkan · NVIDIA L4

Stand: the same NVIDIA L4 (24 GB), over Vulkan (the default backend). Without a prompt: 18.7 tok/s.

![Inkling-2-Mini on Vulkan](https://huggingface.co/infosave/Inkling-2-Mini-cmf/resolve/main/assets/speed_vulkan_en.png)

| prompt, tokens | first token, s | decode, tok/s | peak card memory, GB |
|---:|---:|---:|---:|
| 512 | 3.99 | 17.5 | 13.7 |
| 2 048 | 17.9 | 16.5 | 13.7 |
| 8 192 | 79.7 | 14.5 | 14.6 |
| 16 384 | 188.0 | 13.0 | 15.5 |

### Metal · MacBook Air M4

Stand: MacBook Air M4 (24 GB), fanless, Metal. Without a prompt: 21.5 tok/s. The file stays memory-mapped; a prompt
is read 256 rows a call, the routed experts as simdgroup GEMMs.

![Inkling-2-Mini on Metal](https://huggingface.co/infosave/Inkling-2-Mini-cmf/resolve/main/assets/speed_metal_en.png)

| prompt, tokens | first token, s | decode, tok/s |
|---:|---:|---:|
| 512 | 3.70 | 20.6 |
| 2 048 | 18.2 | 19.3 |
| 8 192 | 86.4 | 17.0 |

### CPU

| stand | decode, tok/s | 512-token prompt | decode after it |
|---|---:|---:|---:|
| MacBook Air M4 (10 cores, 24 GB), `CMF_GPU=0` | 10.3 | 22.6 s | 9.7 |
| Colab L4 host (Xeon 2.2 GHz, 6 cores / 12 threads), `CMF_GPU=0` | 2.9 | 70.1 s | 2.8 |

## Usage

```bash
cortiq run Inkling-2-Mini-q4tp.cmf                                   # interactive chat
cortiq run Inkling-2-Mini-q4tp.cmf --prompt "..."                    # one answer, with its reasoning
cortiq run Inkling-2-Mini-q4tp.cmf --prompt "..." --no-think         # the least reasoning
cortiq run Inkling-2-Mini-q4tp.cmf --prompt "..." --reasoning-effort low
cortiq run Inkling-2-Mini-q4tp.cmf --prompt-file task.txt --max-tokens 2048
```

The embedded template is the model's own: a turn opens with a thinking-effort line (`high` by default; `none`,
`minimal`, `low`, `medium`, `xhigh` or a number from 0 to 0.99 with `--reasoning-effort`), and the model writes a
thinking part before its answer, shown between `<think>` and `</think>`. `--no-think` asks for effort `none`; as the
release notes, the model may still reason a little in its answer. The file carries the release's sampling
(temperature 1.0); `--greedy` gives deterministic output.

### OpenAI-compatible server

`cortiq serve` puts the file behind an OpenAI-compatible HTTP API, with a web dashboard on the same port.

```bash
CMF_GPU=cuda cortiq serve Inkling-2-Mini-q4tp.cmf --port 8080 --host 127.0.0.1

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Plan a three-step migration from MySQL to Postgres."}],
  "max_tokens": 1024,
  "reasoning_effort": "low"
}'
```

| endpoint | what it does |
|---|---|
| `POST /v1/chat/completions` | a chat turn; `stream: true` sends the answer as it is decoded, `tools` lets the model call functions |
| `POST /v1/completions` | the continuation of a `prompt` |
| `GET /v1/models` | the served model |
| `GET /` | the web dashboard |

- The thinking part comes back as `message.reasoning_content`; `"enable_thinking": false` asks for effort `none`,
  `reasoning_effort` sets it as in the CLI.
- Tools go in `tools` in the OpenAI format; the model's JSON tool calls come back as `message.tool_calls`.
- One sequence at a time: the server runs this file with one slot. `--host 127.0.0.1` keeps it local; no API key is
  checked.

## Long context

The model holds 1 048 576 positions; cortiq caps a conversation at 32 768 by default (`CMF_MAX_SEQ` sets it). It has
no rotary positions: attention adds a learned relative bias over the last 1 024 distances, 20 of its 24 layers
attend over a 1 024-token window, and past 8 192 positions the 4 full layers scale their queries with the logarithm
of the position. Each
of the 48 layer applications keeps its own keys and values: on CUDA the windowed ones a ring of 1 536 rows, the 8 full
ones 32 KB a token (1.33 GB of the card for 32 768 positions).

## Verify the download

```bash
sha256sum -c Inkling-2-Mini-q4tp.cmf.sha256
cortiq info Inkling-2-Mini-q4tp.cmf
```

| file | sha256 |
|---|---|
| `Inkling-2-Mini-q4tp.cmf` | `f3560aeb6bc98b608629a1454037f88607a217d7d1a568c6257dfb1b941d45c8` |

## License

Apache-2.0, as the original release ([LICENSE](LICENSE)); use of the model is also subject to Thinking Machines'
[acceptable use policy](https://thinkingmachines.ai/model-acceptable-use-policy/). The weights were converted to CMF
and quantized by cortiq. The original model card's notes on intended use, limitations and safety apply to this file
as well.

---

## Документация на русском

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) — модель Thinking Machines со смесью
экспертов на 26,4 млрд параметров (3,4 млрд активных на токен; 12 средних слоёв выполняются трижды, 5,4 млрд
эффективных), упакованная в один файл [CMF](https://github.com/infosave2007/cmf). Его запускает `cortiq` — движок на
Rust без Python и ML-фреймворков: GPU NVIDIA через CUDA или Vulkan, AMD и Intel через Vulkan/DX12, Apple silicon через
Metal, а также CPU. В файле языковая модель; визуального энкодера и многотокенного драфтера релиза в нём нет.
Измерено:

- **14,69 ГБ, все веса на одной карте 24 ГБ.** NVIDIA L4, CUDA: 51,7 ток/с; промпт из 2 048 токенов читается за
  3,82 с, из 32 768 — за 111 с, после чего модель пишет 39,1 ток/с в 15,1 ГБ карты.
- **Та же карта через Vulkan:** 18,7 ток/с в 13,7 ГБ; **MacBook Air M4 (24 ГБ):** 21,5 ток/с через Metal, 10,3 на
  одном CPU.
- **Точность:** перплексия WikiText-2 9,057 против 8,944 у собственных весов релиза в f32 (Transformers), 7,383
  против 7,310 на окнах по 4 096 токенов. CUDA, Vulkan и Metal расходятся не больше чем на 0,1 %, а на пять вопросов в
  чате все три отвечают слово в слово одинаково.

### Быстрый старт

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Inkling-2-Mini-cmf Inkling-2-Mini-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run Inkling-2-Mini-q4tp.cmf --prompt "Почему небо голубое? Ответь в трёх предложениях."
```

Нужен cortiq 0.9.11 или новее; готовые сборки для Linux, macOS и Windows — на
[странице релизов](https://github.com/infosave2007/cmf/releases). Для Vulkan, Metal или CPU уберите `--features cuda` и
`CMF_GPU=cuda`.

### Файлы

| файл | квантование | размер | ppl WikiText-2 | ppl Rust | ppl Python | WikiText-2, окна по 4 096 токенов |
|---|---|---:|---:|---:|---:|---:|
| `Inkling-2-Mini-q4tp.cmf` | 4 бита: маршрутизируемые эксперты, внимание и остальные плотные матрицы; голова 8 бит | 14,69 ГБ | 9,057 | 3,148 | 2,861 | 7,383 |
| эталон: Transformers f32 | эксперты релиза в NVFP4 / MXFP8, раскодированные | 19,53 ГБ | 8,944 | 3,124 | 2,851 | 7,310 |

Перплексия на 12 окнах по 512 токенов WikiText-2 test, 1,8 МБ кода на Rust и 477 КБ на Python и на 2 окнах по 4 096
токенов WikiText-2, окна одинаковые для всех строк, CUDA. Релиз хранит маршрутизируемых экспертов в NVFP4 (слои 0–19)
и MXFP8 (20–23); они раскодированы и заново закодированы в q4tp. Внимание, его выходной гейт и проекции
относительных позиций округлены GPTQ на 3 072 токенах прозы, Rust и Python, которых нет в оценочных текстах; голова
8-битная, эмбеддинг f16, роутеры, свёртки, запросы смешивания блоков и гейты сообщений — f32. Эталон — слои релиза
(transformers PR #51050) в f32 с точно раскодированными экспертами.

Тот же файл на других бэкендах:

| бэкенд | ppl WikiText-2 | ppl Python | WikiText-2, окна по 4 096 токенов |
|---|---:|---:|---:|
| CUDA (NVIDIA L4) | 9,057 | 2,861 | 7,383 |
| Vulkan (NVIDIA L4) | 9,057 | — | 7,377 |
| Metal (Apple M4) | 9,055 | 2,866 | 7,380 |

### Производительность

Один поток, `cortiq bench --core`, cortiq 0.9.11: время до первого токена после промпта такой длины и скорость
генерации после него (64 токена). Память карты — пик по `nvidia-smi` за прогон. Каждый раздел — свой бэкенд.

#### CUDA · NVIDIA L4

Стенд: NVIDIA L4 (24 ГБ), Google Colab, бэкенд CUDA (`CMF_GPU=cuda`). Без промпта: 51,7 ток/с. Промпт читается
по 512 строк за вызов: веса через cuBLAS, маршрутизируемые эксперты сгруппированы, так что каждый читается один раз
за вызов.

![Inkling-2-Mini на CUDA](https://huggingface.co/infosave/Inkling-2-Mini-cmf/resolve/main/assets/speed_cuda_ru.png)

| промпт, токенов | первый токен, с | генерация, ток/с | пик памяти карты, ГБ |
|---:|---:|---:|---:|
| 512 | 0,93 | 49,6 | 14,0 |
| 2 048 | 3,82 | 48,5 | 14,1 |
| 8 192 | 18,2 | 45,9 | 14,3 |
| 16 384 | 42,7 | 43,5 | 14,5 |
| 32 768 | 111,0 | 39,1 | 15,1 |

#### Vulkan · NVIDIA L4

Стенд: та же NVIDIA L4 (24 ГБ) через Vulkan (бэкенд по умолчанию). Без промпта: 18,7 ток/с.

![Inkling-2-Mini на Vulkan](https://huggingface.co/infosave/Inkling-2-Mini-cmf/resolve/main/assets/speed_vulkan_ru.png)

| промпт, токенов | первый токен, с | генерация, ток/с | пик памяти карты, ГБ |
|---:|---:|---:|---:|
| 512 | 3,99 | 17,5 | 13,7 |
| 2 048 | 17,9 | 16,5 | 13,7 |
| 8 192 | 79,7 | 14,5 | 14,6 |
| 16 384 | 188,0 | 13,0 | 15,5 |

#### Metal · MacBook Air M4

Стенд: MacBook Air M4 (24 ГБ), без вентилятора, Metal. Без промпта: 21,5 ток/с. Файл остаётся отображённым в память;
промпт читается по 256 строк за вызов, маршрутизируемые эксперты — GEMM на simdgroup-матрицах.

![Inkling-2-Mini на Metal](https://huggingface.co/infosave/Inkling-2-Mini-cmf/resolve/main/assets/speed_metal_ru.png)

| промпт, токенов | первый токен, с | генерация, ток/с |
|---:|---:|---:|
| 512 | 3,70 | 20,6 |
| 2 048 | 18,2 | 19,3 |
| 8 192 | 86,4 | 17,0 |

#### CPU

| стенд | генерация, ток/с | промпт 512 токенов | генерация после него |
|---|---:|---:|---:|
| MacBook Air M4 (10 ядер, 24 ГБ), `CMF_GPU=0` | 10,3 | 22,6 с | 9,7 |
| хост Colab L4 (Xeon 2,2 ГГц, 6 ядер / 12 потоков), `CMF_GPU=0` | 2,9 | 70,1 с | 2,8 |

### Использование

```bash
cortiq run Inkling-2-Mini-q4tp.cmf                                   # интерактивный чат
cortiq run Inkling-2-Mini-q4tp.cmf --prompt "..."                    # один ответ с рассуждением
cortiq run Inkling-2-Mini-q4tp.cmf --prompt "..." --no-think         # минимум рассуждения
cortiq run Inkling-2-Mini-q4tp.cmf --prompt "..." --reasoning-effort low
cortiq run Inkling-2-Mini-q4tp.cmf --prompt-file task.txt --max-tokens 2048
```

Встроенный шаблон — собственный шаблон модели: ход начинается со строки об усилии рассуждения (по умолчанию `high`;
`none`, `minimal`, `low`, `medium`, `xhigh` или число от 0 до 0,99 через `--reasoning-effort`), и модель пишет часть
рассуждения перед ответом — она выводится между `<think>` и `</think>`. `--no-think` просит усилие `none`; как
отмечает релиз, модель всё равно может немного рассуждать в ответе. Файл несёт сэмплирование релиза (температура
1,0); `--greedy` даёт детерминированный вывод.

#### OpenAI-совместимый сервер

`cortiq serve` отдаёт файл через HTTP API, совместимый с OpenAI, с веб-панелью на том же порту.

```bash
CMF_GPU=cuda cortiq serve Inkling-2-Mini-q4tp.cmf --port 8080 --host 127.0.0.1

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Составь план миграции с MySQL на Postgres в три шага."}],
  "max_tokens": 1024,
  "reasoning_effort": "low"
}'
```

| эндпоинт | что делает |
|---|---|
| `POST /v1/chat/completions` | ход чата; `stream: true` присылает ответ по мере генерации, `tools` позволяет модели вызывать функции |
| `POST /v1/completions` | продолжение `prompt` |
| `GET /v1/models` | обслуживаемая модель |
| `GET /` | веб-панель |

- Часть рассуждения приходит в `message.reasoning_content`; `"enable_thinking": false` просит усилие `none`,
  `reasoning_effort` задаёт его, как в CLI.
- Инструменты передаются в `tools` в формате OpenAI; JSON-вызовы модели возвращаются в `message.tool_calls`.
- Одна последовательность за раз: сервер запускает этот файл с одним слотом. `--host 127.0.0.1` оставляет сервер
  локальным; ключ API не проверяется.

### Длинный контекст

Модель держит 1 048 576 позиций; cortiq по умолчанию ограничивает разговор 32 768 токенами (задаётся `CMF_MAX_SEQ`).
Ротационных позиций у неё нет: внимание добавляет обученный относительный сдвиг по последним 1 024 расстояниям, 20 из
24 слоёв смотрят в окно 1 024 токена, а 4 полных слоя после 8 192 позиций масштабируют запросы логарифмом позиции.
Каждое из 48 применений слоёв хранит свои ключи и значения: на CUDA оконные — кольцо из 1 536 строк, 8 полных — 32 КБ
на токен (1,33 ГБ карты на 32 768 позиций).

### Проверка загрузки

```bash
sha256sum -c Inkling-2-Mini-q4tp.cmf.sha256
cortiq info Inkling-2-Mini-q4tp.cmf
```

| файл | sha256 |
|---|---|
| `Inkling-2-Mini-q4tp.cmf` | `f3560aeb6bc98b608629a1454037f88607a217d7d1a568c6257dfb1b941d45c8` |

### Лицензия

Apache-2.0, как у исходного релиза ([LICENSE](LICENSE)); использование модели также подчиняется
[политике допустимого использования](https://thinkingmachines.ai/model-acceptable-use-policy/) Thinking Machines.
Веса сконвертированы в CMF и квантованы cortiq. Замечания исходной карточки о назначении, ограничениях и безопасности
относятся и к этому файлу.

---

## 中文文档

[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) 是 Thinking Machines 的 264 亿参数混合专家模型
（每个 token 激活 34 亿参数；中间 12 层运行三遍，等效 54 亿），这里打包为单个 [CMF](https://github.com/infosave2007/cmf)
文件，由 `cortiq` 运行——一个不依赖 Python 和机器学习框架的 Rust 推理引擎：NVIDIA 显卡走 CUDA 或 Vulkan，AMD、Intel
显卡走 Vulkan/DX12，Apple silicon 走 Metal，也可在 CPU 上运行。文件包含语言模型；发布版的视觉编码器和多 token 草稿头不在其中。实测：

- **14.69 GB，全部权重放在一块 24 GB 显卡上。** NVIDIA L4，CUDA：51.7 tok/s；2 048 token 的提示词读取 3.82 秒，
  32 768 token 读取 111 秒，之后以 39.1 tok/s 生成，占用 15.1 GB 显存。
- **同一块显卡，Vulkan：** 18.7 tok/s，占用 13.7 GB；**MacBook Air M4（24 GB）：** Metal 21.5 tok/s，仅 CPU 10.3。
- **精度：** WikiText-2 困惑度 9.057，发布版自身权重在 f32（Transformers）下为 8.944；4 096 token 窗口上 7.383，参考为
  7.310。CUDA、Vulkan 和 Metal 之间相差不超过 0.1%，五个对话问题在三者上的回答逐词一致。

### 快速开始

```bash
cargo install cortiq-cli --features cuda
hf download infosave/Inkling-2-Mini-cmf Inkling-2-Mini-q4tp.cmf --local-dir .
CMF_GPU=cuda cortiq run Inkling-2-Mini-q4tp.cmf --prompt "天空为什么是蓝色的？用三句话回答。"
```

需要 cortiq 0.9.11 或更高版本；Linux、macOS 和 Windows 的预编译程序见[发布页面](https://github.com/infosave2007/cmf/releases)。
使用 Vulkan、Metal 或 CPU 时去掉 `--features cuda` 和 `CMF_GPU=cuda`。

### 文件

| 文件 | 量化 | 大小 | WikiText-2 困惑度 | Rust 困惑度 | Python 困惑度 | WikiText-2，4 096 token 窗口 |
|---|---|---:|---:|---:|---:|---:|
| `Inkling-2-Mini-q4tp.cmf` | 4 位：路由专家、注意力及其他稠密矩阵；输出头 8 位 | 14.69 GB | 9.057 | 3.148 | 2.861 | 7.383 |
| 参考：Transformers f32 | 发布版的 NVFP4 / MXFP8 专家，解码后 | 19.53 GB | 8.944 | 3.124 | 2.851 | 7.310 |

困惑度在 WikiText-2 test 的 12 个 512 token 窗口、1.8 MB Rust 和 477 KB Python 代码上计算，另有 WikiText-2 的 2 个
4 096 token 窗口，各行使用相同窗口，CUDA。发布版的路由专家以 NVFP4（第 0–19 层）和 MXFP8（第 20–23 层）存储，这里解码后
重新编码为 q4tp。注意力、其输出门和相对位置投影用 GPTQ 在 3 072 个 token 的散文、Rust 和 Python 上取整（评测文本中不含
这些内容）；输出头为 8 位，词嵌入为 f16，路由器、卷积、块混合查询和消息门为 f32。参考在 f32 下运行发布版的各层
（transformers PR #51050），专家精确解码。

同一文件在其他后端上：

| 后端 | WikiText-2 困惑度 | Python 困惑度 | WikiText-2，4 096 token 窗口 |
|---|---:|---:|---:|
| CUDA（NVIDIA L4） | 9.057 | 2.861 | 7.383 |
| Vulkan（NVIDIA L4） | 9.057 | — | 7.377 |
| Metal（Apple M4） | 9.055 | 2.866 | 7.380 |

### 性能

单路，`cortiq bench --core`，cortiq 0.9.11：给定长度提示词后的首 token 时间及其后的生成速度（64 个 token）。
显存为运行期间 `nvidia-smi` 显示的峰值。各后端的小节请分别阅读。

#### CUDA · NVIDIA L4

测试平台：NVIDIA L4（24 GB），Google Colab，CUDA 后端（`CMF_GPU=cuda`）。无提示词时：51.7 tok/s。提示词每次调用读取
512 行：权重走 cuBLAS，路由专家按专家分组，每次调用每个专家只读一次。

![Inkling-2-Mini 在 CUDA 上](https://huggingface.co/infosave/Inkling-2-Mini-cmf/resolve/main/assets/speed_cuda_zh.png)

| 提示词，token | 首 token，秒 | 生成，tok/s | 显存峰值，GB |
|---:|---:|---:|---:|
| 512 | 0.93 | 49.6 | 14.0 |
| 2 048 | 3.82 | 48.5 | 14.1 |
| 8 192 | 18.2 | 45.9 | 14.3 |
| 16 384 | 42.7 | 43.5 | 14.5 |
| 32 768 | 111.0 | 39.1 | 15.1 |

#### Vulkan · NVIDIA L4

测试平台：同一块 NVIDIA L4（24 GB），Vulkan（默认后端）。无提示词时：18.7 tok/s。

![Inkling-2-Mini 在 Vulkan 上](https://huggingface.co/infosave/Inkling-2-Mini-cmf/resolve/main/assets/speed_vulkan_zh.png)

| 提示词，token | 首 token，秒 | 生成，tok/s | 显存峰值，GB |
|---:|---:|---:|---:|
| 512 | 3.99 | 17.5 | 13.7 |
| 2 048 | 17.9 | 16.5 | 13.7 |
| 8 192 | 79.7 | 14.5 | 14.6 |
| 16 384 | 188.0 | 13.0 | 15.5 |

#### Metal · MacBook Air M4

测试平台：MacBook Air M4（24 GB），无风扇，Metal。无提示词时：21.5 tok/s。文件保持内存映射；提示词每次调用读取 256 行，
路由专家以 simdgroup 矩阵 GEMM 计算。

![Inkling-2-Mini 在 Metal 上](https://huggingface.co/infosave/Inkling-2-Mini-cmf/resolve/main/assets/speed_metal_zh.png)

| 提示词，token | 首 token，秒 | 生成，tok/s |
|---:|---:|---:|
| 512 | 3.70 | 20.6 |
| 2 048 | 18.2 | 19.3 |
| 8 192 | 86.4 | 17.0 |

#### CPU

| 测试平台 | 生成，tok/s | 512 token 提示词 | 其后生成 |
|---|---:|---:|---:|
| MacBook Air M4（10 核，24 GB），`CMF_GPU=0` | 10.3 | 22.6 秒 | 9.7 |
| Colab L4 主机（Xeon 2.2 GHz，6 核 / 12 线程），`CMF_GPU=0` | 2.9 | 70.1 秒 | 2.8 |

### 使用

```bash
cortiq run Inkling-2-Mini-q4tp.cmf                                   # 交互式对话
cortiq run Inkling-2-Mini-q4tp.cmf --prompt "..."                    # 单次回答，含推理
cortiq run Inkling-2-Mini-q4tp.cmf --prompt "..." --no-think         # 最少推理
cortiq run Inkling-2-Mini-q4tp.cmf --prompt "..." --reasoning-effort low
cortiq run Inkling-2-Mini-q4tp.cmf --prompt-file task.txt --max-tokens 2048
```

内置模板是模型自己的：每轮以一行推理强度开头（默认 `high`；可用 `--reasoning-effort` 设为 `none`、`minimal`、`low`、
`medium`、`xhigh` 或 0 到 0.99 的数字），模型在回答前先写推理部分，显示在 `<think>` 与 `</think>` 之间。`--no-think`
请求强度 `none`；如发布版所述，模型在回答中仍可能有少量推理。文件带有发布版的采样设置（温度 1.0）；`--greedy` 给出确定性输出。

#### OpenAI 兼容服务器

`cortiq serve` 以 OpenAI 兼容的 HTTP API 提供该文件，同一端口上还有网页面板。

```bash
CMF_GPU=cuda cortiq serve Inkling-2-Mini-q4tp.cmf --port 8080 --host 127.0.0.1

curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "制定一个从 MySQL 迁移到 Postgres 的三步计划。"}],
  "max_tokens": 1024,
  "reasoning_effort": "low"
}'
```

| 端点 | 作用 |
|---|---|
| `POST /v1/chat/completions` | 一轮对话；`stream: true` 边生成边返回，`tools` 让模型调用函数 |
| `POST /v1/completions` | 续写 `prompt` |
| `GET /v1/models` | 所服务的模型 |
| `GET /` | 网页面板 |

- 推理部分在 `message.reasoning_content` 中返回；`"enable_thinking": false` 请求强度 `none`，`reasoning_effort` 的设置与 CLI 相同。
- 工具以 OpenAI 格式放在 `tools` 中；模型的 JSON 工具调用以 `message.tool_calls` 返回。
- 一次一个序列：服务器以单槽运行该文件。`--host 127.0.0.1` 使服务器只在本机可访问；不校验 API key。

### 长上下文

模型支持 1 048 576 个位置；cortiq 默认将一次对话限制为 32 768 个 token（用 `CMF_MAX_SEQ` 设置）。模型没有旋转位置编码：
注意力在最近 1 024 个距离上加入学习得到的相对偏置，24 层中有 20 层在 1 024 token 的窗口内注意，4 个全注意力层在 8 192 个
位置之后按位置的对数缩放查询。48 次层应用各自保留键和值：在 CUDA 上，窗口层为 1 536 行的环形缓冲，8 个全注意力应用每个
token 32 KB（32 768 个位置占 1.33 GB 显存）。

### 校验下载

```bash
sha256sum -c Inkling-2-Mini-q4tp.cmf.sha256
cortiq info Inkling-2-Mini-q4tp.cmf
```

| 文件 | sha256 |
|---|---|
| `Inkling-2-Mini-q4tp.cmf` | `f3560aeb6bc98b608629a1454037f88607a217d7d1a568c6257dfb1b941d45c8` |

### 许可证

Apache-2.0，与原始发布版相同（[LICENSE](LICENSE)）；模型的使用还须遵守 Thinking Machines 的
[可接受使用政策](https://thinkingmachines.ai/model-acceptable-use-policy/)。权重由 cortiq 转换为 CMF 并量化。原始模型卡中
关于预期用途、局限性和安全性的说明同样适用于本文件。
