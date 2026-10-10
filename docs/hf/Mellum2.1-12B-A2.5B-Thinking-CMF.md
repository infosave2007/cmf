---
license: apache-2.0
library_name: cortiq
base_model: JetBrains/Mellum2.1-12B-A2.5B-Thinking
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - moe
  - code
  - reasoning
  - tool-calling
  - vulkan
  - metal
language:
  - en
---

# Mellum2.1 12B-A2.5B Thinking — CMF

[JetBrains Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)
— a code-focused reasoning model (mixture of experts: 12.15B parameters, 2.5B
active per token, 64 experts with top-8 routing; 21 sliding-window and 7
full-attention layers; 131k-token context; tool calling) — packaged as single
[CMF](https://github.com/infosave2007/cmf) files. They run with `cortiq`, a
Rust inference engine with no Python and no ML framework: NVIDIA, AMD and Intel
GPUs through Vulkan/DX12, Apple silicon through Metal, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/Mellum2.1-12B-A2.5B-Thinking-CMF mellum2.1-12b-a2.5b-thinking-q4tp.cmf --local-dir .
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "Write a Python function that merges two sorted lists." --no-think
```

Use cortiq 0.8.15 or later; prebuilt binaries for Linux, macOS and Windows are
on the [releases page](https://github.com/infosave2007/cmf/releases). The
engine picks the GPU and its settings itself. Without `--no-think` the model
reasons in a `<think>` block before answering, so raise `--max-tokens`
(default 256).

## Files

| file | experts | size | code ppl | wiki ppl | top-1 agreement | recommended for |
|---|---|---:|---:|---:|---:|---|
| `mellum2.1-12b-a2.5b-thinking-q4tp.cmf` | 4-bit | 6.88 GB | 3.242 | 7.407 | 93.4 % | **default**: fastest on every GPU, fits 8 GB of VRAM |
| `mellum2.1-12b-a2.5b-thinking-q8_2f.cmf` | 8-bit | 12.22 GB | 3.146 | 7.180 | 98.1 % | near-lossless; CPU, or a GPU with 16 GB+ (see below) |

Attention, embeddings and the output head are 8-bit in both files; the router
and norms stay 16-bit. Quality is measured against the unquantized model run
by the same engine (code ppl 3.182, wiki ppl 7.120): perplexity over 2048
tokens of Python standard-library code and of wikitext-2 test, and the share of
positions where the file's top prediction matches the unquantized model's.

## Performance

`cortiq bench --core --ignore-eos`, one stream, medians of three runs, cortiq 0.8.15:

| hardware | backend | file | decode, tok/s | decode at 3-4k tokens | 3-4k-token prompt, tok/s | time to first token |
|---|---|---|---:|---:|---:|---:|
| RTX 3090 (24 GB) | Vulkan | `q4tp` | 150 | 129 | 382 | 10.5 s |
| RTX 3090 (24 GB) | Vulkan | `q8_2f` | 140 | 132 | 547 | 7.3 s |
| Mac mini M4 (24 GB) | Metal | `q4tp` | 46.8 | 40.9 | 413 | 7.3 s |
| Mac mini M4 (24 GB) | Metal | `q8_2f` | 35.6 | 31.2 | 318 | 9.8 s |
| Mac mini M4 (24 GB) | CPU, 10 cores | `q4tp` | 34.9 | 17.3 | 110 | 31.7 s |

The RTX 3090 rows use a 4000-token prompt; the M4 rows use 3000. On Vulkan the `q8_2f` decode
kernels need 32-lane subgroups (NVIDIA); other GPUs run its prompt on the GPU and decode op by op.

## Usage

```bash
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf                             # interactive chat
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "..."              # one answer, with reasoning
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "..." --no-think   # answer without the thinking block
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "..." --greedy     # deterministic output
```

### Sampling

JetBrains' examples use temperature 0.6 and top-p 0.95:

```bash
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --temperature 0.6 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

Without flags the CLI uses temperature 0.7, top-p 0.9, top-k 40, min-p 0.05,
repetition penalty 1.1 and at most 256 new tokens.

### OpenAI-compatible server

```bash
cortiq serve mellum2.1-12b-a2.5b-thinking-q4tp.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mellum", "temperature": 0.6, "top_p": 0.95,
       "messages": [{"role": "user", "content": "Implement stable merge sort in Python."}]}'
```

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, plus a web
dashboard on the same port. `--host 127.0.0.1` keeps the server local. The reply
reads `<think>…</think>answer`; `"enable_thinking": false` gives a direct
answer.

### Tool calling

Send `tools` in the usual OpenAI format; the embedded template lists them the
way the model was trained, and the model answers with
`<tool_call>{"name": …, "arguments": …}</tool_call>`. The server returns that as
`message.tool_calls` with `finish_reason: "tool_calls"`, streamed or not. In 6
of 6 test conversations the call and the answer from its result were correct,
with and without the thinking block (Metal, `q4tp` file).

## Hardware

**Apple silicon (Metal).** Runs on the GPU out of the box. The `q4tp` file
needs about 8 GB of unified memory.

**Choosing a GPU (Vulkan).** `cortiq gpu` lists the adapters;
`CMF_GPU_ADAPTER=1` (index or name substring) picks one. If the list is empty
on Linux, install `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`; on headless
machines set `XDG_RUNTIME_DIR=/tmp`.

**CPU only.** `CMF_GPU=0` keeps everything on the CPU.

## Long context

The model supports 131 072 tokens; the KV cache holds 32 768 by default. Raise
it with `CMF_MAX_SEQ`:

```bash
CMF_MAX_SEQ=131072 cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "..." --max-tokens 4096
```

The cache takes 115 KB per token: 3.7 GB at 32k tokens.

## Verify the download

```bash
sha256sum -c mellum2.1-12b-a2.5b-thinking-q4tp.cmf.sha256
cortiq verify mellum2.1-12b-a2.5b-thinking-q4tp.cmf
cortiq info mellum2.1-12b-a2.5b-thinking-q4tp.cmf
```

Weights derive from JetBrains' release and remain under its Apache-2.0 terms.
Review generated code before running it.

---

## Документация на русском

[JetBrains Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)
— модель для кода с рассуждениями (смесь экспертов: 12.15 млрд параметров,
2.5 млрд активных на токен, 64 эксперта с выбором top-8; 21 слой со скользящим
окном и 7 слоёв с полным вниманием; контекст 131k токенов; вызов инструментов)
в виде отдельных файлов [CMF](https://github.com/infosave2007/cmf). Запускается
движком `cortiq` на Rust без Python и ML-фреймворков: GPU NVIDIA, AMD и Intel
через Vulkan/DX12, Apple silicon через Metal, а также CPU.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/Mellum2.1-12B-A2.5B-Thinking-CMF mellum2.1-12b-a2.5b-thinking-q4tp.cmf --local-dir .
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "Напиши функцию на Python, которая сливает два отсортированных списка." --no-think
```

Используйте cortiq 0.8.15 или новее; готовые сборки для Linux, macOS и Windows —
на [странице релизов](https://github.com/infosave2007/cmf/releases). Движок сам
выбирает GPU и настройки. Без `--no-think` модель сначала рассуждает в блоке
`<think>`, поэтому увеличьте `--max-tokens` (по умолчанию 256).

### Файлы

| файл | эксперты | размер | ppl код | ppl wiki | совпадение top-1 | назначение |
|---|---|---:|---:|---:|---:|---|
| `mellum2.1-12b-a2.5b-thinking-q4tp.cmf` | 4 бита | 6.88 ГБ | 3.242 | 7.407 | 93.4 % | **по умолчанию**: самый быстрый на любом GPU, помещается в 8 ГБ видеопамяти |
| `mellum2.1-12b-a2.5b-thinking-q8_2f.cmf` | 8 бит | 12.22 ГБ | 3.146 | 7.180 | 98.1 % | почти без потерь; CPU или GPU от 16 ГБ (см. ниже) |

Внимание, эмбеддинги и выходная голова в обоих файлах 8-битные; роутер и
нормализации — 16 бит. Качество измерено относительно неквантованной модели на
том же движке (ppl кода 3.182, wiki 7.120): перплексия на 2048 токенах кода
стандартной библиотеки Python и wikitext-2 test, а также доля позиций, где
лучший прогноз файла совпадает с прогнозом неквантованной модели.

### Скорость

`cortiq bench --core --ignore-eos`, один поток, медианы трёх запусков, cortiq 0.8.15:

| железо | бэкенд | файл | генерация, ток/с | генерация на 3-4k токенах | промпт 3-4k токенов, ток/с | время до первого токена |
|---|---|---|---:|---:|---:|---:|
| RTX 3090 (24 ГБ) | Vulkan | `q4tp` | 150 | 129 | 382 | 10.5 с |
| RTX 3090 (24 ГБ) | Vulkan | `q8_2f` | 140 | 132 | 547 | 7.3 с |
| Mac mini M4 (24 ГБ) | Metal | `q4tp` | 46.8 | 40.9 | 413 | 7.3 с |
| Mac mini M4 (24 ГБ) | Metal | `q8_2f` | 35.6 | 31.2 | 318 | 9.8 с |
| Mac mini M4 (24 ГБ) | CPU, 10 ядер | `q4tp` | 34.9 | 17.3 | 110 | 31.7 с |

Строки RTX 3090 — с промптом 4000 токенов, строки M4 — 3000. На Vulkan ядрам генерации `q8_2f`
нужны subgroup по 32 линии (NVIDIA); на других GPU промпт идёт на GPU, а генерация — по операциям.

### Использование

```bash
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf                             # интерактивный чат
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "..."              # один ответ с рассуждением
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "..." --no-think   # ответ без блока рассуждений
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "..." --greedy     # детерминированный вывод
```

В примерах JetBrains используются temperature 0.6 и top-p 0.95:

```bash
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --temperature 0.6 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

Без флагов CLI использует temperature 0.7, top-p 0.9, top-k 40, min-p 0.05,
repetition penalty 1.1 и не более 256 новых токенов.

**Сервер с OpenAI-совместимым API:**

```bash
cortiq serve mellum2.1-12b-a2.5b-thinking-q4tp.cmf --port 8080
```

Эндпоинты `/v1/chat/completions`, `/v1/completions`, `/v1/models` и веб-панель
на том же порту. `--host 127.0.0.1` оставляет сервер локальным. Ответ имеет вид
`<think>…</think>ответ`; `"enable_thinking": false` даёт прямой ответ.

**Вызов инструментов.** Передавайте `tools` в обычном формате OpenAI; модель
отвечает `<tool_call>{"name": …, "arguments": …}</tool_call>`, сервер возвращает
это как `message.tool_calls` с `finish_reason: "tool_calls"`, в потоковом режиме
тоже. В 6 из 6 тестовых диалогов вызов и ответ по результату были верны, с
рассуждениями и без (Metal, файл `q4tp`).

### Железо

**Apple silicon (Metal).** Работает на GPU без настройки; файлу `q4tp` нужно
около 8 ГБ общей памяти.

**Выбор GPU (Vulkan).** `cortiq gpu` показывает адаптеры; `CMF_GPU_ADAPTER=1`
(индекс или часть имени) закрепляет нужный. Если на Linux список пуст,
установите `libvulkan1 libglvnd0 libegl1 libgl1 libglx0`; на машинах без дисплея
задайте `XDG_RUNTIME_DIR=/tmp`.

**Только CPU.** `CMF_GPU=0` оставляет всё на процессоре.

### Длинный контекст

Модель поддерживает 131 072 токена; KV-кэш по умолчанию рассчитан на 32 768.
Увеличьте его через `CMF_MAX_SEQ`. Кэш занимает 115 КБ на токен: 3.7 ГБ на 32k.

### Проверка загрузки

```bash
sha256sum -c mellum2.1-12b-a2.5b-thinking-q4tp.cmf.sha256
cortiq verify mellum2.1-12b-a2.5b-thinking-q4tp.cmf
```

Веса получены из релиза JetBrains и распространяются на условиях Apache-2.0.
Проверяйте сгенерированный код перед запуском.

---

## 中文文档

[JetBrains Mellum2.1-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking)
是一款面向代码的推理模型（混合专家：总参数 121.5 亿，每个 token 激活 25 亿，64 个专家、
top-8 路由；21 个滑动窗口层与 7 个全注意力层；131k 上下文；支持工具调用），打包为独立的
[CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq` 运行——一个不依赖 Python
和机器学习框架的 Rust 推理引擎：NVIDIA、AMD、Intel 显卡走 Vulkan/DX12，Apple silicon
走 Metal，也可在 CPU 上运行。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/Mellum2.1-12B-A2.5B-Thinking-CMF mellum2.1-12b-a2.5b-thinking-q4tp.cmf --local-dir .
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "用 Python 写一个合并两个有序列表的函数。" --no-think
```

建议使用 cortiq 0.8.15 或更新版本；预编译二进制见
[发布页面](https://github.com/infosave2007/cmf/releases)。不加 `--no-think` 时模型会先在
`<think>` 块中思考，请调高 `--max-tokens`（默认 256）。

### 文件

| 文件 | 专家 | 大小 | 代码困惑度 | wiki 困惑度 | top-1 一致率 | 适用场景 |
|---|---|---:|---:|---:|---:|---|
| `mellum2.1-12b-a2.5b-thinking-q4tp.cmf` | 4 位 | 6.88 GB | 3.242 | 7.407 | 93.4% | **默认**：在所有 GPU 上最快，8 GB 显存即可 |
| `mellum2.1-12b-a2.5b-thinking-q8_2f.cmf` | 8 位 | 12.22 GB | 3.146 | 7.180 | 98.1% | 几乎无损；CPU 或 16 GB 以上显存（见下文） |

两个文件的注意力、嵌入层和输出头均为 8 位，路由器和归一化为 16 位。质量以同一引擎运行的未量化
模型为参照（代码困惑度 3.182，wiki 7.120）：在 Python 标准库代码和 wikitext-2 test 的
2048 个 token 上测得困惑度，以及最优预测与未量化模型一致的位置比例。

### 性能

`cortiq bench --core --ignore-eos`，单流，三次运行取中位数，cortiq 0.8.15：

| 硬件 | 后端 | 文件 | 解码 tok/s | 3-4k token 处解码 | 3-4k token 提示词 tok/s | 首 token 时间 |
|---|---|---|---:|---:|---:|---:|
| RTX 3090（24 GB） | Vulkan | `q4tp` | 150 | 129 | 382 | 10.5 s |
| RTX 3090（24 GB） | Vulkan | `q8_2f` | 140 | 132 | 547 | 7.3 s |
| Mac mini M4（24 GB） | Metal | `q4tp` | 46.8 | 40.9 | 413 | 7.3 s |
| Mac mini M4（24 GB） | Metal | `q8_2f` | 35.6 | 31.2 | 318 | 9.8 s |
| Mac mini M4（24 GB） | CPU，10 核 | `q4tp` | 34.9 | 17.3 | 110 | 31.7 s |

RTX 3090 使用 4000 token 提示词，M4 使用 3000。Vulkan 上 `q8_2f` 的解码内核需要 32 通道 subgroup（NVIDIA）；
其他 GPU 上提示词在 GPU 上处理，解码逐算子运行。

### 使用

```bash
cortiq run mellum2.1-12b-a2.5b-thinking-q4tp.cmf --prompt "..." --no-think   # 直接回答
cortiq serve mellum2.1-12b-a2.5b-thinking-q4tp.cmf --port 8080               # OpenAI 兼容 API
```

JetBrains 的示例使用 temperature 0.6、top-p 0.95。工具调用：按 OpenAI 格式传入 `tools`，
服务器会把模型的 `<tool_call>{…}</tool_call>` 转换为 `message.tool_calls`；在 6 个测试对话中
调用和基于结果的回答全部正确（开启与关闭思考均如此）。

### 长上下文

模型支持 131 072 个 token；KV 缓存默认 32 768，可用 `CMF_MAX_SEQ` 调高。缓存每个 token
约占 115 KB：32k 时为 3.7 GB。

权重来自 JetBrains 的发布，遵循其 Apache-2.0 许可。运行生成的代码前请先审阅。
