---
license: apache-2.0
library_name: cortiq
base_model: DRUPEs/CherryPit-1.2-2B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - cmf
  - cortiq
  - quantized
  - code
  - web-development
  - tool-calling
  - vulkan
  - metal
language:
  - en
  - ru
  - zh
---

# CherryPit-1.2-2B — CMF

[CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B) by DRUPEs — a 2.5B coding assistant
fine-tuned from MiniCPM5-2B for web apps, Python, math and tool calling (42 layers, 128k context, a
thinking mode) — packaged as single [CMF](https://github.com/infosave2007/cmf) files. They run with
`cortiq`, a Rust inference engine with no Python and no ML framework: NVIDIA, AMD and Intel GPUs through
Vulkan/DX12, Apple silicon through Metal, and the CPU.

[English](#quick-start) · [Русский](#документация-на-русском) · [中文](#中文文档)

## Quick start

```bash
cargo install cortiq-cli
hf download infosave/CherryPit-1.2-2B-cmf CherryPit-1.2-2B-q8_2f.cmf --local-dir .
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "Build a React component with a search box and a list that loads on open." --no-think --max-tokens 1024
```

Use cortiq 0.8.63 or later; prebuilt binaries for Linux, macOS and Windows are on the
[releases page](https://github.com/infosave2007/cmf/releases). `--no-think` asks for a direct answer.

## Files

| file | quantization | size | WikiText-2 ppl | Python ppl | recommended for |
|---|---|---:|---:|---:|---|
| `CherryPit-1.2-2B-q8_2f.cmf` | 8-bit | 2.54 GB | 18.46 | 5.86 | **default**: the bf16 model's quality |
| `CherryPit-1.2-2B-q4tp.cmf` | 4-bit GPTQ; attention, embeddings and head 8-bit | 1.77 GB | 19.40 | 6.15 | smaller and faster |

Perplexity over 12 windows of 512 tokens: WikiText-2 test, where the bf16 model scores 18.45, and 477 KB
of Python (sources of the `transformers` library). The 4-bit weights are rounded with GPTQ on 4 096
tokens of WikiText-2, code and chats in the model's own template. On 16 coding requests, greedy and two
sampled answers each, neither file answers with a made-up tool call.

For long prompts on little memory: [CherryPit-1.2-2B Turbo](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf).

## Performance

One stream, `cortiq bench --core`; the prompt columns give the time to the first token and the decode
speed after a prompt of that length.

| hardware | backend | file | decode, tok/s | 2k prompt, first token | decode at 2k | 8k prompt, first token | decode at 8k | peak memory |
|---|---|---|---:|---:|---:|---:|---:|---|
| RTX 5090 (32 GB) | Vulkan | `q8_2f` | 190 | 1.3 s | 151 | 7.4 s | 109 | 3.0–4.3 GB VRAM |
| RTX 5090 (32 GB) | Vulkan | `q4tp` | 187 | 1.0 s | 156 | 6.3 s | 124 | 1.9–3.2 GB VRAM |
| MacBook Air M4 (24 GB) | Metal | `q8_2f` | 26.3 | 5.5 s | 24.0 | 33.7 s | 15.5 | 4.1–5.3 GB RAM |
| MacBook Air M4 (24 GB) | Metal | `q4tp` | 42.8 | 4.5 s | 35.9 | 31.4 s | 20.9 | 2.5–3.9 GB RAM |
| MacBook Air M4 (24 GB) | CPU, 10 cores | `q8_2f` | 25.2 | 9.5 s | 25.6 | — | — | 3.1 GB RAM |
| MacBook Air M4 (24 GB) | CPU, 10 cores | `q4tp` | 30.5 | 9.9 s | 25.5 | — | — | 2.3 GB RAM |

The MacBook Air has no fan and the base M4's memory bus; an M4 Pro or Max was not measured. Peak memory
is for the 2k and 8k prompts.

## Usage

```bash
cortiq run CherryPit-1.2-2B-q8_2f.cmf                             # interactive chat
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "..."              # the model decides whether to reason first
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "..." --no-think   # a direct answer
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "..." --greedy     # deterministic output
```

DRUPEs recommends temperature 1.0 and top-p 0.95:

```bash
cortiq run CherryPit-1.2-2B-q8_2f.cmf --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

Without flags the CLI uses temperature 0.7, top-p 0.9, top-k 40, min-p 0.05, repetition penalty 1.1 and
at most 256 new tokens; a reasoning answer needs more `--max-tokens`. The embedded template carries the
model's own system prompt: asked who it is, it answers as CherryPit 1.2. With `--no-think` it also opens
some other answers with that sentence, as the bf16 weights do.

### OpenAI-compatible server

```bash
cortiq serve CherryPit-1.2-2B-q8_2f.cmf --port 8080

curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "CherryPit-1.2-2B-q8_2f", "temperature": 1.0, "top_p": 0.95,
       "messages": [{"role": "user", "content": "Write a debounce function in JavaScript."}]}'
```

`"enable_thinking": false` in the request gives a direct answer. Endpoints: `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, plus a web dashboard on the same port.

### Tool calling

Send `tools` in the OpenAI format; the template puts them in the system prompt the way the model was
trained, and the model calls a tool in XML:

```
<function name="get_weather"><param name="city">Paris</param></function>
```

The server returns it as `message.tool_calls` with `finish_reason: "tool_calls"`. Prompts rendered by
cortiq were checked against `transformers` on seven conversations (system prompt, two turns, tools, thinking
on and off, non-English text): the same tokens in all seven.

## Long context

The model supports 131 072 tokens; the KV cache holds 32 768 by default, about 86 KB a token (2.8 GB).
Raise it with `CMF_MAX_SEQ`:

```bash
CMF_MAX_SEQ=131072 cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt-file long.txt --max-tokens 4096
```

The full 131 072 tokens take 11.3 GB of cache, and decoding slows down as the context grows. The
[Turbo file](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf) holds memory and decode speed
in place on long prompts.

## Verify the download

```bash
sha256sum -c CherryPit-1.2-2B-q8_2f.cmf.sha256
cortiq verify CherryPit-1.2-2B-q8_2f.cmf
cortiq info CherryPit-1.2-2B-q8_2f.cmf
```

Weights: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), from
[DRUPEs/CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B).

---

## Документация на русском

[CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B) от DRUPEs — ассистент для кода на
2,5 млрд параметров, дообученный из MiniCPM5-2B для веб-приложений, Python, математики и вызова
инструментов (42 слоя, контекст 128k, режим рассуждений), — в виде отдельных файлов
[CMF](https://github.com/infosave2007/cmf). Запускается движком `cortiq` на Rust без Python и
ML-фреймворков: GPU NVIDIA, AMD и Intel через Vulkan/DX12, Apple silicon через Metal, а также CPU.

### Быстрый старт

```bash
cargo install cortiq-cli
hf download infosave/CherryPit-1.2-2B-cmf CherryPit-1.2-2B-q8_2f.cmf --local-dir .
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "Build a React component with a search box and a list that loads on open." --no-think --max-tokens 1024
```

Нужен cortiq 0.8.63 или новее; готовые сборки — на [странице релизов](https://github.com/infosave2007/cmf/releases).
`--no-think` даёт прямой ответ.

### Файлы

| файл | квантизация | размер | ppl WikiText-2 | ppl Python | назначение |
|---|---|---:|---:|---:|---|
| `CherryPit-1.2-2B-q8_2f.cmf` | 8 бит | 2,54 ГБ | 18,46 | 5,86 | **по умолчанию**: качество bf16 |
| `CherryPit-1.2-2B-q4tp.cmf` | 4 бита GPTQ; внимание, эмбеддинги и голова 8 бит | 1,77 ГБ | 19,40 | 6,15 | меньше и быстрее |

Перплексия на 12 окнах по 512 токенов: WikiText-2 test (bf16 — 18,45) и 477 КБ кода на Python (исходники
библиотеки `transformers`). 4-битные веса округлены GPTQ на 4 096 токенах WikiText-2, кода и диалогов в
шаблоне самой модели. На 16 задачах по коду (жадный ответ и два сэмплированных на каждую) ни один файл
не отвечает выдуманным вызовом инструмента.

Для длинных промптов при малой памяти: [CherryPit-1.2-2B Turbo](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf).

### Скорость

Один поток, `cortiq bench --core`; в столбцах промпта — время до первого токена и скорость генерации
после промпта такой длины.

| железо | бэкенд | файл | генерация, ток/с | промпт 2k, первый токен | генерация на 2k | промпт 8k, первый токен | генерация на 8k | пик памяти |
|---|---|---|---:|---:|---:|---:|---:|---|
| RTX 5090 (32 ГБ) | Vulkan | `q8_2f` | 190 | 1,3 с | 151 | 7,4 с | 109 | 3,0–4,3 ГБ VRAM |
| RTX 5090 (32 ГБ) | Vulkan | `q4tp` | 187 | 1,0 с | 156 | 6,3 с | 124 | 1,9–3,2 ГБ VRAM |
| MacBook Air M4 (24 ГБ) | Metal | `q8_2f` | 26,3 | 5,5 с | 24,0 | 33,7 с | 15,5 | 4,1–5,3 ГБ RAM |
| MacBook Air M4 (24 ГБ) | Metal | `q4tp` | 42,8 | 4,5 с | 35,9 | 31,4 с | 20,9 | 2,5–3,9 ГБ RAM |
| MacBook Air M4 (24 ГБ) | CPU, 10 ядер | `q8_2f` | 25,2 | 9,5 с | 25,6 | — | — | 3,1 ГБ RAM |
| MacBook Air M4 (24 ГБ) | CPU, 10 ядер | `q4tp` | 30,5 | 9,9 с | 25,5 | — | — | 2,3 ГБ RAM |

У MacBook Air нет вентилятора и шина памяти базового M4; M4 Pro и Max не измерялись. Пик памяти — для
промптов 2k и 8k.

### Использование

```bash
cortiq run CherryPit-1.2-2B-q8_2f.cmf                             # интерактивный чат
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "..."              # модель сама решает, рассуждать ли
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "..." --no-think   # прямой ответ
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "..." --greedy     # детерминированный вывод
```

DRUPEs рекомендует temperature 1.0 и top-p 0.95:

```bash
cortiq run CherryPit-1.2-2B-q8_2f.cmf --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

Встроенный шаблон несёт системный промпт модели: на вопрос «кто ты» она отвечает как CherryPit 1.2; с
`--no-think` она начинает этой фразой и некоторые другие ответы — так же, как веса bf16.

Сервер: `cortiq serve CherryPit-1.2-2B-q8_2f.cmf --port 8080` (OpenAI-совместимый API, `"enable_thinking":
false` — прямой ответ). Инструменты передаются в `tools`; вызов модели в XML
(`<function name="..."><param name="...">...</param></function>`) сервер возвращает как
`message.tool_calls`. Промпты cortiq сверены с `transformers` на семи диалогах: токены совпали во всех.

### Длинный контекст

Модель поддерживает 131 072 токена; кэш KV по умолчанию — 32 768 токенов, около 86 КБ на токен (2,8 ГБ).
Больше — через `CMF_MAX_SEQ=131072` (11,3 ГБ кэша). Для длинных промптов при малой памяти —
[Turbo-файл](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf).

### Проверка загрузки

```bash
sha256sum -c CherryPit-1.2-2B-q8_2f.cmf.sha256
cortiq verify CherryPit-1.2-2B-q8_2f.cmf
```

Веса: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), от
[DRUPEs/CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B).

---

## 中文文档

DRUPEs 的 [CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B) 是一个 25 亿参数的代码助手，
由 MiniCPM5-2B 微调而来，擅长 Web 应用、Python、数学和工具调用（42 层，128k 上下文，支持思考模式），
这里打包为独立的 [CMF](https://github.com/infosave2007/cmf) 文件，由 `cortiq` 运行——一个不依赖 Python
和机器学习框架的 Rust 推理引擎：NVIDIA、AMD、Intel 显卡走 Vulkan/DX12，Apple silicon 走 Metal，也可在
CPU 上运行。

### 快速开始

```bash
cargo install cortiq-cli
hf download infosave/CherryPit-1.2-2B-cmf CherryPit-1.2-2B-q8_2f.cmf --local-dir .
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "Build a React component with a search box and a list that loads on open." --no-think --max-tokens 1024
```

需要 cortiq 0.8.63 或更新版本；预编译二进制见[发布页面](https://github.com/infosave2007/cmf/releases)。
`--no-think` 直接给出回答。

### 文件

| 文件 | 量化 | 大小 | WikiText-2 困惑度 | Python 困惑度 | 适用场景 |
|---|---|---:|---:|---:|---|
| `CherryPit-1.2-2B-q8_2f.cmf` | 8 位 | 2.54 GB | 18.46 | 5.86 | **默认**：与 bf16 质量一致 |
| `CherryPit-1.2-2B-q4tp.cmf` | 4 位 GPTQ；注意力、嵌入层和输出头为 8 位 | 1.77 GB | 19.40 | 6.15 | 更小、更快 |

困惑度基于 12 个 512 token 的窗口：WikiText-2 test（bf16 为 18.45）以及 477 KB 的 Python 代码（`transformers`
库的源码）。4 位权重用 GPTQ 在 4 096 个 token 上取整，数据为 WikiText-2、代码以及模型自身模板中的对话。在
16 个编程请求上（每个一次贪心和两次采样），两个文件都没有以虚构的工具调用作答。

长提示词、内存有限时请用 [CherryPit-1.2-2B Turbo](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf)。

### 性能

单流，`cortiq bench --core`；提示词列给出首 token 时间以及该长度提示词之后的生成速度。

| 硬件 | 后端 | 文件 | 生成，tok/s | 2k 提示词首 token | 2k 处生成 | 8k 提示词首 token | 8k 处生成 | 内存峰值 |
|---|---|---|---:|---:|---:|---:|---:|---|
| RTX 5090（32 GB） | Vulkan | `q8_2f` | 190 | 1.3 s | 151 | 7.4 s | 109 | 3.0–4.3 GB 显存 |
| RTX 5090（32 GB） | Vulkan | `q4tp` | 187 | 1.0 s | 156 | 6.3 s | 124 | 1.9–3.2 GB 显存 |
| MacBook Air M4（24 GB） | Metal | `q8_2f` | 26.3 | 5.5 s | 24.0 | 33.7 s | 15.5 | 4.1–5.3 GB 内存 |
| MacBook Air M4（24 GB） | Metal | `q4tp` | 42.8 | 4.5 s | 35.9 | 31.4 s | 20.9 | 2.5–3.9 GB 内存 |
| MacBook Air M4（24 GB） | CPU，10 核 | `q8_2f` | 25.2 | 9.5 s | 25.6 | — | — | 3.1 GB 内存 |
| MacBook Air M4（24 GB） | CPU，10 核 | `q4tp` | 30.5 | 9.9 s | 25.5 | — | — | 2.3 GB 内存 |

MacBook Air 无风扇，采用基础款 M4 的内存总线；未测试 M4 Pro 或 Max。内存峰值对应 2k 和 8k 提示词。

### 使用

```bash
cortiq run CherryPit-1.2-2B-q8_2f.cmf                             # 交互式对话
cortiq run CherryPit-1.2-2B-q8_2f.cmf --prompt "..." --no-think   # 直接回答
cortiq run CherryPit-1.2-2B-q8_2f.cmf --temperature 1.0 --top-p 0.95 --top-k 0 --min-p 0 --rep-penalty 1.0 --max-tokens 4096 --prompt "..."
```

DRUPEs 推荐 temperature 1.0、top-p 0.95。内置模板带有模型自己的系统提示词：被问到身份时它自称 CherryPit 1.2；
使用 `--no-think` 时，它也会用这句话开头回答其他一些问题，bf16 权重同样如此。

服务端：`cortiq serve CherryPit-1.2-2B-q8_2f.cmf --port 8080`（OpenAI 兼容 API，请求中的
`"enable_thinking": false` 表示直接回答）。工具通过 `tools` 传入；模型以 XML 调用工具，服务端将其返回为
`message.tool_calls`。cortiq 渲染的提示词已在七段对话上与 `transformers` 对照，token 全部一致。

### 长上下文

模型支持 131 072 个 token；KV 缓存默认 32 768 个 token，每个 token 约 86 KB（2.8 GB）。用
`CMF_MAX_SEQ=131072` 可扩展到完整长度（11.3 GB 缓存）。长提示词、内存有限时请用
[Turbo 文件](https://huggingface.co/infosave/CherryPit-1.2-2B-Turbo-cmf)。

### 校验下载

```bash
sha256sum -c CherryPit-1.2-2B-q8_2f.cmf.sha256
cortiq verify CherryPit-1.2-2B-q8_2f.cmf
```

权重：[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)，来自
[DRUPEs/CherryPit-1.2-2B](https://huggingface.co/DRUPEs/CherryPit-1.2-2B)。
